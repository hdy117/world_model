"""Teaching-oriented World Model demo for the Chinese learning guide.

This file defines a tiny CPU-only PyTorch workflow that mirrors the core
World Model loop: encode an observation, predict the next latent state from an
action, decode the next observation, predict reward/continuation, and roll out
future states inside the learned model.  The demo intentionally uses synthetic
low-dimensional dynamics so the code can run without Gym, MuJoCo, image data,
or a GPU.

Run from the repository root:

    python world_model_toy_demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DemoConfig:
    """Configuration for the synthetic World Model experiment.

    The dataclass owns all reproducibility and shape parameters used by the
    data generator, model, training loop, and rollout check.  Keeping these
    values in one object makes it obvious which assumptions the toy example
    depends on.
    """

    # Number of continuous values in each synthetic observation vector.
    state_dim: int = 4
    # Number of continuous values in each synthetic action vector.
    action_dim: int = 2
    # Size of the bottleneck representation used as the learned latent state.
    latent_dim: int = 8
    # Width of the small MLP layers used by encoder, dynamics, and heads.
    hidden_dim: int = 32
    # Number of transitions sampled per optimization step.
    batch_size: int = 64
    # Number of optimizer steps; kept small so the demo is quick on CPU.
    train_steps: int = 160
    # Adam learning rate for all World Model parameters.
    learning_rate: float = 3e-3
    # Discount-like horizon used only to demonstrate imagined rollout length.
    rollout_horizon: int = 6
    # Seed used for deterministic synthetic data and repeatable demo output.
    seed: int = 7


class SyntheticDynamicsDataset:
    """Generates one-step transitions from a fixed hidden linear system.

    This class stands in for a real environment or replay buffer.  It produces
    batches shaped like model-based RL data: observation, action,
    next_observation, reward, and continuation.  The hidden transition matrices
    are private implementation details so the World Model has to learn the
    transition from sampled data rather than reading the true equations.
    """

    def __init__(self, config: DemoConfig) -> None:
        """Create deterministic hidden dynamics for the synthetic task.

        Args:
            config: Shared demo configuration containing shapes and seed.
        """

        self.config = config
        generator = torch.Generator().manual_seed(config.seed)
        # Hidden state transition matrix; scaled to keep rollouts numerically stable.
        self._state_matrix = 0.75 * torch.eye(config.state_dim)
        # Hidden action matrix maps low-dimensional actions into state changes.
        self._action_matrix = torch.randn(
            config.state_dim, config.action_dim, generator=generator
        ) * 0.25
        # Reward vector defines which state directions are considered useful.
        self._reward_vector = torch.randn(config.state_dim, generator=generator)
        # Private generator keeps random sampling reproducible across calls.
        self._generator = generator

    def sample_batch(self, batch_size: int) -> Dict[str, Tensor]:
        """Sample a batch of independent one-step transitions.

        Args:
            batch_size: Number of transitions to generate.

        Returns:
            Dictionary with tensors for observations, actions, next observations,
            rewards, and continuations.  Continuation is 1.0 when the synthetic
            episode should continue and 0.0 when it is treated as terminal.
        """

        observation = torch.randn(
            batch_size, self.config.state_dim, generator=self._generator
        )
        action = torch.randn(batch_size, self.config.action_dim, generator=self._generator)
        next_observation = torch.tanh(
            observation @ self._state_matrix.T + action @ self._action_matrix.T
        )
        reward = (next_observation @ self._reward_vector).unsqueeze(-1)
        # Continuation target gives the model a simple done-like prediction head.
        continuation = (next_observation.norm(dim=-1, keepdim=True) < 1.75).float()
        return {
            "observation": observation,
            "action": action,
            "next_observation": next_observation,
            "reward": reward,
            "continuation": continuation,
        }


class Encoder(nn.Module):
    """Maps raw observations into a compact latent state."""

    def __init__(self, state_dim: int, hidden_dim: int, latent_dim: int) -> None:
        """Build a small MLP encoder.

        Args:
            state_dim: Width of the input observation vector.
            hidden_dim: Width of the hidden layer.
            latent_dim: Width of the learned latent state.
        """

        super().__init__()
        # Sequential MLP is enough for low-dimensional observations in this demo.
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, observation: Tensor) -> Tensor:
        """Encode observations into latent states.

        Args:
            observation: Tensor with shape ``[batch, state_dim]``.

        Returns:
            Latent tensor with shape ``[batch, latent_dim]``.
        """

        return self.network(observation)


class DynamicsModel(nn.Module):
    """Predicts the next latent state from current latent state and action."""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int) -> None:
        """Build the transition model used for imagined rollouts.

        Args:
            latent_dim: Width of the current and next latent vectors.
            action_dim: Width of the action vector.
            hidden_dim: Width of the hidden layer.
        """

        super().__init__()
        # Dynamics consumes both what the model believes and what the agent does.
        self.network = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, latent: Tensor, action: Tensor) -> Tensor:
        """Predict the next latent state.

        Args:
            latent: Current latent tensor with shape ``[batch, latent_dim]``.
            action: Action tensor with shape ``[batch, action_dim]``.

        Returns:
            Predicted next latent tensor with shape ``[batch, latent_dim]``.
        """

        return self.network(torch.cat([latent, action], dim=-1))


class Decoder(nn.Module):
    """Decodes latent states back to observation space for a reconstruction target."""

    def __init__(self, latent_dim: int, hidden_dim: int, state_dim: int) -> None:
        """Build a tiny decoder for next-observation prediction.

        Args:
            latent_dim: Width of the latent input.
            hidden_dim: Width of the hidden layer.
            state_dim: Width of the reconstructed observation vector.
        """

        super().__init__()
        # Decoder makes the toy model easy to inspect because predictions live in state space.
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, latent: Tensor) -> Tensor:
        """Decode latent states into predicted observations.

        Args:
            latent: Tensor with shape ``[batch, latent_dim]``.

        Returns:
            Predicted observation tensor with shape ``[batch, state_dim]``.
        """

        return self.network(latent)


class PredictionHead(nn.Module):
    """Predicts a scalar target such as reward or continuation."""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int) -> None:
        """Build a scalar head conditioned on latent state and action.

        Args:
            latent_dim: Width of the latent state.
            action_dim: Width of the action vector.
            hidden_dim: Width of the hidden layer.
        """

        super().__init__()
        # The same structure is reused for reward and continuation targets.
        self.network = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent: Tensor, action: Tensor) -> Tensor:
        """Predict a scalar from latent state and action.

        Args:
            latent: Current latent tensor with shape ``[batch, latent_dim]``.
            action: Action tensor with shape ``[batch, action_dim]``.

        Returns:
            Scalar prediction with shape ``[batch, 1]``.
        """

        return self.network(torch.cat([latent, action], dim=-1))


class WorldModel(nn.Module):
    """Composes encoder, dynamics, decoder, reward head, and continuation head."""

    def __init__(self, config: DemoConfig) -> None:
        """Create all trainable modules used by the toy World Model.

        Args:
            config: Shared demo configuration with model dimensions.
        """

        super().__init__()
        # Encoder is the public boundary from observations into latent space.
        self.encoder = Encoder(config.state_dim, config.hidden_dim, config.latent_dim)
        # Dynamics is the part that lets the model imagine future latent states.
        self.dynamics = DynamicsModel(
            config.latent_dim, config.action_dim, config.hidden_dim
        )
        # Decoder provides a simple next-state prediction loss for this demo.
        self.decoder = Decoder(config.latent_dim, config.hidden_dim, config.state_dim)
        # Reward head predicts the scalar signal used for planning or policy learning.
        self.reward_head = PredictionHead(
            config.latent_dim, config.action_dim, config.hidden_dim
        )
        # Continuation head predicts whether rollout should keep going.
        self.continuation_head = PredictionHead(
            config.latent_dim, config.action_dim, config.hidden_dim
        )

    def forward(self, observation: Tensor, action: Tensor) -> Dict[str, Tensor]:
        """Run one learned transition step.

        Args:
            observation: Current observation tensor with shape ``[batch, state_dim]``.
            action: Current action tensor with shape ``[batch, action_dim]``.

        Returns:
            Dictionary containing current latent, predicted next latent,
            predicted next observation, reward logits, and continuation logits.
        """

        latent = self.encoder(observation)
        next_latent = self.dynamics(latent, action)
        return {
            "latent": latent,
            "next_latent": next_latent,
            "next_observation": self.decoder(next_latent),
            "reward": self.reward_head(latent, action),
            "continuation_logit": self.continuation_head(latent, action),
        }

    def rollout(self, observation: Tensor, actions: Tensor) -> Tensor:
        """Imagine a sequence of future observations without touching a real environment.

        Args:
            observation: Initial observation with shape ``[batch, state_dim]``.
            actions: Planned actions with shape ``[horizon, batch, action_dim]``.

        Returns:
            Predicted observations with shape ``[horizon, batch, state_dim]``.
        """

        latent = self.encoder(observation)
        predictions = []
        for action in actions:
            # Each step feeds the previous imagined latent into the learned dynamics.
            latent = self.dynamics(latent, action)
            predictions.append(self.decoder(latent))
        return torch.stack(predictions, dim=0)


def compute_loss(model: WorldModel, batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, float]]:
    """Compute World Model losses for one synthetic batch.

    Args:
        model: Trainable World Model.
        batch: Transition batch produced by ``SyntheticDynamicsDataset``.

    Returns:
        Total loss tensor and a detached metrics dictionary for logging.
    """

    prediction = model(batch["observation"], batch["action"])
    next_latent_target = model.encoder(batch["next_observation"]).detach()
    state_loss = F.mse_loss(prediction["next_observation"], batch["next_observation"])
    latent_loss = F.mse_loss(prediction["next_latent"], next_latent_target)
    reward_loss = F.mse_loss(prediction["reward"], batch["reward"])
    continuation_loss = F.binary_cross_entropy_with_logits(
        prediction["continuation_logit"], batch["continuation"]
    )
    total = state_loss + latent_loss + reward_loss + 0.2 * continuation_loss
    metrics = {
        "total": float(total.detach()),
        "state": float(state_loss.detach()),
        "latent": float(latent_loss.detach()),
        "reward": float(reward_loss.detach()),
        "continuation": float(continuation_loss.detach()),
    }
    return total, metrics


def train_demo(config: DemoConfig) -> Tuple[WorldModel, Dict[str, float]]:
    """Train the toy World Model for a small number of CPU steps.

    Args:
        config: Shared demo configuration.

    Returns:
        The trained model and final scalar metrics.
    """

    torch.manual_seed(config.seed)
    dataset = SyntheticDynamicsDataset(config)
    model = WorldModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    final_metrics: Dict[str, float] = {}
    for _ in range(config.train_steps):
        batch = dataset.sample_batch(config.batch_size)
        loss, final_metrics = compute_loss(model, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping keeps this demo stable if dimensions are changed.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
    return model, final_metrics


def verify_rollout(model: WorldModel, config: DemoConfig) -> Tuple[int, Tuple[int, ...]]:
    """Run a deterministic imagined rollout shape check.

    Args:
        model: Trained World Model.
        config: Shared demo configuration with rollout horizon and dimensions.

    Returns:
        Rollout horizon and the full output tensor shape.
    """

    observation = torch.zeros(1, config.state_dim)
    actions = torch.zeros(config.rollout_horizon, 1, config.action_dim)
    imagined_observations = model.rollout(observation, actions)
    return config.rollout_horizon, tuple(imagined_observations.shape)


def main() -> None:
    """Run training and print a compact verification report."""

    config = DemoConfig()
    model, metrics = train_demo(config)
    horizon, rollout_shape = verify_rollout(model, config)
    print("World Model toy demo finished.")
    print(f"final_loss={metrics['total']:.4f}")
    print(f"state_loss={metrics['state']:.4f}")
    print(f"reward_loss={metrics['reward']:.4f}")
    print(f"rollout_horizon={horizon}")
    print(f"rollout_shape={rollout_shape}")


if __name__ == "__main__":
    main()

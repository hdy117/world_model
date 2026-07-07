# World Model 运行时序设计

## 设计思想

时序设计说明各对象在训练、规划和评估中的调用顺序。核心原则是：`Trainer` 负责改变模型参数，`Planner` 只负责选择 action，`WorldModel` 只负责编码和想象，`Evaluator` 只负责真实环境验证。这样 Developer 阶段写代码时不会把训练、规划、评估逻辑混在一起。

## 总体流程图

```mermaid
sequenceDiagram
    actor Script
    participant Trainer
    participant Env as Environment
    participant Buffer as ReplayBuffer
    participant WM as WorldModel
    participant Opt as Optimizer
    participant Planner
    participant Evaluator

    Script->>Trainer: fit(num_steps)

    loop data collection
        Trainer->>Env: reset()
        Env-->>Trainer: observation
        Trainer->>Env: step(action)
        Env-->>Trainer: next_observation, reward, done
        Trainer->>Buffer: add(Transition)
    end

    loop model training
        Trainer->>Buffer: sample(batch_size)
        Buffer-->>Trainer: batch
        Trainer->>WM: encode / imagine / predict_reward
        WM-->>Trainer: predictions
        Trainer->>Trainer: compute losses
        Trainer->>Opt: step()
        Opt-->>WM: updated parameters
    end

    Script->>Evaluator: evaluate(num_episodes)

    loop evaluation episode
        Evaluator->>Env: reset()
        Env-->>Evaluator: observation
        Evaluator->>Planner: select_action(observation, WM)
        Planner->>WM: imagine(candidate action sequences)
        WM-->>Planner: imagined trajectories and rewards
        Planner-->>Evaluator: selected action
        Evaluator->>Env: step(selected action)
        Env-->>Evaluator: next_observation, reward, done
    end

    Evaluator-->>Script: metrics
```

## 训练时序

| 顺序 | 调用方 | 被调用方 | 目的 | 是否修改状态 |
|---|---|---|---|---|
| 1 | `Script` | `Trainer.fit` | 启动训练 | 是 |
| 2 | `Trainer` | `Environment.reset/step` | 收集真实经验 | 环境状态变化 |
| 3 | `Trainer` | `ReplayBuffer.add` | 保存 transition | buffer 变化 |
| 4 | `Trainer` | `ReplayBuffer.sample` | 获取 batch | 否 |
| 5 | `Trainer` | `WorldModel` | 生成 latent/reward predictions | 否 |
| 6 | `Trainer` | private loss helper | 计算教学 loss | 否 |
| 7 | `Trainer` | optimizer | 更新模型参数 | 模型参数变化 |

## 规划时序

| 顺序 | 调用方 | 被调用方 | 目的 | 约束 |
|---|---|---|---|---|
| 1 | `Evaluator` 或 collection loop | `Planner.select_action` | 请求 action | 不更新模型 |
| 2 | `Planner` | `WorldModel.encode` | 得到 current latent | 不推进 env |
| 3 | `Planner` | candidate sampler | 生成候选 action sequence | seed 可控 |
| 4 | `Planner` | `WorldModel.imagine` | 模型内 rollout | 只使用 learned model |
| 5 | `Planner` | scoring helper | 累计 predicted reward | 可使用 discount |
| 6 | `Planner` | return best first action | 输出真实 action | 不写 buffer |

## 评估时序

| 顺序 | 调用方 | 被调用方 | 目的 |
|---|---|---|---|
| 1 | `Script` | `Evaluator.evaluate` | 运行若干 episode |
| 2 | `Evaluator` | `Environment.reset` | 初始化真实环境 |
| 3 | `Evaluator` | `Planner.select_action` | 基于 world model 决策 |
| 4 | `Evaluator` | `Environment.step` | 在真实环境执行 |
| 5 | `Evaluator` | metrics accumulator | 统计 return、length、success |
| 6 | `Evaluator` | return metrics | 输出验收指标 |

## 状态变化边界

| 组件 | 可变状态 | 谁可以修改 |
|---|---|---|
| `Environment` | current state, step count | `reset`, `step` |
| `ReplayBuffer` | stored transitions | `Trainer.collect_episode` |
| `WorldModel` | model parameters | `Trainer.train_step` via optimizer |
| `Planner` | horizon、candidate count、seed | 构造或配置阶段 |
| `Evaluator` | temporary episode metrics | `Evaluator.run_episode` |

## 验收检查点

| 检查 | 通过标准 |
|---|---|
| 训练和评估分离 | `Evaluator` 不调用 optimizer |
| 规划和环境分离 | `WorldModel.imagine` 不调用 `Environment.step` |
| 数据收集清晰 | transition 只由真实 env 交互产生 |
| 图表一致 | 类名与 `class_design.md` 完全一致 |
| 教学边界清晰 | 明确 random shooting 是最小 planner，不是 MuZero MCTS 或 Dreamer actor-critic |

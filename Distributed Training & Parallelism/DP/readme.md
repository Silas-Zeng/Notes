# Data Parallelism（DP）数据并行

## 1. 基本原理

数据并行的核心思想是：将完整的模型复制到多张 GPU 上，每张 GPU 处理不同的训练数据，通过梯度同步保证模型参数的一致性。

假设有 N 张 GPU，每张 GPU 分别执行 Forward 和 Backward，得到局部梯度：

$$
g_i=\nabla_\theta L_i(\theta)
$$

当各 GPU 的局部 Loss 使用相同数量的样本计算平均值时，全局梯度为：

$$
g=\frac{1}{N}\sum_{i=1}^{N}g_i
$$

最后使用全局梯度更新参数：

$$
\theta_{t+1}=\theta_t-\eta g
$$

因此，多张 GPU 处理不同数据并同步梯度，在数学上等价于单张 GPU 处理相同的全局 Batch。

## 2. DP 与 DDP

DP 是一种并行策略，而 PyTorch 中的 DataParallel 和 DistributedDataParallel（DDP）是两种具体实现。

| | DataParallel | DDP |
|---|---|---|
| 进程 | 单进程、多线程 | 多进程 |
| 梯度同步 | 归约到主卡 | All-Reduce |
| 参数更新 | 主卡更新 | 各卡独立更新 |
| 多机训练 | 不支持 | 支持 |

DDP 的训练流程：

1. 每张 GPU 保存完整的模型。
2. 各 GPU 独立执行 Forward。
3. 各 GPU 独立执行 Backward。
4. 通过 All-Reduce 同步梯度。
5. 各 GPU 使用相同的梯度更新参数。

由于每张 GPU 的初始参数和归约后的梯度相同，因此更新后的模型参数也保持一致。

## 3. 通信机制

DDP 主要使用 All-Reduce 进行梯度同步。

经典 Ring All-Reduce 可以分解为两个阶段：

$$
\text{All-Reduce}=\text{Reduce-Scatter}+\text{All-Gather}
$$

- Reduce-Scatter：对梯度进行归约，每张 GPU 获得全局梯度的一部分。
- All-Gather：收集各 GPU 的梯度分片，恢复完整的全局梯度。

假设完整梯度的数据大小为 $\Phi$ Bytes，GPU 数量为 $N$，则 Ring All-Reduce 的单卡发送量为：

$$
V_{\mathrm{RS}}=\frac{N-1}{N}\Phi
$$

$$
V_{\mathrm{AG}}=\frac{N-1}{N}\Phi
$$

因此：

$$
V_{\mathrm{AR}}=2\frac{N-1}{N}\Phi
$$

当 GPU 数量足够大时，通信量近似为 $2\Phi$。

## 4. DP 的性能优化

### Gradient Accumulation（梯度累积）

将一个较大的 Batch 拆分成多个 Microbatch，分别进行前向和反向传播，累积梯度后再统一更新参数。

$$
GBS=MBS\times DP\times GAS
$$

通过 DDP 的 `no_sync()`，可以在前几个 Microbatch 中暂时关闭梯度同步，只在最后一个 Microbatch 进行梯度归约，从而减少通信次数。

### Gradient Bucket（梯度分桶）

将多个参数的梯度放入同一个 Bucket，合并执行 All-Reduce，避免大量小规模通信产生的启动开销。

### Communication-Computation Overlap（通信计算重叠）

反向传播过程中，不必等待所有梯度计算完成。

当一个 Bucket 中的梯度全部就绪后，即可启动对应的 All-Reduce，同时继续计算其他层的梯度，从而隐藏部分通信开销。

## 5. DP 的局限性

DDP 中，每张 GPU 都需要保存完整的模型参数、梯度和优化器状态，存在大量显存冗余。

增加 GPU 数量虽然能够提高数据处理能力，却不能直接降低单卡模型状态的显存占用。

为了解决这一问题，后续引入了 ZeRO 和 FSDP，通过对模型状态进行分片，进一步降低显存占用。

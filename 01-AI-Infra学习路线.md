# AI Infrastructure 学习路线

> 让模型跑得更快、更省、更稳 —— 这是AI落地的最后一公里

---

## 📌 方向定位

AI Infra 是连接算法与工程的桥梁，核心能力包括：
- 分布式训练（多卡/多机协同）
- 推理优化（量化、加速、部署）
- GPU资源管理（显存、调度、监控）
- 训练稳定性（容错、恢复、调试）

**适合人群**：有深度学习基础，熟悉PyTorch，想往架构/平台方向发展的算法工程师

---

## 🗺️ 学习路线

### 第一阶段：GPU编程基础（4-6周）

#### 目标
- 理解GPU硬件架构
- 能写基础CUDA程序
- 会用profiling工具分析性能

#### 学习内容

**1. GPU架构（第1-2周）**

| 概念 | 说明 | 重要性 |
|------|------|:------:|
| SM（流多处理器） | GPU的基本计算单元 | ⭐⭐⭐⭐⭐ |
| CUDA Core | 浮点/整数运算单元 | ⭐⭐⭐⭐ |
| Tensor Core | 矩阵运算专用单元（FP16/INT8） | ⭐⭐⭐⭐⭐ |
| HBM/GDDR | 显存类型与带宽 | ⭐⭐⭐⭐⭐ |
| Warp | 32个线程的执行单元 | ⭐⭐⭐⭐ |
| 共享内存 | 片上高速存储 | ⭐⭐⭐⭐ |
| L1/L2 Cache | 缓存层次 | ⭐⭐⭐ |

**2. CUDA编程（第2-4周）**

```python
# 最小CUDA示例：向量加法
import torch

a = torch.randn(1024, device='cuda')
b = torch.randn(1024, device='cuda')
c = a + b  # 底层调用CUDA kernel
```

必须掌握的知识点：
- 线程层次：Thread → Block → Grid
- 内存层次：Global Memory → Shared Memory → Registers
- 同步机制：`__syncthreads()`
- 常见优化：合并访存（Coalesced Access）、避免分支发散

**3. Profiling工具（第4-6周）**

| 工具 | 用途 | 命令示例 |
|------|------|---------|
| `torch.profiler` | PyTorch性能分析 | `with torch.profiler.profile() as prof:` |
| `nsight systems` | 系统级性能分析 | `nsys profile python train.py` |
| `nsight compute` | Kernel级性能分析 | `ncu --set full python train.py` |
| `torch.cuda.memory_summary()` | 显存分析 | 训练中调用 |
| `nvtop` | GPU实时监控 | 终端运行 |

#### 实操项目
```
项目：单卡GPU性能分析
1. 用PyTorch训练ResNet-50（ImageNet或CIFAR-10）
2. 用nsight systems做trace分析
3. 找出训练pipeline中的瓶颈（数据加载？计算？通信？）
4. 优化数据加载（num_workers、pin_memory、prefetch）
5. 对比优化前后的吞吐量
```

---

### 第二阶段：分布式训练（8-12周）

#### 目标
- 掌握主流分布式训练范式
- 能在多卡/多机上训练大模型
- 理解通信原语和优化策略

#### 学习内容

**1. 数据并行（Data Parallelism）**

```
原理：每个GPU持有完整模型副本，数据切分到各GPU
适用：模型能放进单卡显存
```

| 技术 | 特点 | 适用场景 |
|------|------|---------|
| DDP（DataParallel） | PyTorch原生，简单 | 单机多卡 |
| FSDP（FullyShardedDP） | PyTorch原生，分片参数 | 大模型 |
| DeepSpeed ZeRO Stage 1 | 优化器状态分片 | 中等模型 |
| DeepSpeed ZeRO Stage 2 | +梯度分片 | 较大模型 |
| DeepSpeed ZeRO Stage 3 | +参数分片 | 超大模型 |

**ZeRO三阶段对比：**
```
Stage 1: 分片优化器状态 → 显存节省 4x
Stage 2: +分片梯度 → 显存节省 8x
Stage 3: +分片参数 → 显存节省 N x（N=GPU数）
```

**2. 模型并行（Model Parallelism）**

```
张量并行（Tensor Parallelism）：
- 将单个层切分到多个GPU
- 适用于超大矩阵运算
- 需要高速互联（NVLink）

流水线并行（Pipeline Parallelism）：
- 将模型按层切分到不同GPU
- GPU间流水线执行
- 需要micro-batch减少bubble
```

**3. 3D并行（混合并行）**

```
大模型训练的终极方案：
- 数据并行 + 张量并行 + 流水线并行
- 例：64卡 = 8路数据并行 × 4路张量并行 × 2路流水线并行
```

**4. 通信原语**

| 原语 | 说明 | 使用场景 |
|------|------|---------|
| AllReduce | 所有节点聚合结果 | DDP梯度同步 |
| AllGather | 所有节点收集完整数据 | FSDP参数恢复 |
| ReduceScatter | 聚合并分发 | FSDP梯度分片 |
| AllToAll | 全交换 | MoE专家路由 |
| Point-to-Point | 点对点通信 | 流水线并行 |

**5. 关键配置项**

```python
# DeepSpeed ZeRO-3 配置示例
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"},
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6
    },
    "bf16": {"enabled": true},
    "train_batch_size": 64,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 8
}
```

#### 实操项目
```
项目1：多卡训练对比实验
1. 用DDP在2卡/4卡上训练GPT-2 Small
2. 用DeepSpeed ZeRO-2训练GPT-2 Medium
3. 用DeepSpeed ZeRO-3 + CPU Offload训练GPT-2 Large
4. 对比各方案的训练速度、显存占用、吞吐量

项目2：训练一个1.3B参数模型
1. 准备中文语料（WuDaoCorpora子集）
2. 搭建GPT-2架构（1.3B参数）
3. 用DeepSpeed ZeRO-3在4卡上预训练
4. 记录loss曲线，分析训练稳定性
```

---

### 第三阶段：推理优化（8-10周）

#### 目标
- 掌握模型量化技术
- 能部署高吞吐推理服务
- 理解推理系统架构

#### 学习内容

**1. 量化技术**

| 方法 | 原理 | 优势 | 劣势 |
|------|------|------|------|
| INT8量化 | 8bit整数推理 | 2x加速 | 精度损失 |
| INT4量化 | 4bit整数推理 | 4x加速 | 精度损失较大 |
| GPTQ | 基于Hessian的后训练量化 | 精度好 | 需要校准数据 |
| AWQ | 激活感知权重量化 | 更好精度 | 计算量稍大 |
| GGUF | llama.cpp格式，CPU友好 | 本地部署 | GPU利用率低 |
| SmoothQuant | 平滑激活值分布 | 训练后量化 | 需要层分析 |

**2. 推理引擎**

| 引擎 | 特点 | 适用场景 |
|------|------|---------|
| **vLLM** | PagedAttention，高吞吐 | 服务端部署 |
| **TensorRT-LLM** | NVIDIA官方，极致优化 | 生产环境 |
| **llama.cpp** | CPU推理，GGUF格式 | 本地/边缘部署 |
| **SGLang** | RadixAttention，结构化生成 | 复杂推理 |
| **Ollama** | 一键部署，用户体验好 | 快速验证 |
| **DeepSpeed-Inference** | 与DeepSpeed集成 | 大规模推理 |

**3. 关键优化技术**

```
Attention优化：
├── Flash Attention（IO感知的精确注意力）
├── PagedAttention（虚拟内存管理KV Cache）
├── MQA/GQA（减少KV头数量）
└── Sliding Window Attention（滑动窗口）

批处理优化：
├── Continuous Batching（动态批处理）
├── Speculative Decampling（投机解码）
├── Prefix Caching（前缀缓存）
└── Chunked Prefill（分块预填充）

显存优化：
├── KV Cache量化
├── 激活重计算（Activation Checkpointing）
├── CPU Offloading
└── 模型分片推理
```

**4. 部署架构**

```
典型生产部署架构：
用户请求 → 负载均衡 → API网关 → vLLM集群 → GPU节点
                                    ↓
                              监控/日志/限流
```

#### 实操项目
```
项目1：推理性能基准测试
1. 用vLLM部署Qwen-7B（你有4070 SUPER，12GB VRAM）
2. 用不同量化方案（FP16/INT8/INT4）分别部署
3. 用locust做压力测试，记录吞吐量和延迟
4. 对比不同配置的性能差异

项目2：构建私有化推理服务
1. 用TensorRT-LLM量化并部署一个模型
2. 写一个FastAPI包装层，实现OpenAI兼容接口
3. 加入认证、限流、日志
4. 部署为Docker服务
```

---

### 第四阶段：生产级系统（6-8周）

#### 目标
- 能设计和维护大规模AI训练/推理集群
- 掌握故障排查和成本优化

#### 学习内容

**1. 集群管理**
- Kubernetes + GPU调度（NVIDIA GPU Operator）
- Volcano（批处理调度器）
- 弹性训练（Elastic Training）

**2. 训练稳定性**
- Loss Spike排查流程
- 梯度爆炸/消失处理
- 混合精度训练的Loss Scaling
- Checkpoint管理与恢复

**3. 成本优化**
- GPU利用率监控与优化
- Spot/竞价实例策略
- 训练任务调度（错峰执行）
- 推理成本分析（$/1M tokens）

---

## 📚 基础要求

### 前置技能
- [ ] Python熟练（不是会写，是精通）
- [ ] PyTorch基础（能自己写训练循环）
- [ ] 深度学习基础（CNN/RNN/Transformer）
- [ ] Linux命令行操作
- [ ] Git版本管理

### 硬件要求
| 阶段 | 最低配置 | 推荐配置 |
|------|---------|---------|
| 第一阶段 | 1x RTX 3060 12GB | 1x RTX 4070 SUPER 12GB |
| 第二阶段 | 2x RTX 3090 24GB | 4x A100 80GB |
| 第三阶段 | 1x RTX 4070 SUPER | 1x A100 80GB |
| 第四阶段 | 云GPU（按需） | 集群环境 |

### 数学基础
- 线性矩阵运算（特征值、SVD）
- 概率统计（贝叶斯、分布）
- 优化理论（梯度下降、Adam）

---

## 📖 学习资源

### 书籍
| 书名 | 作者 | 说明 |
|------|------|------|
| 《Programming Massively Parallel Processors》 | Kirk & Hwu | CUDA编程经典 |
| 《CUDA Programming: A Developer's Guide》 | Sanders & Kandrot | CUDA实战 |
| 《High Performance Computing》 | Severance | HPC入门 |

### 在线课程
| 课程 | 平台 | 说明 |
|------|------|------|
| CUDA Programming | NVIDIA DLI | 官方CUDA课程 |
| Systems for ML | Stanford | CS329S，系统设计 |
| Parallel Computing | Coursera | GPU编程入门 |

### 官方文档（必读）
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch Distributed Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [DeepSpeed Documentation](https://www.deepspeed.ai/tutorials/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)

### GitHub仓库
- [deepspeed-examples](https://github.com/microsoft/DeepSpeedExamples) — DeepSpeed示例
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — NVIDIA分布式训练
- [FSDP examples](https://github.com/pytorch/examples/tree/main/distributed) — PyTorch FSDP
- [vLLM](https://github.com/vllm-project/vllm) — 推理引擎
- [SGLang](https://github.com/sgl-project/sglang) — 高性能推理

### 社区/博客
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog)
- [Lil'Log (Lilian Weng)](https://lilianweng.github.io) — 深度技术博客
- [Hugging Face Blog](https://huggingface.co/blog)
- 知乎「AI Infra」话题
- Reddit r/MachineLearning

---

## 📄 关键论文

### CUDA/GPU架构
| 论文 | 年份 | 核心贡献 |
|------|:----:|---------|
| A Survey of GPU Computing | 2023 | GPU计算全景综述 |

### 分布式训练
| 论文 | 年份 | 核心贡献 |
|------|:----:|---------|
| ZeRO: Memory Optimizations | 2020 | DeepSpeed ZeRO三阶段 |
| Megatron-LM | 2020 | 大规模Transformer训练 |
| GPipe | 2019 | 流水线并行 |
| PipeDream | 2019 | 异步流水线并行 |
| Ring-AllReduce | 2017 | 环形通信原语 |
| PyTorch FSDP | 2023 | PyTorch原生全分片 |
| DeepSeek-V3 Infrastructure | 2024 | MoE训练系统设计 |

### 推理优化
| 论文 | 年份 | 核心贡献 |
|------|:----:|---------|
| Flash Attention | 2022 | IO感知精确注意力 |
| Flash Attention 2 | 2023 | 优化并行和工作分配 |
| PagedAttention (vLLM) | 2023 | 虚拟内存管理KV Cache |
| GPTQ | 2022 | LLM后训练量化 |
| AWQ | 2023 | 激活感知权重量化 |
| SmoothQuant | 2023 | 平滑激活值量化 |
| Speculative Decoding | 2023 | 投机解码加速 |
| Continuous Batching | 2023 | 动态批处理 |
| TensorRT-LLM | 2023 | NVIDIA推理优化引擎 |
| SGLang RadixAttention | 2024 | 结构化生成加速 |

### 系统设计
| 论文 | 年份 | 核心贡献 |
|------|:----:|---------|
| PaLM: Scaling Language Modeling | 2022 | Google大规模训练系统 |
| BLOOM: BigScience 176B | 2022 | 多国协作训练176B模型 |
| Megatron-Turing NLG | 2022 | 530B模型训练 |

---

## 🔧 实操项目清单（按阶段）

### P0：入门必做（1-2个月）
1. ✅ 用CUDA写一个向量加法程序
2. ✅ 用nsight systems分析ResNet训练性能
3. ✅ 用DDP在2卡上训练一个简单模型
4. ✅ 用DeepSpeed ZeRO-3训练GPT-2 Small

### P1：核心技能（3-6个月）
5. ✅ 用vLLM部署一个7B模型，压测性能
6. ✅ 用GPTQ/AWQ量化一个13B模型
7. ✅ 用TensorRT-LLM部署并对比性能
8. ✅ 搭建完整的推理API服务（FastAPI + 认证 + 监控）

### P2：进阶能力（6-12个月）
9. ✅ 设计并实现一个弹性训练系统
10. ✅ 搭建K8s + GPU调度集群
11. ✅ 完成一个端到端的大模型训练Pipeline
12. ✅ 输出技术博客/开源项目

---

## ⚠️ 常见坑与注意事项

1. **不要跳过CUDA基础** — 很多人直接跳到DeepSpeed，但不理解底层，排查问题时很痛苦
2. **显存是第一瓶颈** — 养成随时监控显存的习惯（`nvidia-smi`、`torch.cuda.memory_summary()`）
3. **通信是性能杀手** — 多卡训练时，通信开销可能占50%以上，要学会分析通信/计算比例
4. **BF16 > FP16** — 如果硬件支持（A100/H100），优先用BF16，不容易overflow
5. **从单卡开始** — 先确保单卡训练正确，再扩展到多卡
6. **看loss曲线** — 训练中最直观的健康指标，loss不降/震荡/spike都要警惕

---

## 📅 推荐时间线

| 周数 | 内容 | 产出 |
|:----:|------|------|
| 1-2 | GPU架构学习 | 笔记/思维导图 |
| 3-4 | CUDA编程入门 | 向量加法/矩阵乘CUDA实现 |
| 5-6 | Profiling工具 | ResNet性能分析报告 |
| 7-10 | 分布式训练基础 | DDP/FSDP训练脚本 |
| 11-14 | DeepSpeed实战 | ZeRO-3训练GPT-2 1.3B |
| 15-18 | 推理优化基础 | vLLM部署+压测报告 |
| 19-22 | 量化技术 | GPTQ/AWQ量化实验 |
| 23-26 | 推理系统搭建 | 私有化推理API服务 |
| 27-30 | 生产级系统 | K8s部署+监控告警 |
| 31-34 | 综合项目 | 端到端训练+部署Pipeline |
| 35-36 | 总结输出 | 技术博客/开源项目 |

---

> 💡 **核心理念**：AI Infra不是一个孤立的方向，它是所有AI应用的底座。掌握它，无论未来做训练、做推理、做产品，都能比别人快一步。

# Agent开发学习路线

> 让LLM从"能聊天"进化到"能干活" —— 规划、工具调用、多步推理、自主决策

---

## 📌 方向定位

Agent是当前AI应用层最热门的方向，核心能力包括：
- Prompt Engineering（不是简单提问，是系统设计）
- Function Calling（让LLM调用外部工具）
- Planning & Reasoning（任务分解与推理）
- Memory系统（短期+长期记忆）
- Multi-Agent（多智能体协作）

**适合人群**：有编程基础，对AI应用感兴趣的开发者/算法工程师

---

---

## 🗺️ 学习路线

### 第一阶段：LLM基础与Prompt Engineering（2-3周）

#### 目标
- 熟练使用LLM API
- 掌握高级Prompt技巧
- 理解LLM的能力边界

#### 学习内容

**1. API使用**

```python
# OpenAI API基础
from openai import OpenAI

client = OpenAI()

# 基础对话
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "你是一个专业的AI助手"},
        {"role": "user", "content": "什么是机器学习？"}
    ],
    temperature=0.7,
    max_tokens=1000
)

# Function Calling
tools = [{
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "搜索网页获取最新信息",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"}
            },
            "required": ["query"]
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    tools=tools,
    tool_choice="auto"
)
```

**2. Prompt Engineering进阶**

| 技术 | 原理 | 适用场景 |
|------|------|---------|
| **Chain-of-Thought (CoT)** | 让模型"一步一步思考" | 复杂推理 |
| **Few-shot** | 给几个示例 | 格式控制 |
| **Zero-shot CoT** | "Let's think step by step" | 通用推理 |
| **Self-Consistency** | 多次采样取多数 | 提高准确率 |
| **Tree-of-Thought** | 树状搜索 | 复杂决策 |
| **ReAct** | Reasoning + Acting | Agent核心模式 |
| **Reflexion** | 自我反思纠错 | 迭代优化 |

**3. LLM能力边界**

```
LLM擅长：
├── 文本生成/改写/翻译
├── 信息提取/总结
├── 简单推理（1-3步）
├── 代码生成（常见场景）
└── 格式化输出

LLM不擅长：
├── 实时信息获取（需要工具）
├── 精确数学计算（需要工具）
├── 大规模数据处理（需要工具）
├── 长期记忆（需要外部存储）
└── 物理世界交互（需要执行器）
```

#### 实操项目
```
项目1：智能问答机器人
1. 用OpenAI API写一个聊天机器人
2. 实现System Prompt设计
3. 加入Few-shot示例
4. 测试不同temperature的效果

项目2：CoT推理实验
1. 用CoT解决数学应用题
2. 对比直接回答 vs CoT的效果
3. 实现Self-Consistency
```

---

### 第二阶段：Function Calling与工具使用（3-4周）

#### 目标
- 掌握Function Calling的完整流程
- 能设计和实现工具系统
- 理解工具编排策略

#### 学习内容

**1. Function Calling机制**

```
用户请求 → LLM判断是否需要工具 → 生成工具调用 → 执行工具 → 结果返回LLM → 生成最终回答

关键点：
├── 工具描述要清晰（LLM靠描述决定是否调用）
├── 参数类型要准确（减少LLM生成错误参数）
├── 错误处理要完善（工具调用可能失败）
└── 多轮工具调用（可能需要多次调用不同工具）
```

**2. 工具设计原则**

| 原则 | 说明 | 示例 |
|------|------|------|
| 单一职责 | 一个工具做一件事 | 搜索和读取分开 |
| 描述清晰 | LLM靠描述理解用途 | `description`写详细 |
| 参数简洁 | 减少LLM出错概率 | 必需参数+可选参数 |
| 返回结构化 | JSON格式返回 | 方便LLM解析 |
| 幂等设计 | 重复调用结果一致 | 避免副作用 |

**3. 常用工具类型**

```python
# 工具分类
tools = {
    # 信息获取
    "web_search": "搜索网页",
    "arxiv_search": "搜索论文",
    "wikipedia": "查询百科",
    
    # 文件操作
    "read_file": "读取文件",
    "write_file": "写入文件",
    "list_directory": "列出目录",
    
    # 代码执行
    "python_repl": "执行Python代码",
    "shell_command": "执行Shell命令",
    
    # 数据处理
    "sql_query": "执行SQL查询",
    "pandas_transform": "数据转换",
    
    # 外部API
    "weather_api": "获取天气",
    "email_send": "发送邮件",
    "calendar": "日历操作",
}
```

**4. 工具编排策略**

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| 单工具调用 | 一次只调一个工具 | 简单任务 |
| 顺序调用 | 工具A的结果作为B的输入 | 多步流水线 |
| 并行调用 | 同时调用多个独立工具 | 信息聚合 |
| 条件调用 | 根据条件选择不同工具 | 决策分支 |
| 递归调用 | 根据结果决定是否继续 | 迭代优化 |

#### 实操项目
```
项目1：研究助手Agent
1. 实现web_search工具（用SerpAPI/Tavily）
2. 实现arxiv_search工具
3. 实现summarize工具（让LLM总结）
4. 编排：搜索→阅读→总结→回答
5. 支持多轮对话

项目2：代码助手Agent
1. 实现python_repl工具
2. 实现shell_command工具
3. 实现file_read/file_write工具
4. 编排：理解需求→写代码→执行→调试
```

---

### 第三阶段：Agent框架实战（4-6周）

#### 目标
- 掌握主流Agent框架
- 能用框架快速搭建Agent应用
- 理解不同框架的设计哲学

#### 学习内容

**1. LangChain**

```
核心概念：
├── Model：LLM封装
├── Prompt：模板管理
├── Chain：处理链（顺序执行）
├── Agent：自主决策（调用工具）
├── Tool：外部工具
├── Memory：记忆管理
└── Retriever：检索增强

安装：
pip install langchain langchain-openai langchain-community
```

```python
# LangChain Agent示例
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """搜索网页获取信息"""
    # 实现搜索逻辑
    return search_results

llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个研究助手，善于搜索和总结信息。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(llm, [search], prompt)
executor = AgentExecutor(agent=agent, tools=[search], verbose=True)

result = executor.invoke({"input": "2024年诺贝尔物理学奖得主是谁？"})
```

**2. LangGraph**

```
核心思想：用状态图（StateGraph）构建复杂Agent流程

优势：
├── 状态管理清晰
├── 支持循环和条件分支
├── 支持人工介入（Human-in-the-loop）
├── 可观测性好
└── 适合生产环境

核心组件：
├── State：共享状态
├── Node：处理节点
├── Edge：连接边（条件/普通）
├── Checkpointer：状态持久化
└── Human-in-the-loop：人工审核节点
```

```python
# LangGraph示例：研究Agent
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

class AgentState(TypedDict):
    messages: list
    research_data: str
    report: str
    needs_more_research: bool

def research_node(state: AgentState):
    """搜索研究"""
    # 调用搜索工具
    return {"research_data": results}

def analyze_node(state: AgentState):
    """分析数据"""
    # LLM分析
    return {"report": analysis}

def should_continue(state: AgentState):
    """判断是否需要更多研究"""
    if state["needs_more_research"]:
        return "research"
    return END

# 构建图
graph = StateGraph(AgentState)
graph.add_node("research", research_node)
graph.add_node("analyze", analyze_node)
graph.add_edge("research", "analyze")
graph.add_conditional_edges("analyze", should_continue)

app = graph.compile()
```

**3. CrewAI**

```
核心思想：多Agent协作，每个Agent有角色、目标、工具

优势：
├── 多Agent协作简单
├── 角色定义清晰
├── 任务自动分配
└── 适合复杂工作流

角色示例：
├── Researcher：负责信息收集
├── Analyst：负责数据分析
├── Writer：负责报告撰写
├── Reviewer：负责质量审核
└── Coordinator：负责任务协调
```

```python
# CrewAI示例
from crewai import Agent, Task, Crew

researcher = Agent(
    role="高级研究分析师",
    goal="发现AI领域最新突破",
    backstory="你是一位资深的AI研究分析师",
    tools=[search_tool, arxiv_tool],
    llm="gpt-4"
)

writer = Agent(
    role="技术报告撰写员",
    goal="将研究结果写成清晰的技术报告",
    backstory="你是一位经验丰富的技术写作专家",
    llm="gpt-4"
)

research_task = Task(
    description="搜索2024年最重要的AI论文",
    agent=researcher,
    expected_output="包含5篇重要论文的详细总结"
)

write_task = Task(
    description="基于研究结果撰写技术报告",
    agent=writer,
    expected_output="一篇2000字的技术报告"
)

crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task])
result = crew.kickoff()
```

**4. 框架对比**

| 框架 | 优势 | 劣势 | 适用场景 |
|------|------|------|---------|
| **LangChain** | 生态最大，工具最多 | 抽象层多，debug困难 | 快速原型 |
| **LangGraph** | 状态图，生产级 | 学习曲线陡 | 复杂Agent |
| **CrewAI** | 多Agent协作简单 | 定制性有限 | 团队协作 |
| **AutoGen** | 微软出品，对话式 | 文档较少 | 研究实验 |
| **Dify** | 低代码，可视化 | 灵活性有限 | 快速部署 |
| **Coze** | 字节出品，集成好 | 平台依赖 | 国内场景 |

#### 实操项目
```
项目1：用LangGraph搭建完整Research Agent
1. 定义状态（research_data, report, messages）
2. 实现研究节点（搜索+阅读）
3. 实现分析节点（LLM总结）
4. 实现报告节点（生成报告）
5. 加入条件分支（是否需要更多研究）
6. 部署为FastAPI服务

项目2：用CrewAI搭建内容创作团队
1. 定义角色：研究员、写手、审核员
2. 设计任务流程：搜索→撰写→审核→修改
3. 实现完整的多Agent协作
4. 优化提示词提升输出质量
```

---

### 第四阶段：Memory与RAG系统（4-6周）

#### 目标
- 掌握Agent记忆系统设计
- 能实现RAG（检索增强生成）
- 理解向量数据库使用

#### 学习内容

**1. 记忆系统**

```
记忆层次：
├── 短期记忆：当前对话上下文（对话历史）
├── 工作记忆：当前任务的中间状态
├── 长期记忆：持久化存储（向量数据库/知识图谱）
└── 情景记忆：过去交互的记录

实现方案：
├── 对话历史：直接存储messages列表
├── 摘要记忆：定期压缩对话历史
├── 向量记忆：embedding + 相似度检索
└── 实体记忆：结构化存储（名字、日期、偏好）
```

**2. RAG（检索增强生成）**

```
完整RAG流程：
文档 → 分块(Chunking) → 向量化(Embedding) → 存储(Vector DB)
                                                    ↓
用户查询 → 向量化 → 相似度检索 → 获取相关文档 → LLM生成回答

关键组件：
├── 文档加载器：PDF/Word/网页/数据库
├── 文本分块器：固定大小/语义分块
├── Embedding模型：text-embedding-3-small
├── 向量数据库：Chroma/Pinecone/Milvus/Weaviate
├── 检索器：相似度搜索/混合搜索
└── 重排器(Reranker)：提高检索精度
```

**3. 向量数据库对比**

| 数据库 | 特点 | 适用场景 |
|--------|------|---------|
| **Chroma** | 轻量，本地运行 | 原型开发 |
| **Pinecone** | 全托管，易用 | 生产环境 |
| **Milvus** | 开源，可扩展 | 大规模部署 |
| **Weaviate** | 混合搜索 | 复杂查询 |
| **Qdrant** | Rust实现，快 | 高性能需求 |
| **FAISS** | Meta出品，纯算法 | 研究实验 |

**4. 高级RAG技术**

| 技术 | 说明 | 效果 |
|------|------|------|
| HyDE | 假设性文档嵌入 | 提高检索相关性 |
| Query Rewriting | 查询改写 | 适应检索系统 |
| Multi-hop RAG | 多跳检索 | 复杂问题 |
| Self-RAG | 自适应检索 | 减少不必要的检索 |
| Graph RAG | 知识图谱+RAG | 关系推理 |

#### 实操项目
```
项目1：个人知识库Agent
1. 用LangChain加载Obsidian笔记（你有Obsidian！）
2. 文档分块 + Embedding + Chroma存储
3. 实现RAG问答
4. 加入记忆系统（对话历史+长期记忆）
5. 部署为本地服务

项目2：企业文档助手
1. 加载多种格式文档（PDF/Word/网页）
2. 实现混合搜索（向量+关键词）
3. 加入Reranker提升精度
4. 实现多轮对话+记忆
5. 加入来源引用（告诉用户答案来自哪篇文档）
```

---

### 第五阶段：生产级Agent系统（6-8周）

#### 目标
- 能设计和部署生产级Agent
- 掌握安全、监控、成本控制
- 理解Agent评估方法

#### 学习内容

**1. 安全与防护**

```
威胁模型：
├── Prompt注入：用户输入恶意指令
├── 工具滥用：LLM调用不安全的工具
├── 信息泄露：泄露系统提示词或敏感数据
└── 无限循环：Agent陷入死循环

防护措施：
├── 输入过滤：检测恶意输入
├── 工具白名单：限制可调用的工具
├── 输出审查：检查输出是否包含敏感信息
├── 超时机制：限制最大执行时间
├── Token限制：限制总token消耗
└── 人工审核：关键操作加入审核节点
```

**2. 监控与可观测性**

```python
# 监控指标
metrics = {
    # 性能指标
    "latency_p50": "响应时间中位数",
    "latency_p95": "95分位响应时间",
    "throughput": "每秒请求数",
    
    # 质量指标
    "task_success_rate": "任务完成率",
    "tool_call_accuracy": "工具调用准确率",
    "user_satisfaction": "用户满意度",
    
    # 成本指标
    "total_tokens": "总token消耗",
    "cost_per_task": "每任务成本",
    "avg_tool_calls": "平均工具调用次数",
    
    # 安全指标
    "injection_attempts": "注入攻击次数",
    "error_rate": "错误率",
    "timeout_rate": "超时率"
}
```

**3. 成本控制**

| 策略 | 说明 | 节省幅度 |
|------|------|:-------:|
| 模型降级 | 简单任务用小模型 | 50-80% |
| 缓存 | 相似查询返回缓存 | 30-60% |
| 批处理 | 合并多个请求 | 20-40% |
| Token限制 | 限制输入输出长度 | 10-30% |
| 工具优化 | 减少不必要的工具调用 | 20-50% |

**4. 评估方法**

```
评估维度：
├── 功能评估：任务是否完成
├── 质量评估：输出是否准确/有用
├── 效率评估：耗时和成本是否合理
├── 安全评估：是否有安全风险
└── 用户体验：交互是否自然

评估方法：
├── 自动评估：用LLM作为评委
├── 基准测试：标准数据集测试
├── A/B测试：对比不同版本
├── 人工评估：专家评审
└── 用户反馈：收集真实用户评价
```

#### 实操项目
```
项目1：完整的AI助手产品
1. 核心功能：问答+搜索+代码执行+文件操作
2. 安全：Prompt注入防护+输出审查
3. 记忆：短期+长期记忆系统
4. 监控：LangSmith/LangFuse接入
5. 成本：token消耗统计+模型降级策略
6. 部署：Docker + FastAPI + 前端

项目2：Agent评估系统
1. 构建测试数据集（50+测试用例）
2. 实现自动评估Pipeline
3. 对比不同模型/提示词的效果
4. 生成评估报告
```

---

## 📚 基础要求

### 前置技能
- [ ] Python熟练
- [ ] HTTP/REST API基础
- [ ] 基本的数据库知识
- [ ] 了解LLM基本概念（不需要深入原理）
- [ ] Git版本管理

### 硬件要求
| 阶段 | 最低配置 | 说明 |
|------|---------|------|
| 第一阶段 | 无 | API调用即可 |
| 第二阶段 | 无 | API调用即可 |
| 第三阶段 | 无 | API调用即可 |
| 第四阶段 | 16GB RAM | 本地向量数据库 |
| 第五阶段 | 云服务器 | 部署需要 |

### 推荐阅读顺序
1. 先学会调API（1天）
2. 再理解Prompt Engineering（1周）
3. 然后学Function Calling（1周）
4. 接着学Agent框架（2-3周）
5. 最后学RAG+生产部署（4-6周）

---

## 📖 学习资源

### 官方文档（必读）
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [CrewAI Documentation](https://docs.crewai.com/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Dify Documentation](https://docs.dify.ai/)

### 在线课程
| 课程 | 平台 | 说明 |
|------|------|------|
| Building with OpenAI | DeepLearning.AI | OpenAI官方教程 |
| LangChain Short Course | DeepLearning.AI | LangChain入门 |
| Building Agents | DeepLearning.AI | Agent系统设计 |
| RAG from Scratch | LangChain YouTube | 从零实现RAG |

### GitHub仓库（必收藏）
- [langchain](https://github.com/langchain-ai/langchain) — LangChain核心库
- [langgraph](https://github.com/langchain-ai/langgraph) — 状态图Agent
- [crewAI](https://github.com/crewAIInc/crewAI) — 多Agent协作
- [autogen](https://github.com/microsoft/autogen) — 微软多Agent
- [Dify](https://github.com/langgenius/dify) — 低代码Agent平台
- [openai-cookbook](https://github.com/openai/openai-cookbook) — OpenAI示例
- [awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps) — LLM应用合集
- [AI-Array](https://github.com/lencx/ChatGPT) — ChatGPT桌面版

### 社区/博客
- [LangChain Blog](https://blog.langchain.dev/)
- [Lilian Weng's Blog](https://lilianweng.github.io) — Agent/LLM深度博客
- [Simon Willison's Blog](https://simonwillison.net) — LLM应用实践
- [Latent Space Podcast](https://www.latent.space/podcast) — AI工程播客
- 知乎「LLM Agent」话题
- 即刻「AI探索」圈子

---

## 📄 关键论文

### Agent基础
| 论文 | 年份 | 核心贡献 |
|------|:----:|---------|
| ReAct | 2022 | Reasoning+Acting范式 |
| Toolformer | 2023 | LLM自学习使用工具 |
| Reflexion | 2023 | 语言Agent自我反思 |
| Voyager | 2023 | 开放世界Agent |
| MRKL | 2022 | 模块化推理+知识+语言 |
| Chameleon | 2023 | 组合式LLM程序 |

### 规划与推理
| 论文 | 年份 | 核心贡献 |
|------|:----:|---------|
| Chain-of-Thought | 2022 | 思维链推理 |
| Tree-of-Thought | 2023 | 树状搜索推理 |
| Graph-of-Thought | 2023 | 图状推理 |
| Self-Consistency | 2023 | 多路径一致性 |
| LATS | 2023 | 语言Agent树搜索 |
| Huginn | 2023 | 规划型Agent |

### 多Agent系统
| 论文 | 年份 | 核心贡献 |
|------|:----:|---------|
| AutoGen | 2023 | 微软多Agent框架 |
| MetaGPT | 2023 | 软件开发多Agent |
| CAMEL | 2023 | 角色扮演Agent协作 |
| ChatDev | 2023 | 虚拟软件公司 |
| AgentVerse | 2023 | 多Agent协作平台 |
| Generative Agents | 2023 | 25个Agent模拟小镇 |

### RAG相关
| 论文 | 年份 | 核心贡献 |
|------|:----:|---------|
| RAG (Original) | 2020 | 检索增强生成 |
| Self-RAG | 2023 | 自适应检索 |
| HyDE | 2023 | 假设性文档嵌入 |
| Graph RAG | 2024 | 知识图谱+RAG |
| CRAG | 2024 | 纠正式RAG |
| Multi-hop RAG | 2024 | 多跳检索 |

### 评估与安全
| 论文 | 年份 | 核心贡献 |
|------|:----:|---------|
| AgentBench | 2023 | Agent评估基准 |
| GAIA | 2023 | 通用AI助手基准 |
| Prompt Injection | 2023 | LLM安全综述 |
| Garak | 2024 | LLM漏洞扫描 |

---

## 🔧 实操项目清单（按阶段）

### P0：入门必做（2-3周）
1. ✅ 用OpenAI API写一个聊天机器人
2. ✅ 实现Chain-of-Thought推理
3. ✅ 实现Function Calling（至少3个工具）
4. ✅ 实现一个简单的ReAct Agent

### P1：框架实战（3-6周）
5. ✅ 用LangChain搭建Research Agent
6. ✅ 用LangGraph搭建带状态的Agent
7. ✅ 用CrewAI搭建多Agent协作系统
8. ✅ 部署一个Agent API服务

### P2：RAG系统（4-6周）
9. ✅ 实现文档加载+分块+向量化
10. ✅ 用Chroma搭建RAG系统
11. ✅ 实现混合搜索（向量+关键词）
12. ✅ 加入Reranker提升精度

### P3：生产级系统（6-8周）
13. ✅ 完整的AI助手产品（含安全、监控、成本控制）
14. ✅ 多Agent协作的复杂工作流
15. ✅ 评估系统+持续优化
16. ✅ 部署到生产环境+用户反馈收集

### P4：高级能力（8-12周）
17. ✅ 自定义Agent框架（理解底层原理）
18. ✅ Agent+RAG+多Agent的复杂系统
19. ✅ Agent安全攻防（Prompt注入防护）
20. ✅ 开源项目贡献/GitHub积累

---

## ⚠️ 常见坑与注意事项

1. **不要过度工程化** — 简单任务不需要Agent，直接API调用更高效
2. **工具描述是关键** — LLM靠`description`决定是否调用，写清楚！
3. **错误处理必须有** — 工具调用会失败，LLM输出会不稳定
4. **成本要控制** — Agent可能一次调用消耗数万token，要设置限制
5. **Prompt注入要防** — 用户可能通过输入劫持Agent行为
6. **不要信任LLM输出** — 关键操作要加验证/人工审核
7. **评估比开发更重要** — 没有评估就没有优化方向
8. **选对框架** — 快速原型用LangChain，生产环境用LangGraph
9. **记忆系统要设计** — 长对话会超出上下文窗口，需要压缩/检索
10. **从小开始** — 先做一个能用的，再迭代优化

---

## 📅 推荐时间线

| 周数 | 内容 | 产出 |
|:----:|------|------|
| 1 | LLM API使用 | 聊天机器人 |
| 2 | Prompt Engineering | CoT/Few-shot实验 |
| 3 | Function Calling | 3+工具调用 |
| 4-5 | LangChain Agent | Research Agent |
| 6-7 | LangGraph | 状态图Agent |
| 8-9 | CrewAI | 多Agent系统 |
| 10-11 | RAG基础 | 知识库问答 |
| 12-13 | RAG进阶 | 混合搜索+Reranker |
| 14-15 | Memory系统 | 长期记忆Agent |
| 16-17 | 安全与防护 | Prompt注入防护 |
| 18-19 | 监控与评估 | 评估系统 |
| 20-21 | 生产部署 | 完整产品 |
| 22-24 | 综合项目 | 开源/GitHub |

---

---

---

> 💡 **核心理念**：Agent开发的关键不是"用什么框架"，而是"解决什么问题"。找到一个真实的痛点，用Agent技术解决它，这才是最有价值的。

# Architecture — Enterprise Research Agent

## Overview

The Enterprise Research Agent is a multi-agent AI pipeline built with **LangGraph**, **OpenAI**, **Tavily**, and **ChromaDB**, exposed via a **Streamlit** web UI. Given a topic, it autonomously collects web evidence, synthesizes structured research, generates a consulting-style strategy report, refines the output, and produces a visual architecture diagram — all exportable as PDF or PowerPoint.

---

## High-Level Data Flow

```
User Input (topic)
       │
       ▼
 [1] Collector Agent  ──── Tavily Web Search (real-time evidence + citations)
       │
       ▼
 [2] Synthesizer Agent ─── RAG Memory (ChromaDB past research) + LLM
       │
       ▼
 [3] Strategist Agent ──── LLM (consulting report + roadmap)
       │
       ▼
 [4] Evaluator Agent ───── LLM (polish + executive refinement)
       │
       ▼
 [5] Diagram Generator ─── OpenAI Image Model (architecture diagram PNG)
       │
       ▼
   Streamlit UI
   ├── Display refined report
   ├── Display architecture diagram
   ├── Download as PDF  (ReportLab)
   ├── Download as PPT  (python-pptx)
   └── Store report to ChromaDB memory
```

---

## Component Breakdown

### 1. Entry Point — Streamlit UI
- Single-page web app launched via `streamlit run enterprise_research_agent.py`.
- Accepts a freeform **topic** string (max 500 characters) as user input.
- Drives the pipeline step-by-step with live progress spinners.
- Renders the final report, architecture diagram, and download buttons.
- All expensive resources (ChromaDB collection, compiled LangGraph workflow) are cached with `@st.cache_resource` to avoid re-initialisation on every rerender.

---

### 2. LangGraph Workflow (Orchestration)
- The five agents are registered as **nodes** in a `StateGraph`.
- Edges define a strict **sequential pipeline**: `collector → synth → strategy → eval → diagram → END`.
- A shared `State` TypedDict object is passed through every node; each agent reads its required fields and writes its output fields back.
- The compiled workflow is cached at module load time.

**State fields:**

| Field          | Written by        | Read by                         |
|----------------|-------------------|---------------------------------|
| `topic`        | UI (initial)      | Collector, Synthesizer, Diagram |
| `evidence`     | Collector         | Synthesizer                     |
| `synthesis`    | Synthesizer       | Strategist                      |
| `strategy`     | Strategist        | Evaluator                       |
| `refined`      | Evaluator         | UI, Exporter, Memory Store      |
| `diagram_path` | Diagram Generator | UI, PPT Exporter                |

---

### 3. Agent 1 — Collector
- Calls the **Tavily Search API** with the user's topic (up to 5 results).
- Returns a list of evidence strings, each formatted as `"<content> (Source: <url>)"`.
- Gracefully degrades: on API failure, injects an error placeholder so downstream agents can still run.

---

### 4. RAG Memory Layer — ChromaDB
- Uses a **persistent ChromaDB** collection (`./chroma_db`) with cosine similarity indexing (`hnsw:space=cosine`).
- **Embeddings** are generated via OpenAI `text-embedding-3-small`.
- `store_memory(text)`: Embeds and stores the final refined report after each successful run, using a UUID as the document ID.
- `retrieve_memory(query)`: Embeds the query and retrieves the **top-3** most semantically similar past reports to enrich future synthesis.
- All embedding calls use **tenacity** retry logic (3 attempts, exponential backoff 1–10 s) to handle transient API errors.

---

### 5. Agent 2 — Synthesizer
- Combines the **live Tavily evidence** with **retrieved past memory** into a single prompt.
- Instructs the LLM (`gpt-5.5-mini`) to produce a structured report with the following sections:
  - Product, Market Gap, Architecture, Benefits, Partner Implications, Risks, Future Outlook.
- Cites sources inline from the collected evidence.

---

### 6. Agent 3 — Strategist
- Takes the synthesized research and prompts the LLM to transform it into a **consulting-style report**.
- Adds: strategic roadmap, competitive positioning, and actionable insights tailored for enterprise decision-makers.

---

### 7. Agent 4 — Evaluator
- Performs a final **refinement pass** over the strategy report.
- Prompts the LLM to improve clarity, depth, and professional tone for executive consumption.
- Falls back to the unrefined strategy output if this step fails.

---

### 8. Agent 5 — Diagram Generator
- Calls the **OpenAI image model** (`gpt-image-1`, 1024×1024) with a prompt derived from the topic.
- The image is returned as base64 JSON, decoded, and written to a **temp file** (OS temp dir + UUID filename) to avoid session collisions.
- The diagram path is passed to the UI for display and to the PPT exporter for embedding.

---

### 9. Export Layer

#### PDF (ReportLab)
- Uses `SimpleDocTemplate` with A4 page size.
- Renders the refined report line-by-line with a title, body text, and spacers.
- Written to OS temp dir and served as a Streamlit download button.

#### PowerPoint (python-pptx)
- Creates a title slide, up to **15 content slides** (one per paragraph block), and a final diagram slide.
- The architecture diagram PNG is embedded at a fixed position on the diagram slide.
- Written to OS temp dir and served as a Streamlit download button.

---

### 10. LLM Calls
- All LLM completions use `openai_client.chat.completions.create` with model `gpt-5.5-mini`.
- All LLM and embedding calls are wrapped in **tenacity retry** decorators (3 attempts, exponential backoff).
- API keys are read from environment variables `OPENAI_API_KEY` and `TAVILY_API_KEY`.

---

## Key Design Decisions

1. **Sequential pipeline over parallel**: Each agent's output is the next agent's input, so strict sequencing is enforced via LangGraph edges.
2. **RAG enrichment**: Past research is retrieved and injected at the synthesis step, so the agent improves with every run.
3. **Graceful degradation**: Each agent catches its own exceptions and injects error markers rather than crashing the pipeline; the UI detects these markers and stops gracefully.
4. **Temp file isolation**: Diagrams and exports are written to the OS temp directory with UUID names to prevent concurrent session collisions.
5. **Resource caching**: ChromaDB and the compiled workflow are cached once per Streamlit server process to avoid redundant initialisation.
6. **Structured logging**: All agents log at INFO/ERROR level with `logging`, enabling production observability without print statements.

---

## Dependencies

| Library        | Role                                      |
|----------------|-------------------------------------------|
| `streamlit`    | Web UI                                    |
| `langgraph`    | Multi-agent orchestration (state graph)   |
| `openai`       | LLM completions + embeddings + image gen  |
| `tavily-python`| Real-time web search with citations       |
| `chromadb`     | Vector store for RAG memory               |
| `tenacity`     | Retry logic for API calls                 |
| `reportlab`    | PDF export                                |
| `python-pptx`  | PowerPoint export                         |

---

## File Structure

```
enterprise_research_agent.py   # Entire application (single-file)
requirements.txt               # Python dependencies
README.md                      # Quick pipeline summary
ARCHITECTURE.md                # This document
chroma_db/                     # Auto-created persistent vector store (gitignore recommended)
```

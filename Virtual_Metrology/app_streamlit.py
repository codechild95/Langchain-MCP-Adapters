# app_streamlit.py
import os
import json
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from typing import List
import sys
from pathlib import Path
from utils.helpers import load_docs_meta

sys.path.append(str(Path(__file__).resolve().parent))


# load env
load_dotenv()

# Local imports from your repo
# - data_ingest.ingest_folder(folder) : 재인덱스(벡터스토어) 생성
# - agents.agent_orchestrator.get_retrieval_chain(), get_agent() : RAG 체인 + agent 접근자
# - utils.helpers.load_docs_meta() : docs meta 로드 (optional)
from data_ingest.ingest import ingest_folder
from agents.agent_orchestrator import get_retrieval_chain, get_agent
from agents.tools import compute_stats

# UI layout
st.set_page_config(page_title="VM AI Agent", layout="wide")
st.title("VM AI Agent — Virtual Metrology Assistant")

st.markdown(
    """
편리한 Web UI로 RAG 기반 VM AI Agent를 사용하세요.\n
- **Upload** : 텍스트 로그 업로드 → `data/sample_logs/`에 저장\n
- **Reindex** : 업로드 후 인덱스를 재생성(벡터스토어 빌드)\n
- **Ask** : 질의 입력 → RAG 체인과 Agent로 답변 및 근거 문서 제공
"""
)

# Paths
DATA_DIR = Path("data/sample_logs")
DATA_DIR.mkdir(parents=True, exist_ok=True)
DOCS_META_PATH = Path("vectorstore/docs_meta.json")

# Sidebar controls
st.sidebar.header("Data / Index")
uploaded = st.sidebar.file_uploader("Upload .txt log (will save to data/sample_logs)", type=["txt", "log", "md"])
if uploaded:
    save_path = DATA_DIR / uploaded.name
    with open(save_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.sidebar.success(f"Saved to: {save_path}")
    st.sidebar.info("Run 'Rebuild index' to include this file in the vectorstore.")

if st.sidebar.button("Rebuild index (run ingest)"):
    with st.spinner("Running ingest to build vectorstore (this may take a little while)..."):
        try:
            ingest_folder(str(DATA_DIR))  # uses data_ingest.ingest.ingest_folder
            st.sidebar.success("Index rebuilt successfully. You can now ask queries.")
        except Exception as e:
            st.sidebar.error(f"Ingest failed: {e}")

if st.sidebar.button("Show indexed docs meta"):
    meta = load_docs_meta(str(DOCS_META_PATH))
    if not meta:
        st.sidebar.warning("No docs meta found. Run ingest first.")
    else:
        st.sidebar.write(meta[:50])  # show head

st.sidebar.markdown("---")
st.sidebar.header("Tool: Quick Stats")
stats_input = st.sidebar.text_area("Enter numbers as JSON list (e.g. [1,2,3])", value="")
if st.sidebar.button("Compute stats"):
    if not stats_input.strip():
        st.sidebar.error("Enter a JSON array first.")
    else:
        try:
            res = compute_stats(stats_input.strip())
            st.sidebar.success("Computed")
            st.sidebar.json(json.loads(res))
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

# Main: Query UI
st.subheader("Ask the VM Agent")
query = st.text_input("Type your question here (e.g., 'Why did Lot L1234 show CD drift?')", value="What is Virtual Metrology?")
col1, col2 = st.columns([3,1])

with col1:
    ask_button = st.button("Ask Agent")

with col2:
    st.write("Controls")
    show_sources_checkbox = st.checkbox("Show source document text", value=True)
    max_sources = st.number_input("Max sources to display", min_value=1, max_value=10, value=5)

# Cached loaders for retrieval chain & agent
# Cached loaders for retrieval chain & agent
@st.cache_resource(show_spinner=False)
def load_rag_and_agent():
    """Load retrieval chain and agent from agent_orchestrator."""
    qa_chain = None
    agent = None

    # Retrieval chain 초기화
    try:
        qa_chain = get_retrieval_chain()  # ✅ 하나의 객체만 반환됨
        st.sidebar.success("✅ Retrieval chain initialized successfully.")
    except Exception as e:
        st.sidebar.error(f"Failed to init retrieval chain: {e}")
        qa_chain = None

    # Agent 초기화
    try:
        agent = get_agent()
        st.sidebar.success("✅ Agent initialized successfully.")
    except Exception as e:
        st.sidebar.warning(f"Failed to init agent: {e}")
        agent = None

    return qa_chain, agent


retrieval_chain, agent = load_rag_and_agent()


# Action: Ask
if ask_button:
    if not query or retrieval_chain is None:
        st.warning("No query entered or retrieval chain not initialized. Rebuild index / check logs.")
    else:
        with st.spinner("Running RAG retrieval and agent..."):
            try:
                # Run retrieval / RAG chain
                # The API of retrieval_chain may vary; handle common cases robustly
                rag_result = None
                try:
                    # Standard LangChain .run or call with dict
                    rag_result = retrieval_chain({"query": query})
                except TypeError:
                    try:
                        rag_result = retrieval_chain.run(query)
                    except Exception:
                        rag_result = retrieval_chain(query)
                # rag_result might be dict-like (with result/source_documents) or plain string
                # Normalize:
                answer_text = None
                source_docs = []
                if isinstance(rag_result, dict):
                    answer_text = rag_result.get("result") or rag_result.get("answer") or str(rag_result)
                    source_docs = rag_result.get("source_documents") or rag_result.get("source_documents", [])
                else:
                    # could be string result; try to call retriever manually to show sources
                    answer_text = str(rag_result)
                    try:
                        # attempt to access retriever to fetch docs
                        retriever = retrieval_chain.retriever
                        source_docs = retriever.get_relevant_documents(query)[:max_sources]
                    except Exception:
                        source_docs = []
                # Display answer
                st.markdown("### 🤖 Agent Answer")
                st.write(answer_text)

                # Display agent tool output (optional)
                if agent is not None:
                    st.markdown("### 🧰 Agent (tool-run preview)")
                    try:
                        # Attempt a simple 'explain' call via agent.run; may be slow / verbose
                        agent_preview = agent.run(query)
                        st.write(agent_preview)
                    except Exception as e:
                        st.info(f"Agent tool-run unavailable or failed: {e}")

                # Display sources
                if source_docs:
                    st.markdown("### 📚 Source Documents")
                    for i, doc in enumerate(source_docs[:max_sources], start=1):
                        # doc can be LangChain Document or dict
                        if hasattr(doc, "page_content"):
                            text = doc.page_content
                            meta = getattr(doc, "metadata", {})
                        elif isinstance(doc, dict):
                            text = doc.get("page_content") or doc.get("text") or ""
                            meta = doc.get("metadata", {}) or {}
                        else:
                            text = str(doc)
                            meta = {}
                        source_label = meta.get("source") or meta.get("filename") or meta.get("doc_id") or f"source_{i}"
                        st.markdown(f"**[{i}] {source_label}**")
                        if show_sources_checkbox:
                            # show truncated text with expand option
                            st.expander("Show document text", expanded=False).write(text[:500] + ("..." if len(text) > 500 else ""))
                else:
                    st.info("No source documents returned by the retrieval chain.")

            except Exception as e:
                st.error(f"Error while running retrieval/agent: {e}")

# Footer: quick helpers / diagnostics
st.markdown("---")
st.markdown("### Diagnostics / Files")
colA, colB = st.columns(2)
with colA:
    st.write("Files in data/sample_logs")
    files = list(DATA_DIR.glob("*"))
    if files:
        for f in files:
            st.write(f.name, "-", f.stat().st_size, "bytes")
            if st.button(f"Open {f.name}"):
                st.code(f.read_text(encoding="utf-8")[:2000])
    else:
        st.write("No files found. Upload logs using the sidebar uploader.")
with colB:
    st.write("Docs meta (vectorstore)")
    if DOCS_META_PATH.exists():
        try:
            meta = json.loads(DOCS_META_PATH.read_text(encoding="utf-8"))
            st.write(f"Loaded docs_meta ({len(meta)} items).")
            if st.button("Show docs_meta (full)"):
                st.json(meta)
        except Exception as e:
            st.error(f"Failed to load docs_meta: {e}")
    else:
        st.info("No docs_meta found. Run ingest to build vectorstore.")

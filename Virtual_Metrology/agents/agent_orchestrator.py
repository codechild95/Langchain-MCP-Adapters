# ✅ 최신 버전 권장 Import 경로
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.memory import ConversationBufferMemory
from langchain_openai import OpenAI  # <- 새로 설치 필요

from langchain.prompts import PromptTemplate
from agents.tools import compute_stats, simple_rule_check
import os


def get_retrieval_chain(vectorstore_path: str = "vectorstore/faiss_index") -> RetrievalQA:
    """
    FAISS 벡터스토어 기반 RAG QA 체인 생성
    """
    if not os.path.exists(vectorstore_path):
        raise FileNotFoundError(f"Vectorstore not found at {vectorstore_path}")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

    llm = OpenAI(temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    return qa_chain


def get_agent():
    """
    간단한 Multi-Agent (Rule-based + LLM) 구성 예시
    """
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    template = """
    You are a Virtual Metrology AI Assistant.
    You analyze semiconductor process logs and generate insights.

    User Question: {question}
    """

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm = OpenAI(temperature=0.3)

    def agent_response(question: str) -> str:
        """
        간단한 LLM 기반 응답 생성
        """
        chain_input = {"question": question}
        try:
            answer = llm(prompt.format(**chain_input))
            return answer
        except Exception as e:
            return f"Error while generating response: {e}"

    return agent_response


def analyze_vm_data(data_list):
    """
    예시 데이터 분석 함수 — compute_stats, simple_rule_check를 호출
    """
    stats = compute_stats(data_list)
    rules = [simple_rule_check(val, 0.005) for val in data_list]
    return {"stats": stats, "rules": rules}

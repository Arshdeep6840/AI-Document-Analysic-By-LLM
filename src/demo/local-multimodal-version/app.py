"""
Modern Chat with Multiple PDF Documents
Uses open models (no gated access), LangChain 0.3+, FAISS, PyMuPDF.
"""

import streamlit as st
import pymupdf
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ----------------------------- Configuration -----------------------------
class Config:
    # Fully open model – no token needed
    USE_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.3
    TOP_P = 0.95
    K_RETRIEVAL = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------- LLM & Embeddings ---------------------------
@st.cache_resource
def load_llm():
    """Load LLM (no authentication required)."""
    tokenizer = AutoTokenizer.from_pretrained(Config.USE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        Config.USE_MODEL,
        device_map="auto" if Config.DEVICE == "cuda" else None,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=Config.MAX_NEW_TOKENS,
        temperature=Config.TEMPERATURE,
        top_p=Config.TOP_P,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return HuggingFacePipeline(pipeline=pipe)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={"device": Config.DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )

# ------------------------- PDF Processing --------------------------------
def extract_text_from_pdfs(pdf_files):
    text = ""
    progress_bar = st.progress(0, text="Extracting PDF text...")
    for i, pdf in enumerate(pdf_files):
        doc = pymupdf.open(stream=pdf.getvalue(), filetype="pdf")
        for page in doc:
            text += page.get_text()
        progress_bar.progress((i + 1) / len(pdf_files))
    progress_bar.empty()
    return text

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    st.info(f"✅ Created {len(chunks)} text chunks")
    return chunks

def build_vector_store(chunks, embeddings):
    with st.spinner("Building vector store..."):
        return FAISS.from_texts(chunks, embeddings)

# -------------------------- RAG Chain ------------------------------------
def create_rag_chain(vector_store, llm):
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": Config.K_RETRIEVAL}
    )
    prompt = ChatPromptTemplate.from_template("""
        You are a helpful AI assistant. Use the following pieces of context to answer the user's question.
        If you don't know the answer, just say that you don't know. Keep the answer concise and factual.

        Context:
        {context}

        Question: {question}

        Answer:
    """)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# -------------------------- Streamlit UI ----------------------------------
def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon="📚")
    st.header("📄 Chat with Multiple PDFs using Local Open LLMs")
    st.markdown("Upload your PDF documents, then ask questions about their content.")

    with st.spinner("Loading LLM (first time may take a while)..."):
        llm = load_llm()
    embeddings = load_embeddings()

    with st.sidebar:
        st.title("📂 Document Management")
        pdf_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        process_btn = st.button("🚀 Process PDFs", type="primary")
        st.divider()
        st.markdown(f"**LLM**: `{Config.USE_MODEL.split('/')[-1]}`")
        st.markdown(f"**Embeddings**: `{Config.EMBEDDING_MODEL.split('/')[-1]}`")
        st.markdown(f"**Retrieval k**: {Config.K_RETRIEVAL}")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if process_btn and pdf_files:
        with st.spinner("Processing PDFs..."):
            raw_text = extract_text_from_pdfs(pdf_files)
            if not raw_text.strip():
                st.error("No text could be extracted from the uploaded PDFs.")
                return
            chunks = split_text(raw_text)
            st.session_state.vector_store = build_vector_store(chunks, embeddings)
            st.success("✅ PDFs processed! You can now ask questions.")
    elif process_btn and not pdf_files:
        st.warning("Please upload at least one PDF file.")

    user_question = st.text_input("💬 Ask a question about your documents:")

    if user_question:
        if st.session_state.vector_store is None:
            st.info("Please upload and process PDFs first using the sidebar.")
        else:
            with st.spinner("Generating answer..."):
                rag_chain = create_rag_chain(st.session_state.vector_store, llm)
                answer = rag_chain.invoke(user_question)
                st.write("### 🤖 Answer")
                st.write(answer)

                with st.expander("🔍 See retrieved chunks"):
                    docs = st.session_state.vector_store.similarity_search(user_question, k=Config.K_RETRIEVAL)
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Chunk {i+1}**")
                        st.write(doc.page_content[:500] + "...")
                        st.divider()

if __name__ == "__main__":
    main()
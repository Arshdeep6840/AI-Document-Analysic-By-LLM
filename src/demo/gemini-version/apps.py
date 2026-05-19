import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import google.generativeai as genai

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings


# -----------------------------------
# Load Environment Variables
# -----------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in .env file")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)


# -----------------------------------
# Extract Text from PDFs
# -----------------------------------
def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text

    return text


# -----------------------------------
# Split Text into Chunks
# -----------------------------------
def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )

    return text_splitter.split_text(text)


# -----------------------------------
# Embedding Model
# -----------------------------------
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# -----------------------------------
# Create Vector Store
# -----------------------------------
def get_vector_store(text_chunks):

    embeddings = get_embeddings()

    vector_store = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    vector_store.save_local("faiss_index")


# -----------------------------------
# Gemini Response
# -----------------------------------
def ask_gemini(context_docs, user_question):

    context = "\n\n".join(
        [doc.page_content for doc in context_docs]
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer the question as detailed as possible
    using only the provided context.

    If the answer is not available in the context,
    say:
    "Answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    formatted_prompt = prompt.format_messages(
        context=context,
        question=user_question
    )

    model = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )

    response = model.invoke(formatted_prompt)

    return response.content


# -----------------------------------
# Handle User Input
# -----------------------------------
def user_input(user_question):

    embeddings = get_embeddings()

    try:
        db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception:
        st.error("Please upload and process PDFs first.")
        return

    docs = db.similarity_search(
        user_question,
        k=4
    )

    response = ask_gemini(
        docs,
        user_question
    )

    st.subheader("Reply")
    st.markdown(response[0]['text'])


# -----------------------------------
# Main Function
# -----------------------------------
def main():

    st.set_page_config(
        page_title="Chat with Multiple PDFs",
        layout="wide"
    )

    st.header("Chat with Multiple PDF using Gemini")

    user_question = st.text_input(
        "Ask a Question from PDF Files"
    )

    if user_question:
        user_input(user_question)

    with st.sidebar:

        st.title("Menu")

        pdf_docs = st.file_uploader(
            "Upload PDF Files",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Submit & Process"):

            if not pdf_docs:
                st.warning("Please upload PDF files.")
                return

            with st.spinner("Processing PDFs..."):

                raw_text = get_pdf_text(pdf_docs)

                if not raw_text.strip():
                    st.error("No text found in PDFs.")
                    return

                text_chunks = get_text_chunks(
                    raw_text
                )

                get_vector_store(
                    text_chunks
                )

                st.success(
                    "PDF Processing Completed!"
                )


if __name__ == "__main__":
    main()
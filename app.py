# app.py
import streamlit as st
import os
import shutil
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from utils import load_documents, split_documents, create_vector_store

# --- PATCH FOR ASYNCIO EVENT LOOP ---
# This is a critical fix for running LangChain loaders in Streamlit
import nest_asyncio
nest_asyncio.apply()

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# --- LLM SELECTION FUNCTION (WITH FALLBACK) ---
def get_llm():
    """
    Initializes and returns the appropriate LLM.
    Tries Google Generative AI first, falls back to a local Ollama model.
    """
    try:
        # Primary choice: Google Gemini Pro
        if not google_api_key:
            st.warning("Google API Key not found. Falling back to local model.")
            raise ValueError("API Key Missing")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)
        # Quick test to see if the API is responsive
        llm.invoke("Hello")
        st.session_state.llm_choice = "Google Gemini"
        return llm
    except Exception as e:
        st.warning(f"Failed to initialize Google Gemini: {e}. Falling back to Ollama.")
        try:
            # Fallback choice: Local Ollama model
            llm = Ollama(model="llama3") # Make sure you have llama3 pulled
            llm.invoke("Hello") # Quick test
            st.session_state.llm_choice = "Local Ollama (llama3)"
            return llm
        except Exception as ollama_e:
            st.error(f"Failed to initialize Ollama as well: {ollama_e}. Please ensure Ollama is running and the model is pulled.")
            return None


# --- RAG CHAIN SETUP ---
def get_retrieval_chain(retriever):
    """
    Creates and returns a retrieval chain for answering questions.
    """
    llm = get_llm()
    if llm is None:
        return None

    # **MODIFIED PROMPT** to allow for general knowledge
    prompt = ChatPromptTemplate.from_template(
        """
        You are an intelligent assistant. Answer the user's question by combining the following context with your own general knowledge.
        If the context is relevant, prioritize it, but feel free to add additional information from your own knowledge to provide a more complete and helpful answer.
        If the context doesn't contain the answer, rely solely on your own knowledge.

        Context: {context}
        Question: {input}
        Answer:
        """
    )
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, stuff_documents_chain)
    return retrieval_chain

# --- STREAMLIT UI ---
st.set_page_config(page_title="My NotebookLM Clone", layout="wide")
st.title("🧠 My NotebookLM Clone")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello! I'm your AI assistant. Please provide some documents to get started.")]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "llm_choice" not in st.session_state:
    st.session_state.llm_choice = "Not yet determined"

# --- SIDEBAR FOR DATA INPUT ---
with st.sidebar:
    st.header("Data Sources")
    st.write("Current LLM in use: **" + st.session_state.llm_choice + "**")


    # PDF Uploader
    pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    # URL Inputs
    url_input = st.text_input("Enter website or YouTube URLs (one per line)")

    if st.button("Process Sources"):
        with st.spinner("Processing... This may take a moment."):
            sources = []
            temp_dir = "./temp_pdfs"
            os.makedirs(temp_dir, exist_ok=True)

            # Handle uploaded PDFs
            if pdf_files:
                for pdf in pdf_files:
                    temp_file_path = os.path.join(temp_dir, pdf.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(pdf.getbuffer())
                    sources.append(temp_file_path)

            # Handle URLs
            if url_input:
                urls = url_input.strip().split("\n")
                sources.extend([url for url in urls if url])

            if not sources:
                st.warning("Please provide at least one data source.")
            else:
                try:
                    # 1. Load Documents
                    docs = load_documents(sources)
                    # 2. Split Documents
                    chunks = split_documents(docs)
                    # 3. Create Vector Store
                    vector_store_result = create_vector_store(chunks)

                    if vector_store_result is not None:
                        st.session_state.vector_store = vector_store_result
                        st.success(f"Processed source(s) successfully!")
                    else:
                        st.error("Failed to create the knowledge base. Please check the warnings above for details on failed sources.")

                    # Clean up the temporary PDF directory
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)

                    # Rerun to update the main page state
                    st.rerun()

                except Exception as e:
                    st.error(f"An unexpected error occurred during processing: {e}")


# --- MAIN CHAT INTERFACE ---
if st.session_state.vector_store is None:
    st.info("Please process some data sources from the sidebar to begin chatting.")
else:
    # Create retriever and retrieval chain
    retriever = st.session_state.vector_store.as_retriever()
    retrieval_chain = get_retrieval_chain(retriever)

    if retrieval_chain:
        # Display chat history
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)

        # Get user input
        user_query = st.chat_input("Ask a question about your documents...")
        if user_query:
            st.session_state.chat_history.append(HumanMessage(content=user_query))

            with st.chat_message("Human"):
                st.write(user_query)

            with st.chat_message("AI"):
                with st.spinner("Thinking..."):
                    response = retrieval_chain.invoke({"input": user_query})
                    st.write(response['answer'])
                    st.session_state.chat_history.append(AIMessage(content=response['answer']))
    else:
        st.error("Could not initialize the language model. Please check your API keys and ensure Ollama is running.")

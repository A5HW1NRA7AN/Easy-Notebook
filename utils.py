# utils.py
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

def _load_youtube_transcript_fallback(url: str):
    """
    A fallback function to load a transcript directly using youtube_transcript_api
    if the primary YoutubeLoader fails.
    """
    try:
        # Extract the video ID from the URL
        video_id = None
        if "v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]

        if not video_id:
            st.warning(f"⚠️ Could not extract video ID from URL: {url}. Skipping.")
            return []

        # Fetch the transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US'])
        
        # Combine transcript text chunks into a single string
        transcript_text = " ".join([item['text'] for item in transcript_list])
        
        # Create a LangChain Document object to maintain consistency
        doc = Document(
            page_content=transcript_text,
            metadata={"source": url, "title": "YouTube Transcript (Fallback)"}
        )
        return [doc]
    except Exception as e:
        st.warning(f"⚠️ Fallback failed: Could not fetch transcript for {url}. Error: {e}. Skipping.")
        return []

def load_documents(sources):
    """
    Loads documents from a list of sources (PDF paths, URLs, YouTube links).
    Adds robust error handling and a fallback for YouTube.
    """
    documents = []
    for source in sources:
        print(f"Processing source: {source}")
        try:
            if source.lower().endswith('.pdf'):
                loader = PyPDFLoader(source)
                documents.extend(loader.load())
            elif 'youtube.com' in source.lower() or 'youtu.be' in source.lower():
                try:
                    # First, try the standard LangChain loader
                    loader = YoutubeLoader.from_youtube_url(source, add_video_info=True, language=["en", "en-US"])
                    documents.extend(loader.load())
                except Exception as youtube_loader_error:
                    # If the primary loader fails, use the fallback
                    st.warning(f"⚠️ Primary YouTube loader failed for {source}. Trying fallback... Error: {youtube_loader_error}")
                    documents.extend(_load_youtube_transcript_fallback(source))
            else: # Assumes it's a web URL
                loader = WebBaseLoader(source)
                documents.extend(loader.load())
        except Exception as e:
            # General catch-all for other unexpected errors
            st.warning(f"⚠️ Failed to load source: {os.path.basename(source)}. Error: {e}. Skipping.")
            print(f"Failed to load source {source}: {e}")

    return documents

def split_documents(documents):
    """
    Splits a list of documents into smaller chunks.
    """
    if not documents:
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(documents)

def create_vector_store(chunks):
    """
    Creates a ChromaDB vector store from document chunks.
    """
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")

    if not chunks:
        st.warning("No text could be extracted from the provided sources. Vector store not created.")
        return None

    valid_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    if not valid_chunks:
        st.warning("Extracted content was empty after filtering. Vector store not created.")
        return None

    print(f"Creating vector store with {len(valid_chunks)} valid chunks.")
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    vector_store = Chroma.from_documents(
        documents=valid_chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )
    return vector_store

# 🧠 EASY-NOTEBOOK: Your Personal AI Knowledge Base

A powerful, self-hosted application that allows you to build a personal knowledge base from your documents, websites, and videos, and chat with them using the power of Large Language Models (LLMs).

## ✨ Overview

This project is a simple yet powerful clone of **Google's NotebookLM**, designed to run on your local machine. It leverages a **Retrieval-Augmented Generation (RAG)** architecture to provide contextually-aware answers from your private data sources.

You can feed it **PDFs**, **website URLs**, and **YouTube video links**. The application processes these sources, stores them in a local vector database, and allows you to have an intelligent conversation with your data through a clean, intuitive chat interface built with **Streamlit**.

The system is designed with flexibility in mind, primarily using the powerful **Google Gemini API**. As a fallback, it can seamlessly switch to a locally running **Ollama model** (like Llama 3 or Gemma 2), ensuring functionality even without an internet connection or when API limits are a concern.

---

## 🚀 Key Features

-   **Multi-Source Data Ingestion**: Load and process information from various formats:
    -   📄 **PDFs**: Upload multiple documents at once.
    -   🌐 **Websites**: Scrape and index content directly from URLs.
    -   📺 **YouTube Videos**: Extract and use video transcripts.
-   **Persistent Knowledge Base**: Uses **ChromaDB** to create a local, persistent vector store of your data.
-   **Intelligent Conversational AI**: Chat with your documents using a sophisticated RAG pipeline built with **LangChain**.
-   **Flexible LLM Support**:
    -   **Primary**: Connects to the **Google Gemini API** for fast, high-quality responses.
    -   **Fallback**: Automatically switches to a local **Ollama** model if the API is unavailable.
-   **User-Friendly Interface**: A simple and interactive UI built with **Streamlit** for easy data management and conversation.

---

## 🛠️ Technology Stack

-   **Frontend**: Streamlit
-   **AI / LLM Framework**: LangChain
-   **LLMs**:
    -   Google Gemini Pro (via `langchain-google-genai`)
    -   Ollama (e.g., Llama 3, Gemma 2) for local fallback
-   **Vector Database**: ChromaDB
-   **Document Loaders**: `PyPDFLoader`, `WebBaseLoader`, `YoutubeLoader`
-   **Environment Management**: `dotenv`

---

## 🏗️ Project Architecture (RAG)

The application follows the Retrieval-Augmented Generation (RAG) pattern:

1.  **Load & Split**: Documents from your sources are loaded and broken down into smaller, manageable **chunks**.
2.  **Embed & Store**: Each chunk is converted into a numerical vector (**embedding**) using Google's embedding model and stored in the **ChromaDB vector store**.
3.  **Retrieve**: When you ask a question, your query is also embedded. The system then searches the vector store for the most semantically similar chunks from your documents.
4.  **Generate**: The retrieved chunks (the context) and your original question are passed to the LLM (Gemini or Ollama), which generates a coherent, context-aware answer.

**

---

## 📁 Directory Structure

Here is the file and folder structure for the project:

```
/easy-notebook/
├── .env                  # Stores API keys and environment variables
├── .gitignore            # Specifies files for Git to ignore
├── app.py                # Main Streamlit UI and application logic
├── chroma_db/            # Directory where the persistent ChromaDB is stored
├── requirements.txt      # Lists all Python dependencies for the project
└── utils.py              # Helper functions for data processing (loading, splitting, etc.)
```

---

## ⚙️ Setup and Installation

Follow these steps to get the project running on your local machine.

### **1. Clone the Repository**

```bash
git clone [https://github.com/your-username/notebooklm-clone.git](https://github.com/your-username/notebooklm-clone.git)
cd notebooklm-clone
```

### **2. Install Dependencies**

It's recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required packages
pip install -r requirements.txt
```

### **3. Set Up API Keys and Local LLMs**

#### **Google AI Studio API Key**

1.  Visit the [Google AI Studio](https://aistudio.google.com/) and create an API key.
2.  Create a file named `.env` in the root of the project directory.
3.  Add your API key to the `.env` file:
    ```
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
    ```

#### **Ollama (Local Fallback)**

1.  [Download and install Ollama](https://ollama.com/) on your machine.
2.  Pull the models you wish to use. It is recommended to have at least one model ready.
    ```bash
    # Example: Pull Llama 3 and Gemma 2
    ollama pull llama3
    ollama pull gemma2
    ```
3.  Ensure the Ollama application is **running in the background** before starting the Streamlit app.

---

## ▶️ How to Run the Application

Once you have completed the setup, run the following command in your terminal from the project's root directory:

```bash
streamlit run app.py
```

Your default web browser will open a new tab at `http://localhost:8501`.

---

## 📖 Usage Guide

1.  **Provide Data Sources**: Use the sidebar to upload your PDF files or paste website/YouTube URLs. You can add multiple sources.
2.  **Process Sources**: Click the **"Process Sources"** button. The app will load, chunk, and embed your documents into the local ChromaDB vector store.
3.  **Start Chatting**: Once processing is complete, the chat interface will be ready. Type your questions into the input box at the bottom and get answers based on the documents you provided.

---

## 🤝 Contributing & Future Work

Contributions are welcome! Feel free to open an issue or submit a pull request.

### **Potential Improvements**

-   **Support for more file types** (e.g., `.docx`, `.txt`, `.csv`).
-   **Chat History Persistence**: Save and load conversation histories.
-   **Source Management**: Ability to view, manage, and delete sources from the knowledge base.
-   **Advanced RAG Techniques**: Implement more complex retrieval strategies for better accuracy.


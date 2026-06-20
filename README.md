# 🤖 Nova — AI Customer Support Bot (SaaS RAG Pipeline)

A production-ready, AI-powered customer support application built using **Streamlit**, **LangChain**, **FAISS**, and **OpenRouter LLMs**. Nova leverages **Retrieval-Augmented Generation (RAG)** to provide accurate, context-aware, and non-hallucinated responses by retrieving information directly from a curated knowledge base.

The application features a modern dark-themed interface and an interactive Learning Panel that demonstrates how user queries are processed through embeddings, vector similarity search, context retrieval, and Large Language Model reasoning in real time.

## ✨ Features

* 🔍 Retrieval-Augmented Generation (RAG) architecture
* 📚 Knowledge-base powered responses
* ⚡ Fast semantic search using FAISS vector database
* 🧠 LangChain-based retrieval and orchestration
* 🤖 LLM integration through OpenRouter
* 🌙 Modern dark-themed UI with Streamlit
* 📊 Interactive Learning Panel for RAG visualization
* 🔒 Reduced hallucinations through context grounding
* 📈 Scalable architecture for SaaS customer support systems

## 🛠️ Tech Stack

* Python
* Streamlit
* LangChain
* FAISS
* OpenRouter API
* Sentence Transformers
* Retrieval-Augmented Generation (RAG)
* Vector Embeddings
* Markdown Knowledge Base

## 🚀 How It Works

1. User submits a support query.
2. Query is converted into vector embeddings.
3. FAISS performs semantic similarity search.
4. Relevant context is retrieved from the knowledge base.
5. LangChain injects retrieved context into the prompt.
6. The LLM generates an accurate and grounded response.
7. The Learning Panel visualizes each step of the retrieval pipeline.

## 📚 Learning Outcomes

* Implemented a complete RAG pipeline from scratch.
* Gained hands-on experience with vector databases and embeddings.
* Learned prompt engineering and context injection techniques.
* Built production-ready AI applications using LangChain.
* Explored semantic search and information retrieval systems.

## 🎯 Use Cases

* SaaS Customer Support
* Internal Knowledge Assistants
* Documentation Search Systems
* AI Helpdesk Automation
* Enterprise Knowledge Management



text
├── app.py                  # Polished Streamlit Application & RAG Pipeline Logic
├── knowledge_base.txt      # Factual Data Source (Markdown-formatted SaaS Rules)
├── requirements.txt        # Lightweight, Dependency-Locked Python Packages
├── .streamlit/
│   └── config.toml         # Native Server Core Theme Overrides
└── faiss_db/               # Auto-Generated Local Flat-File Vector Storage (Git-Ignored)
🚀 Local Installation & Configuration
Clone the Repo:

Bash
git clone [https://github.com/your-username/saas_bot.git](https://github.com/your-username/saas_bot.git)
cd saas_bot
Install Streamline Packages:

Bash
pip install -r requirements.txt
Establish Environment Token:
Create a .env file in the root folder:

Code snippet
OPENAI_API_KEY="sk-or-v1-your-key-here"
Launch Application:

Bash
streamlit run app.py
🔧 Seamlessly Swapping Inference Models
To test different model characteristics, simply open app.py and modify the model identifier inside the ChatOpenAI building block:

Python
# Currently Active (Fast, Lightweight):
model="meta-llama/llama-3-8b-instruct:free"

# Alternative Available Free-Tier Options:
# model="mistralai/mistral-7b-instruct:free"
# model="google/gemma-2-9b-it:free"
# model="microsoft/phi-3-mini-128k-instruct:free"

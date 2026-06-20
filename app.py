"""
╔══════════════════════════════════════════════════════════════╗
║         NovaSoft AI Customer Support Bot                     ║
║         Built with: Streamlit + LangChain + FAISS + OpenAI  ║
╚══════════════════════════════════════════════════════════════╝

LEARNING GUIDE - Read this before diving into the code!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🧠 WHAT IS RAG? (Retrieval-Augmented Generation)
   RAG = Give the AI your own documents so it answers
   questions based on YOUR data, not just its training.

   Flow:
   User Question → Search your docs → Find relevant chunks
   → Send chunks + question to AI → Get accurate answer

📚 KEY CONCEPTS YOU'LL LEARN:
   1. Embeddings   - Turning text into numbers (vectors)
   2. Vector Store - A database that stores those numbers (FAISS)
   3. Retriever    - Searches the vector store for relevant info
   4. LLM Chain    - Sends context + question to the AI model
   5. Streamlit    - Builds the web UI with pure Python

🔧 TECH STACK:
   • Streamlit      → Web interface (no HTML/CSS needed!)
   • LangChain      → Framework that connects everything
   • FAISS          → Fast vector search library (by Meta)
   • OpenRouter     → API gateway to access AI models free/cheap
   • OpenAI SDK     → Used to talk to OpenRouter's API
"""

# ─────────────────────────────────────────
# SECTION 1: IMPORTS
# ─────────────────────────────────────────
# Standard Python libraries
import os
import time

# Streamlit - our web framework
import streamlit as st

# LangChain components - the AI pipeline pieces
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# ↑ ChatOpenAI = the LLM (brain), OpenAIEmbeddings = converts text to vectors

from langchain_community.document_loaders import PyPDFLoader, TextLoader
# ↑ These load documents from files (PDF or plain text)

from langchain_text_splitters import RecursiveCharacterTextSplitter
# ↑ Splits long documents into smaller chunks for better search

from langchain_community.vectorstores import FAISS
# ↑ Our vector database - stores and searches embeddings

from langchain_core.prompts import ChatPromptTemplate
# ↑ Templates for how we talk to the AI

from langchain_core.output_parsers import StrOutputParser
# ↑ Converts AI response object into a plain string

from langchain_core.runnables import RunnablePassthrough
# ↑ Passes data through the chain unchanged

from dotenv import load_dotenv
# ↑ Loads API keys from .env file (for local development)

# Load environment variables from .env file
load_dotenv()

# ─────────────────────────────────────────
# SECTION 2: PAGE CONFIG & CUSTOM CSS
# ─────────────────────────────────────────
# This MUST be the first Streamlit command
st.set_page_config(
    page_title="NovaSoft Support AI",
    page_icon="🤖",
    layout="wide",                    # Use full screen width
    initial_sidebar_state="expanded"  # Show sidebar by default
)

# Custom CSS to style our app beautifully
# Streamlit lets you inject raw CSS using st.markdown with unsafe_allow_html=True
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global Reset & Theme ── */
*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: #0F1117;
    font-family: 'Inter', sans-serif;
    color: #E2E8F0;
}

/* ── Hide default Streamlit elements ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #141621 !important;
    border-right: 1px solid #2D3148;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #94A3B8;
    font-size: 13px;
}

/* ── Main header ── */
.main-header {
    background: linear-gradient(135deg, #1E2130 0%, #16192A 100%);
    border: 1px solid #2D3148;
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #6C63FF, #00D4AA, #6C63FF);
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
}
@keyframes shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
.header-icon {
    font-size: 48px;
    line-height: 1;
}
.header-title {
    font-size: 28px;
    font-weight: 700;
    color: #F1F5F9;
    margin: 0;
    letter-spacing: -0.5px;
}
.header-subtitle {
    font-size: 14px;
    color: #64748B;
    margin: 4px 0 0 0;
}
.status-badge {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 8px;
    background: #0D2818;
    border: 1px solid #1A5C35;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 13px;
    color: #00D4AA;
    font-weight: 500;
}
.status-dot {
    width: 8px; height: 8px;
    background: #00D4AA;
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
}

/* ── Chat container ── */
.chat-container {
    background: #141621;
    border: 1px solid #2D3148;
    border-radius: 16px;
    padding: 24px;
    min-height: 400px;
    max-height: 520px;
    overflow-y: auto;
    margin-bottom: 16px;
    scroll-behavior: smooth;
}
.chat-container::-webkit-scrollbar { width: 4px; }
.chat-container::-webkit-scrollbar-track { background: transparent; }
.chat-container::-webkit-scrollbar-thumb { background: #2D3148; border-radius: 2px; }

/* ── Messages ── */
.message-wrapper {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
    animation: fadeSlideIn 0.3s ease;
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.message-wrapper.user { flex-direction: row-reverse; }

.avatar {
    width: 36px; height: 36px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}
.avatar.bot  { background: linear-gradient(135deg, #6C63FF, #9B8FFF); }
.avatar.user { background: linear-gradient(135deg, #00D4AA, #00A882); }

.message-bubble {
    max-width: 75%;
    padding: 14px 18px;
    border-radius: 16px;
    font-size: 14.5px;
    line-height: 1.65;
}
.message-bubble.bot {
    background: #1E2130;
    border: 1px solid #2D3148;
    border-top-left-radius: 4px;
    color: #CBD5E1;
}
.message-bubble.user {
    background: linear-gradient(135deg, #4A44B5, #6C63FF);
    border-top-right-radius: 4px;
    color: #F1F5F9;
}
.message-time {
    font-size: 11px;
    color: #475569;
    margin-top: 6px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Welcome message ── */
.welcome-box {
    text-align: center;
    padding: 48px 24px;
    color: #475569;
}
.welcome-box .big-icon { font-size: 64px; margin-bottom: 16px; }
.welcome-box h3 { color: #94A3B8; font-weight: 500; margin-bottom: 8px; }
.welcome-box p  { font-size: 14px; }

/* ── Suggestion chips ── */
.chips-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 16px;
}
.chip {
    background: #1E2130;
    border: 1px solid #2D3148;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 13px;
    color: #94A3B8;
    cursor: pointer;
    transition: all 0.2s ease;
    font-family: 'Inter', sans-serif;
}
.chip:hover {
    border-color: #6C63FF;
    color: #A5A0FF;
    background: #1A1B2E;
}

/* ── Learning panel ── */
.learn-card {
    background: #141621;
    border: 1px solid #2D3148;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
}
.learn-card-title {
    font-size: 12px;
    font-weight: 600;
    color: #6C63FF;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 8px;
}
.learn-card p {
    font-size: 13px;
    color: #64748B;
    line-height: 1.6;
    margin: 0;
}
.code-tag {
    font-family: 'JetBrains Mono', monospace;
    background: #1E2130;
    border: 1px solid #2D3148;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 12px;
    color: #00D4AA;
}

/* ── Stats row ── */
.stats-row {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
}
.stat-card {
    flex: 1;
    background: #141621;
    border: 1px solid #2D3148;
    border-radius: 12px;
    padding: 14px;
    text-align: center;
}
.stat-value {
    font-size: 22px;
    font-weight: 700;
    color: #6C63FF;
    font-family: 'JetBrains Mono', monospace;
}
.stat-label {
    font-size: 11px;
    color: #475569;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Source pills ── */
.sources-box {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #2D3148;
}
.source-pill {
    display: inline-block;
    background: #0D1424;
    border: 1px solid #1E3A5F;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    color: #60A5FA;
    font-family: 'JetBrains Mono', monospace;
    margin: 2px 4px 2px 0;
}

/* ── Streamlit input styling ── */
.stChatInput > div {
    border-radius: 12px !important;
    border-color: #2D3148 !important;
    background: #141621 !important;
}
.stChatInput textarea {
    color: #E2E8F0 !important;
    background: transparent !important;
}

/* ── Sidebar sections ── */
.sidebar-section {
    background: #1A1D2E;
    border: 1px solid #2D3148;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 14px;
}
.sidebar-label {
    font-size: 11px;
    font-weight: 600;
    color: #6C63FF;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 10px;
}

/* ── Alerts ── */
.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SECTION 3: CONSTANTS & CONFIGURATION
# ─────────────────────────────────────────
# These are fixed values used throughout the app
DB_DIR = "./faiss_db"                   # Where we save our vector index
KB_FILE = "knowledge_base.txt"          # Our text knowledge base file


# ─────────────────────────────────────────
# SECTION 4: THE RAG PIPELINE
# ─────────────────────────────────────────
# @st.cache_resource means: run this function ONCE, then reuse the result.
# Without caching, the AI model + vector DB would reload on every user action!
@st.cache_resource
def build_rag_pipeline():
    """
    LEARNING: This function builds the entire RAG pipeline.
    
    Steps:
    1. Create embedding model (converts text → vectors)
    2. Load or build FAISS vector store
    3. Create retriever (searches the vector store)
    4. Create LLM (the AI brain)
    5. Create prompt template
    6. Chain everything together with LCEL (LangChain Expression Language)
    
    Returns: (rag_chain, retriever, doc_count)
    """
    
    # ── Step 1: Get API key ──
    # st.secrets is how Streamlit Cloud stores secrets
    # os.getenv is the fallback for local .env file
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    
    if not api_key:
        raise ValueError("No OPENAI_API_KEY found. Add it to Streamlit Secrets or .env file.")
    
    # ── Step 2: Create Embedding Model ──
    # Embeddings turn text into a list of numbers (a "vector")
    # Similar meaning = similar numbers = close together in vector space
    # We use OpenRouter to access the embedding model cheaply
    embedding_model = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",  # Model that creates vectors
        openai_api_base="https://openrouter.ai/api/v1",  # Route through OpenRouter
        openai_api_key=api_key
    )
    
    # ── Step 3: Load or Build FAISS Vector Store ──
    if os.path.exists(DB_DIR):
        # If we already built it before, just load from disk (fast!)
        vector_store = FAISS.load_local(
            DB_DIR,
            embedding_model,
            allow_dangerous_deserialization=True  # Required flag for FAISS loading
        )
        # Count how many text chunks we have stored
        doc_count = vector_store.index.ntotal
    else:
        # First time setup: build the vector store from our knowledge base
        # Step 3a: Load the text file
        if not os.path.exists(KB_FILE):
            raise FileNotFoundError(f"Knowledge base file '{KB_FILE}' not found!")
        
        loader = TextLoader(KB_FILE, encoding="utf-8")
        documents = loader.load()
        # documents = list of Document objects, each has .page_content and .metadata
        
        # Step 3b: Split into chunks
        # We can't send the whole document to AI (token limits!)
        # So we split it into overlapping chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,      # Each chunk = ~600 characters
            chunk_overlap=100,   # 100 chars overlap between chunks (prevents cut-off context)
            separators=["\n\n", "\n", ".", " "]  # Try to split at natural boundaries
        )
        chunks = text_splitter.split_documents(documents)
        # chunks = list of smaller Document objects
        
        # Step 3c: Embed and index all chunks
        # This calls the embedding API for every chunk and stores results in FAISS
        vector_store = FAISS.from_documents(chunks, embedding_model)
        
        # Step 3d: Save to disk so we don't rebuild every time
        vector_store.save_local(DB_DIR)
        doc_count = len(chunks)
    
    # ── Step 4: Create Retriever ──
    # The retriever takes a query and returns the most relevant chunks
    # k=4 means "return top 4 most similar chunks"
    retriever = vector_store.as_retriever(
        search_type="similarity",  # Use cosine similarity to find matches
        search_kwargs={"k": 4}     # Return top 4 chunks
    )
    
    # ── Step 5: Create the LLM (AI Model) ──
    # This is the "brain" that generates the final answer
    llm = ChatOpenAI(
        model="meta-llama/llama-3-8b-instruct:free",      # Free model on OpenRouter
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
        temperature=0.3,    # 0 = very factual/consistent, 1 = more creative/varied
        max_tokens=800      # Max length of AI response
    )
    
    # ── Step 6: Create the Prompt Template ──
    # This is the "instruction manual" we give to the AI
    # {context} = the retrieved chunks will go here
    # {input}   = the user's question will go here
    system_prompt = """You are Nova, a friendly and knowledgeable AI customer support agent for NovaSoft — a SaaS project management platform.

Your personality:
- Warm, professional, and concise
- You use the customer's exact question to find the right answer
- You always answer from the provided context only
- If you don't know something, you say so honestly and offer to escalate

Instructions:
1. Answer ONLY using the information in the context below
2. If the answer isn't in the context, say: "I don't have that specific information right now. Please contact our support team at support@novasoft.io or visit help.novasoft.io"
3. Keep answers clear and structured — use bullet points for lists
4. End with a helpful follow-up offer when appropriate

Context from knowledge base:
{context}"""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    # ── Step 7: Build the Chain with LCEL ──
    # LCEL = LangChain Expression Language
    # The | (pipe) operator chains steps together, like Linux pipes
    # 
    # Chain flow:
    # User question → retriever finds chunks → format_docs joins them →
    # prompt fills in context+question → llm generates answer → parser extracts text
    
    def format_docs(docs):
        """Join retrieved document chunks into one big context string."""
        return "\n\n---\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        # Step A: Prepare inputs for the prompt
        {
            "context": retriever | format_docs,  # Retrieve docs, then format them
            "input": RunnablePassthrough()        # Pass user question through unchanged
        }
        | prompt_template    # Step B: Fill the prompt template
        | llm                # Step C: Send to AI model
        | StrOutputParser()  # Step D: Extract text from AI response object
    )
    
    return rag_chain, retriever, doc_count


# ─────────────────────────────────────────
# SECTION 5: HELPER FUNCTIONS
# ─────────────────────────────────────────

def get_current_time():
    """Return formatted current time string."""
    return time.strftime("%H:%M")

def render_message(role, content, timestamp=""):
    """
    Render a single chat message as styled HTML.
    role = "bot" or "user"
    """
    if role == "bot":
        avatar = "🤖"
        bubble_class = "bot"
        wrapper_class = ""
    else:
        avatar = "👤"
        bubble_class = "user"
        wrapper_class = "user"
    
    time_html = f'<div class="message-time">{timestamp}</div>' if timestamp else ""
    
    return f"""
    <div class="message-wrapper {wrapper_class}">
        <div class="avatar {role}">{avatar}</div>
        <div>
            <div class="message-bubble {bubble_class}">{content}</div>
            {time_html}
        </div>
    </div>
    """

SUGGESTIONS = [
    "💰 What are your pricing plans?",
    "🚀 How do I get started?",
    "🔌 What integrations do you support?",
    "🔒 How do I enable two-factor authentication?",
    "📊 What analytics features are available?",
    "🆘 My payment failed, what do I do?",
]


# ─────────────────────────────────────────
# SECTION 6: SIDEBAR
# ─────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 24px;">
        <div style="font-size:40px">🤖</div>
        <div style="font-size:16px; font-weight:700; color:#F1F5F9; margin-top:8px">Nova Support</div>
        <div style="font-size:12px; color:#475569; margin-top:4px">Powered by RAG + LangChain</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Knowledge Base Status ──
    st.markdown('<div class="sidebar-label">📚 Knowledge Base</div>', unsafe_allow_html=True)
    
    kb_exists = os.path.exists(KB_FILE)
    db_exists = os.path.exists(DB_DIR)
    
    if kb_exists:
        kb_size = os.path.getsize(KB_FILE) // 1024
        st.success(f"✅ `{KB_FILE}` loaded ({kb_size} KB)")
    else:
        st.error(f"❌ `{KB_FILE}` not found")
    
    if db_exists:
        st.success("✅ Vector index ready")
    else:
        st.info("⏳ Index will build on first query")

    st.markdown("---")
    
    # ── Learning Mode Toggle ──
    st.markdown('<div class="sidebar-label">🎓 Learning Mode</div>', unsafe_allow_html=True)
    show_learning = st.toggle("Show learning panel", value=True)
    show_sources = st.toggle("Show retrieved sources", value=True)
    
    st.markdown("---")
    
    # ── Quick Stats ──
    st.markdown('<div class="sidebar-label">📈 Session Stats</div>', unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    q_count = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.markdown(f"""
    <div class="sidebar-section">
        <div style="display:flex; justify-content:space-between; margin-bottom:8px">
            <span style="color:#64748B; font-size:13px">Questions asked</span>
            <span style="color:#6C63FF; font-weight:600; font-family:monospace">{q_count}</span>
        </div>
        <div style="display:flex; justify-content:space-between">
            <span style="color:#64748B; font-size:13px">Vector DB</span>
            <span style="color:#00D4AA; font-weight:600; font-size:12px">FAISS</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Clear Chat ──
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # ── Architecture diagram ──
    if show_learning:
        st.markdown("---")
        st.markdown('<div class="sidebar-label">🔧 RAG Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace; font-size:11px; color:#475569; line-height:2">
        User Question<br>
        &nbsp;&nbsp;&nbsp;↓<br>
        <span style="color:#6C63FF">Embedding Model</span><br>
        &nbsp;&nbsp;&nbsp;↓<br>
        <span style="color:#6C63FF">FAISS Search</span><br>
        &nbsp;&nbsp;&nbsp;↓<br>
        Top 4 Chunks<br>
        &nbsp;&nbsp;&nbsp;↓<br>
        <span style="color:#00D4AA">Prompt Template</span><br>
        &nbsp;&nbsp;&nbsp;↓<br>
        <span style="color:#00D4AA">LLM (OpenRouter)</span><br>
        &nbsp;&nbsp;&nbsp;↓<br>
        Final Answer ✨
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# SECTION 7: MAIN LAYOUT
# ─────────────────────────────────────────

col_chat, col_learn = st.columns([2.2, 1] if show_learning else [1, 0])

with col_chat:
    # ── Header ──
    st.markdown("""
    <div class="main-header">
        <div class="header-icon">🤖</div>
        <div>
            <p class="header-title">Nova — AI Support Agent</p>
            <p class="header-subtitle">Ask me anything about NovaSoft</p>
        </div>
        <div class="status-badge">
            <div class="status-dot"></div>
            Online
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Initialize RAG pipeline ──
    try:
        with st.spinner("🔧 Initializing AI pipeline..."):
            rag_chain, retriever, doc_count = build_rag_pipeline()
        pipeline_ready = True
    except Exception as e:
        st.error(f"⚠️ Pipeline Error: {e}")
        st.info("💡 Make sure `OPENAI_API_KEY` is set in Streamlit Cloud → App Settings → Secrets")
        pipeline_ready = False

    # ── Chat Messages Area ──
    chat_html = '<div class="chat-container" id="chat-area">'
    
    if not st.session_state.messages:
        # Show welcome screen when no messages
        chat_html += """
        <div class="welcome-box">
            <div class="big-icon">👋</div>
            <h3>Hi! I'm Nova, your AI support agent.</h3>
            <p>I know everything about NovaSoft — pricing, features, integrations,<br>
            troubleshooting, and more. Ask me anything!</p>
        </div>
        """
    else:
        for msg in st.session_state.messages:
            chat_html += render_message(
                msg["role"],
                msg["content"],
                msg.get("time", "")
            )
    
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)

    # ── Suggestion Chips ──
    if not st.session_state.messages:
        st.markdown('<div class="chips-row">', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, suggestion in enumerate(SUGGESTIONS):
            with cols[i % 3]:
                if st.button(suggestion, key=f"chip_{i}", use_container_width=True):
                    # When chip is clicked, treat it as user input
                    clean = suggestion.split(" ", 1)[1]  # Remove emoji prefix
                    st.session_state.pending_question = clean
                    st.rerun()

    # ── Chat Input ──
    user_input = st.chat_input("Ask Nova anything about NovaSoft...", disabled=not pipeline_ready)
    
    # Handle chip click (stored as pending_question)
    if "pending_question" in st.session_state:
        user_input = st.session_state.pop("pending_question")

    # ── Process User Input ──
    if user_input and pipeline_ready:
        current_time = get_current_time()
        
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "time": current_time
        })
        
        # Get AI response
        with st.spinner("Nova is thinking..."):
            try:
                # This invokes the entire RAG chain:
                # user_input → retrieve → prompt → LLM → answer
                answer = rag_chain.invoke(user_input)
                
                # If show_sources is on, also fetch the source chunks
                if show_sources:
                    source_docs = retriever.invoke(user_input)
                    sources_html = '<div class="sources-box"><span style="font-size:11px;color:#475569">📎 Retrieved from: </span>'
                    seen = set()
                    for doc in source_docs:
                        # Show a snippet of each source chunk
                        snippet = doc.page_content[:50].replace('\n', ' ').strip()
                        if snippet not in seen:
                            sources_html += f'<span class="source-pill">{snippet}...</span>'
                            seen.add(snippet)
                    sources_html += '</div>'
                    full_answer = answer + sources_html
                else:
                    full_answer = answer
                
                st.session_state.messages.append({
                    "role": "bot",
                    "content": full_answer,
                    "time": current_time
                })
                
            except Exception as e:
                st.session_state.messages.append({
                    "role": "bot",
                    "content": f"⚠️ Sorry, I ran into an error: {str(e)}",
                    "time": current_time
                })
        
        st.rerun()


# ─────────────────────────────────────────
# SECTION 8: LEARNING PANEL (right column)
# ─────────────────────────────────────────

if show_learning:
    with col_learn:
        st.markdown("""
        <div style="padding-top: 8px;">
        <div style="font-size:13px; font-weight:600; color:#6C63FF; 
                    text-transform:uppercase; letter-spacing:0.8px; margin-bottom:16px">
            🎓 Learning Panel
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="learn-card">
            <div class="learn-card-title">💡 What just happened?</div>
            <p>When you sent a message:<br><br>
            1️⃣ Your text was converted to a <b style="color:#E2E8F0">vector</b> (list of numbers)<br><br>
            2️⃣ <b style="color:#E2E8F0">FAISS</b> searched for the 4 most similar chunks in our knowledge base<br><br>
            3️⃣ Those chunks were injected into the <b style="color:#E2E8F0">prompt</b> as context<br><br>
            4️⃣ The <b style="color:#E2E8F0">LLM</b> generated an answer using only that context</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="learn-card">
            <div class="learn-card-title">🧠 Key Concept: Embeddings</div>
            <p>An embedding turns words into numbers.<br><br>
            <span class="code-tag">"pricing"</span> and <span class="code-tag">"cost"</span> 
            will have <b style="color:#E2E8F0">similar vectors</b> because they mean similar things.<br><br>
            This is why the AI can find relevant info even if you use different words!</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="learn-card">
            <div class="learn-card-title">⛓️ Key Concept: LCEL Chain</div>
            <p>The <span class="code-tag">|</span> pipe operator chains steps:<br><br>
            <span class="code-tag" style="display:block; margin:4px 0">retriever</span>
            <span style="color:#475569; padding-left:8px">↓ finds chunks</span><br>
            <span class="code-tag" style="display:block; margin:4px 0">format_docs</span>
            <span style="color:#475569; padding-left:8px">↓ joins them</span><br>
            <span class="code-tag" style="display:block; margin:4px 0">prompt</span>
            <span style="color:#475569; padding-left:8px">↓ fills template</span><br>
            <span class="code-tag" style="display:block; margin:4px 0">llm</span>
            <span style="color:#475569; padding-left:8px">↓ generates answer</span><br>
            <span class="code-tag" style="display:block; margin:4px 0">StrOutputParser</span>
            <span style="color:#475569; padding-left:8px">↓ extracts text</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="learn-card">
            <div class="learn-card-title">📁 Project Structure</div>
            <p>
            <span class="code-tag">app.py</span> — Main Streamlit app<br><br>
            <span class="code-tag">knowledge_base.txt</span> — Your data source<br><br>
            <span class="code-tag">faiss_db/</span> — Auto-generated vector index<br><br>
            <span class="code-tag">requirements.txt</span> — Python dependencies
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="learn-card">
            <div class="learn-card-title">🔑 Key Functions</div>
            <p>
            <span class="code-tag">@st.cache_resource</span><br>
            <span style="font-size:12px">Runs once, caches result</span><br><br>
            <span class="code-tag">FAISS.from_documents()</span><br>
            <span style="font-size:12px">Embeds & indexes chunks</span><br><br>
            <span class="code-tag">retriever.invoke()</span><br>
            <span style="font-size:12px">Searches vector store</span><br><br>
            <span class="code-tag">rag_chain.invoke()</span><br>
            <span style="font-size:12px">Runs the full pipeline</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

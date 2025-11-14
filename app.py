import streamlit as st
import os # Keep os for client creation, though we don't use getenv
import time
from pathlib import Path
from PyPDF2 import PdfReader
import re
import random

import chromadb
# Import specific components needed for explicit client configuration
from chromadb.config import Settings, System, DEFAULT_TENANT, DEFAULT_DATABASE

from google import genai
from google.api_core.exceptions import ResourceExhausted
from google.genai.types import EmbedContentConfig

# ---------------- Load Streamlit Secret ----------------
# The API key is now loaded from Streamlit's st.secrets
if "GEMINI_API_KEY" not in st.secrets:
    st.error("GEMINI_API_KEY not found in st.secrets. Please configure your secrets.toml file or deployment secrets.")
    st.stop()

api_key = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)

# ---------------- Embedding Wrapper ----------------
class GeminiEmbeddingFunction:
    def __init__(self, model="gemini-embedding-001", batch_size=10):
        self.model = model
        self.batch_size = batch_size

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        input = [chunk for chunk in input if chunk.strip()]
        if not input:
            return []

        embeddings = []
        for i in range(0, len(input), self.batch_size):
            batch = input[i:i + self.batch_size]
            try:
                response = client.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
            except ResourceExhausted:
                # Add a small delay and retry on ResourceExhausted error
                time.sleep(1)
                response = client.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
            embeddings.extend([list(emb.values) for emb in response.embeddings])
        return embeddings

embedding_function = GeminiEmbeddingFunction(batch_size=10)

# ---------------- Streamlit App ----------------
st.set_page_config(page_title="AI Study Agent", page_icon="🤖")
st.title("🤖 AI Study Agent")

# Personalization: Simple profile
if "user_name" not in st.session_state:
    st.session_state.user_name = "Student"
if "course_name" not in st.session_state:
    st.session_state.course_name = "Your Course"

# Define the explicit ChromaDB settings object
CHROMA_SETTINGS = Settings(
    chroma_client_impl="chromadb.db.impl.duckdb.persistent.PersistentDuckDB",
    chroma_api_impl="chromadb.api.local.LocalAPI",
    is_persistent=True
)

with st.sidebar:
    st.header("Personalize Your Agent")
    st.session_state.user_name = st.text_input("Your Name", value=st.session_state.user_name)
    st.session_state.course_name = st.text_input("Course Name", value=st.session_state.course_name)
    st.info(f"Hello, {st.session_state.user_name}! Ready to study {st.session_state.course_name}?")

    # Reset Database Button
    if st.button("Reset Database (Delete All Collections)"):
        try:
            # Use the explicit settings here too
            chroma_client = chromadb.PersistentClient(path="./chroma_db", settings=CHROMA_SETTINGS)
            for collection_name in chroma_client.list_collections():
                chroma_client.delete_collection(collection_name.name)
            st.session_state.vector_dbs = {}
            st.success("Database reset! Re-upload PDFs.")
        except Exception as e:
            st.error(f"Error resetting database: {e}")

# Initialize session state
if "vector_dbs" not in st.session_state:
    st.session_state.vector_dbs = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of (user_msg, agent_response)

# Load existing collections from persistent ChromaDB on app start
try:
    # --- FIX APPLIED HERE: Using explicit settings object ---
    chroma_client = chromadb.PersistentClient(path="./chroma_db", settings=CHROMA_SETTINGS)
    # --------------------------------------------------------
    for collection in chroma_client.list_collections():
        st.session_state.vector_dbs[collection.name] = collection
except Exception as e:
    st.error(f"Error loading ChromaDB: {e}. Use 'Reset Database' to fix.")
    chroma_client = None

# ---------------- PDF Upload ----------------
st.header("Upload Course PDFs")
uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded_files and chroma_client:
    for uploaded_file in uploaded_files:
        pdf = PdfReader(uploaded_file)
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        if not text.strip():
            st.error(f"PDF '{uploaded_file.name}' has no readable text.")
            continue

        st.success(f"PDF '{uploaded_file.name}' loaded successfully!")

        # Split text into chunks
        chunk_size = 500
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        chunks = [c.strip() for c in chunks if c.strip()]
        st.info(f"PDF split into {len(chunks)} chunks.")

        collection_name = re.sub(r"[^a-zA-Z0-9._-]", "_", uploaded_file.name).strip("_")

        if collection_name in st.session_state.vector_dbs:
            collection = st.session_state.vector_dbs[collection_name]
            st.info(f"Using existing collection for {uploaded_file.name}.")
        else:
            try:
                collection = chroma_client.create_collection(
                    name=collection_name,
                    embedding_function=embedding_function
                )
                st.session_state.vector_dbs[collection_name] = collection
            except Exception as e:
                st.error(f"Error creating collection: {e}")
                continue

        existing_count = collection.count()
        if existing_count == 0:
            embeddings = embedding_function(chunks)
            if embeddings:
                ids = [str(i) for i in range(len(chunks))]
                try:
                    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
                    st.success(f"{len(chunks)} chunks stored in vector DB for {uploaded_file.name}.")
                except Exception as e:
                    st.error(f"Error adding to collection: {e}")
            else:
                st.error("No embeddings generated. Check PDF content or API key.")
        else:
            st.info(f"Chunks already exist for {uploaded_file.name}.")

# ---------------- Agent Functions ----------------
def get_pdf_context(question, top_k=3):
    context_text = ""
    corrupted_collections = []
    for collection_name, collection in list(st.session_state.vector_dbs.items()):
        try:
            if collection.count() > 0:
                q_embeddings = embedding_function(question)
                if q_embeddings:
                    results = collection.query(query_embeddings=q_embeddings, n_results=top_k)
                    context_chunks = results.get("documents", [[]])[0]
                    context_text += "\n".join(context_chunks) + "\n"
        except Exception as e:
            st.warning(f"Error retrieving context from {collection_name}: {e}. Deleting corrupted collection.")
            corrupted_collections.append(collection_name)
    
    # Delete corrupted collections
    for collection_name in corrupted_collections:
        try:
            # Need to re-initialize client for deletion, using the explicit settings
            delete_client = chromadb.PersistentClient(path="./chroma_db", settings=CHROMA_SETTINGS)
            delete_client.delete_collection(collection_name)
            del st.session_state.vector_dbs[collection_name]
        except Exception as e:
            st.error(f"Error deleting collection {collection_name}: {e}")
    
    return context_text.strip()

def generate_summary(topic):
    context = get_pdf_context(f"summarize {topic}")
    if context:
        prompt = f"Summarize the following content on '{topic}' in a concise, study-friendly way:\n\n{context[:2000]}"
    else:
        prompt = f"Summarize '{topic}' based on general knowledge."
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return response.text
    except Exception as e:
        return f"Error generating summary: {e}"

def generate_quiz(topic):
    context = get_pdf_context(f"quiz on {topic}")
    if context:
        prompt = f"Create a 5-question multiple-choice quiz on '{topic}' based on this content. Include answers at the end.\n\n{context[:2000]}"
    else:
        prompt = f"Create a 5-question multiple-choice quiz on '{topic}' based on general knowledge. Include answers at the end."
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return response.text
    except Exception as e:
        return f"Error generating quiz: {e}"

def generate_flashcards(topic):
    context = get_pdf_context(f"flashcards for {topic}")
    if context:
        prompt = f"Create 5 flashcards (question-answer pairs) on '{topic}' from this content:\n\n{context[:2000]}"
    else:
        prompt = f"Create 5 flashcards (question-answer pairs) on '{topic}' based on general knowledge."
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return response.text
    except Exception as e:
        return f"Error generating flashcards: {e}"

def handle_agent_response(user_msg):
    user_msg = user_msg.strip().lower()
    response = ""

    # Check for special commands
    if user_msg.startswith("summarize"):
        topic = user_msg.replace("summarize", "").strip()
        response = generate_summary(topic)
    elif user_msg.startswith("quiz me on"):
        topic = user_msg.replace("quiz me on", "").strip()
        response = generate_quiz(topic)
    elif user_msg.startswith("flashcards for"):
        topic = user_msg.replace("flashcards for", "").strip()
        response = generate_flashcards(topic)
    else:
        # General question: Use PDF context + history
        context = get_pdf_context(user_msg)
        history = "\n".join([f"User: {u}\nAgent: {a}" for u, a in st.session_state.chat_history[-5:]])  # Last 5 exchanges
        if context:
            prompt = f"You are a personalized study agent for {st.session_state.user_name} studying {st.session_state.course_name}. Answer based on the context and conversation history. If context is insufficient, use general knowledge.\n\nContext:\n{context[:2000]}\n\nHistory:\n{history}\n\nQuestion: {user_msg}\nAnswer:"
        else:
            prompt = f"You are a personalized study agent for {st.session_state.user_name} studying {st.session_state.course_name}. Answer the question based on conversation history and general knowledge.\n\nHistory:\n{history}\n\nQuestion: {user_msg}\nAnswer:"
        try:
            llm_response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            response = llm_response.text
        except Exception as e:
            response = f"Error: {e}"

    # Add to history
    st.session_state.chat_history.append((user_msg, response))
    return response

# ---------------- Chat Interface ----------------
st.header("Chat with Your Study Agent")
st.info("Ask questions, or try commands like 'Summarize [topic]', 'Quiz me on [topic]', or 'Flashcards for [topic]'.")

# Display chat history
for user_msg, agent_resp in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(user_msg)
    with st.chat_message("assistant"):
        st.write(agent_resp)

# Chat input
if prompt := st.chat_input("Type your message..."):
    # User message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Agent response
    with st.chat_message("assistant"):
        response = handle_agent_response(prompt)
        st.write(response)
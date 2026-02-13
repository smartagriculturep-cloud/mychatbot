import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import tiktoken
import os

# LLM Providers
from groq import Groq
from openai import OpenAI

# -------------------------
# STREAMLIT PAGE
# -------------------------
st.set_page_config(page_title="Advanced RAG Chatbot")
st.title("Our Chatbot")

# -------------------------
# SIDEBAR CONFIG
# -------------------------
st.sidebar.header("⚙️ Model Configuration")

provider = st.sidebar.selectbox("Select API Provider", ["Groq", "OpenAI"])

api_key = st.sidebar.text_input("Enter API Key", type="password")

if provider == "Groq":
    model_name = st.sidebar.selectbox(
        "Select Groq Model",
        ["llama3-8b-8192", "llama3-70b-8192","llama-3.1-8b-instant"]
    )
else:
    model_name = st.sidebar.selectbox(
        "Select OpenAI Model",
        ["gpt-4o-mini", "gpt-4o"]
    )

temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.3)

# -------------------------
# TOKEN COUNTER
# -------------------------
def count_tokens(text, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

# -------------------------
# RULE BASED RESPONSES
# -------------------------
def rule_based_response(query):
    rules = {
        "hi": "Hello! 👋 How can I help you?",
        "hello": "Hi there! Ask me something from your uploaded file.",
        "bye": "Goodbye! Have a productive day!"
    }

    for key in rules:
        if key in query.lower():
            return rules[key]
    return None

# -------------------------
# READ FILE
# -------------------------
def read_file(uploaded_file):
    text = ""
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text()
    else:
        text = uploaded_file.read().decode("utf-8")
    return text

# -------------------------
# CHROMADB SETUP (Persistent)
# -------------------------
chroma_client = chromadb.Client(
    settings=chromadb.Settings(
        persist_directory="./chroma_db",
        is_persistent=True
    )
)

collection = chroma_client.get_or_create_collection(name="rag_collection")

# -------------------------
# FILE UPLOAD
# -------------------------
uploaded_file = st.file_uploader("Upload PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    document_text = read_file(uploaded_file)

    chunks = [document_text[i:i+500] for i in range(0, len(document_text), 500)]

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            ids=[f"{uploaded_file.name}_{i}"]
        )

    st.success("File processed and stored in ChromaDB!")

# -------------------------
# CHAT HISTORY
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------
# USER QUERY
# -------------------------
query = st.text_input("Ask your question")

if query and api_key:

    # RULE CHECK
    rule_response = rule_based_response(query)

    if rule_response:
        response = rule_response
        input_tokens = count_tokens(query)
        output_tokens = count_tokens(response)

    else:
        # Retrieve from Chroma
        results = collection.query(
            query_texts=[query],
            n_results=3
        )

        retrieved_docs = " ".join(results["documents"][0])

        prompt = f"""
        Answer the question using the context below.

        Context:
        {retrieved_docs}

        Question:
        {query}
        """

        input_tokens = count_tokens(prompt)

        # PROVIDER CALL
        if provider == "Groq":
            client = Groq(api_key=api_key)
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            response = completion.choices[0].message.content

        else:
            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            response = completion.choices[0].message.content

        output_tokens = count_tokens(response)

    # Save History
    st.session_state.history.append((query, response))

    # Display Response
    st.write("### 🤖 Response")
    st.write(response)

    # Token Monitoring
    st.write("### 📊 Token Usage")
    st.write(f"Input Tokens: {input_tokens}")
    st.write(f"Output Tokens: {output_tokens}")
    st.write(f"Total Tokens: {input_tokens + output_tokens}")

elif query and not api_key:
    st.warning("⚠️ Please enter your API key in sidebar.")

# -------------------------
# CHAT HISTORY DISPLAY
# -------------------------
if st.session_state.history:
    st.write("### 📝 Chat History")
    for q, r in st.session_state.history:
        st.write(f"**You:** {q}")
        st.write(f"**Bot:** {r}")


#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
import streamlit as st
from groq import Groq
import time
import os

GROQ_API_KEY = os.getenv("api")
client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(page_title="Groq AI Chat", page_icon="🤖")

st.title("🤖 Groq AI Assistant")
st.write("Streaming response with animation ✨")

# -----------------------
# SESSION MEMORY
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------
# DISPLAY CHAT HISTORY
# -----------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown("### 👤 You")
        st.write(msg["content"])
    else:
        st.markdown("### 🤖 Assistant")
        st.write(msg["content"])

st.markdown("---")

# -----------------------
# USER INPUT
# -----------------------
prompt = st.text_input("Ask something...")

if st.button("Send") and prompt:

    # Store user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    # Show user message immediately
    st.markdown("### 👤 You")
    st.write(prompt)

    st.markdown("### 🤖 Assistant")

    message_placeholder = st.empty()
    full_response = ""

    # Streaming Response
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=st.session_state.messages,
        temperature=0.7,
        stream=True,
    )

    for chunk in completion:
        if chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.02)

    message_placeholder.markdown(full_response)

    # Save assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )

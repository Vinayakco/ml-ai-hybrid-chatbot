import streamlit as st
import time
from chatbot import get_response
import streamlit as st
from chatbot import get_response

# Page config
st.set_page_config(page_title="ML Chatbot", page_icon="🤖")

# Title
st.title("🤖 VIMI -AI")
st.caption("Trained with TF-IDF + Logistic Regression")

# Chat history initialize karo
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hi! I am ML Chatbot. How can I help you? 😊"
    })

# Purane messages dikhao
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    # User message dikhao
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    with st.chat_message("user"):
        st.write(user_input)

   # Bot response lo
    response = get_response(user_input)

    # Streaming effect ke liye generator function
    def stream_response():
        for word in response.split(" "):
            yield word + " "
            time.sleep(0.03)

    # Bot message dikhao (word by word)
    with st.chat_message("assistant"):
        full_response = st.write_stream(stream_response())

    # History mein save karo
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })
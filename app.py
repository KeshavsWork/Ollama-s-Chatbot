import streamlit as st
import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

 
# Load environment variables
 
load_dotenv()

# (Optional) LangSmith tracing – learning/debugging only
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With Ollama"

 
# Streamlit App Title
 
st.title("Generative AI Chat Assistant")

 
# Sidebar Controls
 
llm_model = st.sidebar.selectbox(
    "Select Open Source Model",
    [ "mistral","llama3:latest", "gemma2:latest"]
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.7
)

max_tokens = st.sidebar.slider(
    "Max Tokens",
    min_value=50,
    max_value=300,
    value=150
)

 
# Session-level Memory
 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

 
# Response Generation Function
 
def generate_response(question, model_name, history):
    """
    Generates response using session-level conversational context.
    This is a learning-level memory implementation.
    """

    llm = Ollama(
        model=model_name,
        temperature=temperature
    )

    output_parser = StrOutputParser()

    # Build messages dynamically using previous conversation
    messages = [
        ("system", "You are a helpful assistant. Answer clearly and concisely.")
    ]

    # Add previous conversation (session memory)
    for user_q, assistant_a in history:
        messages.append(("user", user_q))
        messages.append(("assistant", assistant_a))

    # Add current user question
    messages.append(("user", question))

    prompt = ChatPromptTemplate.from_messages(messages)

    chain = prompt | llm | output_parser
    response = chain.invoke({})

    return response

 
# User Input Section
 
st.write("Ask a question below:")

user_input = st.text_input("You:")

 
# Generate & Display Response
 
if user_input:
    answer = generate_response(
        user_input,
        llm_model,
        st.session_state.chat_history
    )

    # Store conversation in session memory
    st.session_state.chat_history.append((user_input, answer))

    st.markdown(f"**Bot:** {answer}")

 
# Display Conversation History
 
if st.session_state.chat_history:
    st.subheader("Conversation History")
    for q, a in st.session_state.chat_history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")

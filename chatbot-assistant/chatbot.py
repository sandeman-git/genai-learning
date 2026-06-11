from dotenv import load_dotenv
import streamlit as st
import os

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# --------------------------
# Load Environment Variables
# --------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLEKEY")


# --------------------------
# Gemini Model
# --------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", ## change model based on requirement
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    max_output_tokens=200          ## Optional, Change output token size if required
)


# --------------------------
# Prompt Template
# --------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant. "
            "Answer clearly and remember previous conversation."
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ]
)

chain = prompt | llm


# --------------------------
# Memory Store
# --------------------------
if "store" not in st.session_state:
    st.session_state.store = {}


def get_session_history(session_id: str):

    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()

    return st.session_state.store[session_id]



# --------------------------
# Add Memory to Chain
# --------------------------
chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)


# --------------------------
# Streamlit UI
# --------------------------
st.title("Gemini LLM Chatbot")

session_id = "user_1"

if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat Input
user_question = st.chat_input("Ask a question")


if user_question:

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    # Invoke chatbot with memory
    response = chatbot.invoke(
        {
            "question": user_question
        },
        config={
            "configurable": {
                "session_id": session_id
            }
        }
    )

    answer = response.content

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

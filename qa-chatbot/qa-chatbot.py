from dotenv import load_dotenv
import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLEKEY")


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  ## change model based on requirment
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    max_output_tokens=500
)


def LLM_Response(question):
    response = llm.invoke(question)
    return response.content

### simple streamlit application


st.title("Simple Q&A Chatbot")

question = st.text_input("Enter your question:")

if st.button("Ask"):
    if question:
        answer = LLM_Response(question)
        st.subheader("Answer:")
        st.write(answer)
    else:
        st.warning("Please enter a question.")


## strealit application

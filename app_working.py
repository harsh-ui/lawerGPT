import streamlit as st
import os
import pandas as pd
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI, OpenAIChat
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from htmlTemplates import css, bot_template, user_template
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
import openai
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from tempfile import NamedTemporaryFile
import re

# Layout as wide and adding custom title
st.set_page_config(page_title = "LegalGPT", layout = "wide")

os.environ["OPENAI_API_KEY"] = "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"
openai_api_key = "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"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# Function to get OpenAI completion
def get_completion(prompt, model="gpt-3.5-turbo-16k"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

# def create_agent(dataframe):
#     # if dataframe is not None:
#     agent = create_csv_agent(
#         OpenAIChat(model = "gpt-4", temperature = 0),
#         dataframe,
#         verbose = True,
#         pandas_kwargs = {'encoding': "unicode_escape"}
#     )
#     return agent

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html = True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html = True)

prompt = """
    As an experienced legal analyst, your task is to use the Context and the question and provide an appropriate response.

    Compose a case brief using only information from the Context. 
    Working step by step, organize the Context outline into well-structured 
    paragraphs in the following sections:

    Case: [name of the case, court, year]
    Questions Presented: The issues the Court must resolve.
    Facts of the Case: the parties, facts and events leading to this case.
    Procedural History: [district court case summary, appeals court case summary, 
    how this issue reached this Court]
    Analysis: subsections
       ["Rules": detailed explanation of how the Court's considers 
       relevant statutes, interpretations, standards, and tests applicable 
       to this case, 
       "case law": names of cases reviewed by the Court and analysis of how 
       the Court relates those cases to the Questions Presented, 
       "Application": detailed explanation of how the Rules and Case Law 
       help the Court reach its conclusions]
    Conclusion: the Court's ruling on the Questions Presented.
    Dissent:  How it disagrees with the holding of this case.}

"""

# Streamlit app
def main():
    # st.title("LegalGPT")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        
    tab1, tab2, tab3 = st.tabs(["Summarization", "LegalBot", "KeyWord"])
    with st.sidebar:
        # Upload file
        uploaded_file = st.file_uploader("Upload File", type = ["pdf"], accept_multiple_files = True)
    if uploaded_file:
        # get pdf text
        raw_text = get_pdf_text(uploaded_file)
        # get the text chunks
        text_chunks = get_text_chunks(raw_text)
        # create vector store
        vectorstore = get_vectorstore(text_chunks)

        with tab1:
            if st.button("Summarize"):
               query = f"""{prompt}"""
               # completion llm
               llm = ChatOpenAI(
                                model_name = 'gpt-4', 
                                temperature = 0.0)
               qa = RetrievalQA.from_chain_type(
                                                llm = llm, 
                                                chain_type = "stuff", 
                                                retriever = vectorstore.as_retriever(search_type = "similarity", search_kwargs = {"k": 20})
                                                )
               response = qa.run(query)
               st.write(response)

        with tab2:

            user_question = st.text_input("Ask a question about your documents:")
            if user_question:
                handle_userinput(user_question)

            # create conversation chain
            st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == "__main__":
    main()

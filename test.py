import streamlit as st
import time
import json
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

from dotenv import load_dotenv
load_dotenv()

st.title("test 01")


if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

loader = PyPDFLoader("grade-11-history-text-book.pdf")
data = loader.load() 

def find_section(text):
    match = re.search(r'(Chapter|Section)\s+\d+[:.\s-]*(.*)', text, re.IGNORECASE)
    if match:
        return match.group(0)
    section = text.split('\n')[0] 
    return section

for doc in data:
    section = find_section(doc.page_content)
    page = doc.metadata.get("page")
    doc.metadata["section"] = section



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3,max_tokens=500)

promt = st.chat_input("Say Something: ")
if promt:
    st.chat_message('user').markdown(promt)
    st.session_state.messages.append({'role':'user','content':promt})

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Use four sentences maximum and keep the answer concise.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}")
])





if promt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    response = rag_chain.invoke({"input": promt})
    st.write(response["answer"])
retrieved_docs = retriever.get_relevant_documents(prompt)
for doc in retrieved_docs:
    page = doc.metadata.get("page", "unknown")
    section = doc.metadata.get("section", "unknown")
    st.write(f"📄 Page: {page}, 🧩 Section: {section}")

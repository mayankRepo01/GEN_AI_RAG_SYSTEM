import boto3
import streamlit as st
import os

import uuid

## Bedrock embedding

os.environ['AWS_ACCESS_KEY_ID'] = '<AWS_ACCESS_KEY_ID>'
os.environ['AWS_SECRET_ACCESS_KEY'] = '<AWS_SECRET_ACCESS_KEY>'

from langchain_community.embeddings import BedrockEmbeddings

# textSpliter

from langchain.text_splitter import RecursiveCharacterTextSplitter

# pdf loader
from langchain_community.document_loaders import PyPDFLoader

# vector store creator
from langchain_community.vectorstores import FAISS

# bedrock
from langchain.llms.bedrock import Bedrock

# prompts and chains

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# s3 client

s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

bedrock_client = boto3.client(service_name="bedrock-runtime", region_name='us-east-1')

bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)


# generate a uuid
def get_unique_id():
    return str(uuid.uuid4())


folder_path = "/tmp/"


def load_index():
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"{folder_path}my_faiss.faiss")
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename=f"{folder_path}my_faiss.pkl")


def get_llm():
    llm = Bedrock(model_id="amazon.titan-text-express-v1", client=bedrock_client,
                  model_kwargs={"maxTokenCount": 512})
    return llm


def get_response(llm, vectorstore, question):
    prompt_template = """
    
    Human: Please use the given context to provide concise answer to the question
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant:"""
    Prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", chain_type_kwargs={"prompt": Prompt},
                                     retriever=vectorstore.as_retriever(search_type="similarity",
                                                                        search_kwargs={"k": 5})
                                     , return_source_documents=True, )
    answer = qa({"query": question})
    return answer["result"]


def main():
    st.header("This is Client Site for Chat with PDF Reader using bedrock, rag")
    load_index()

    dir_list = os.listdir(folder_path)
    st.write(f"Files and Directories in {folder_path}")
    st.write(dir_list)

    faiss_index = FAISS.load_local(
        index_name="my_faiss",
        folder_path=folder_path,
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization=True
    )

    st.write("index is ready")
    question = st.text_input("Please Ask your Question")
    if st.button("Ask Question"):
        with st.spinner("Querying..."):
            llm = get_llm()
            st.write(get_response(llm=llm, vectorstore=faiss_index, question=question))
            st.success("Done")


if __name__ == "__main__":
    main()

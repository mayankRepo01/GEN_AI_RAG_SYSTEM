import boto3
import streamlit as st
import os

import uuid

## Bedrock embedding

os.environ['AWS_ACCESS_KEY_ID'] = '<AWS_ACCESS_KEY_ID>'
os.environ['AWS_SECRET_ACCESS_KEY'] = '<AWS_SECRET_ACCESS_KEY>'

from langchain_community.embeddings import BedrockEmbeddings

## textSpliter

from langchain.text_splitter import RecursiveCharacterTextSplitter

## pdf loader
from langchain_community.document_loaders import PyPDFLoader

## vector store creator
from langchain_community.vectorstores import FAISS

## s3 client

s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

bedrock_client = boto3.client(service_name="bedrock-runtime",region_name='us-east-1')

bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)


##generate a uuid
def get_unique_id():
    return str(uuid.uuid4())


## split the pages into text chunks
def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs


def create_vector_store(request_id, splitted_docs):
    vector_store_faiss = FAISS.from_documents(splitted_docs, bedrock_embeddings)
    file_name = f"{request_id}.bin"
    folder_path = "/tmp/"
    vector_store_faiss.save_local(index_name=file_name, folder_path=folder_path)
    ## upload to s3
    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".faiss", Bucket=BUCKET_NAME, Key="my_faiss.faiss")
    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".pkl", Bucket=BUCKET_NAME, Key="my_faiss.pkl")
    return True


def main():
    st.write("This is Admin site for Chat with PDF Reader")
    uploaded_file = st.file_uploader("choose a file", "pdf")
    if uploaded_file is not None:
        request_id = get_unique_id()
        ##f is creating a formated-string
        st.write(f"Request_id : {request_id}")
        saved_file_name = f"{request_id}.pdf"
        with open(saved_file_name, mode="wb") as w:
            w.write(uploaded_file.getvalue())
        pdf_loader = PyPDFLoader(saved_file_name)
        pages = pdf_loader.load_and_split()
        st.write(f"Total Pages : {len(pages)}")
        splitted_docs = split_text(pages, 1000, 200)
        st.write(f"Splitted docs length : {len(splitted_docs)} ")
        st.write("============================================")
        st.write(splitted_docs[0])
        st.write("============================================")
        st.write(splitted_docs[1])

        st.write("Creating a vector store")
        is_saved=create_vector_store(request_id, splitted_docs)
        if is_saved:
            st.write("vctorstore is created, data is saved in the S3 bucket")
        else:
            st.write("vctorstore creation failed")


if __name__ == "__main__":
    main()

Aws Resources will be required for running this RAG System

S3 Bucket = genai-bedrock-data-bucket
Model Details
AWS Credentials


`cd user`

``docker build -t pdf_reader_user .``

`docker run -e BUCKET_NAME=genai-bedrock-data-bucket -v ~/.aws:/root/.aws/ -p 32000:8083 -it pdf_reader_user`
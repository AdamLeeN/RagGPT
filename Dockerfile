# syntax=docker/dockerfile:1
FROM alpine as builder

WORKDIR /app

# wget embedding model weight from alpine (does not exist from slim-buster)
RUN wget "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz" -O - | \
    tar -xzf - -C /app
    
RUN mkdir -p /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2
COPY /app/onnx /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx

FROM python:3.11-slim-bookworm as base

ENV RAG_EMBEDDING_MODEL="all-MiniLM-L6-v2"
# device type for whisper tts and embbeding models - "cpu" (default), "cuda" (nvidia gpu and CUDA required) or "mps" (apple silicon) - choosing this right can lead to better performance
ENV RAG_EMBEDDING_MODEL_DEVICE_TYPE="cpu"
ENV RAG_EMBEDDING_MODEL_DIR="/app/data/cache/embedding/models"
ENV SENTENCE_TRANSFORMERS_HOME $RAG_EMBEDDING_MODEL_DIR

ENV OPENAI_API_BASE_URL "https://api.adamchatbot.chat/v1"
ENV OPENAI_API_KEY "sk-OQyJrrKA7y7g4vdUCbDcAf768bB7468d8153BfD93b37Cf42"


ENV SCARF_NO_ANALYTICS true
ENV DO_NOT_TRACK true

######## Preloaded models ########




COPY ./requirements.txt ./requirements.txt


RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu 
RUN pip3 install -r requirements.txt

# Install pandoc and netcat
# RUN python -c "import pypandoc; pypandoc.download_pandoc()"
RUN apt-get update \
    && apt-get install -y pandoc netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# preload embedding model
RUN python -c "import os; from chromadb.utils import embedding_functions; sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=os.environ['RAG_EMBEDDING_MODEL'], device=os.environ['RAG_EMBEDDING_MODEL_DEVICE_TYPE'])"


# copy backend files
COPY . .

CMD [ "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8080" ]

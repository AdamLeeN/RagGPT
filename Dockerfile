# syntax=docker/dockerfile:1
FROM alpine as builder

WORKDIR /app

# wget embedding model weight from alpine (does not exist from slim-buster)
RUN wget "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz" -O - | \
    tar -xzf - -C /app





FROM python:3.11-slim-bookworm as base



ENV OPENAI_API_BASE_URL "https://api.adamchatbot.chat/v1"
ENV OPENAI_API_KEY "sk-OQyJrrKA7y7g4vdUCbDcAf768bB7468d8153BfD93b37Cf42"


ENV SCARF_NO_ANALYTICS true
ENV DO_NOT_TRACK true

######## Preloaded models ########




COPY ./requirements.txt ./requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


RUN pip3 install -r requirements.txt --no-cache-dir

# Install pandoc and netcat
# RUN python -c "import pypandoc; pypandoc.download_pandoc()"
RUN apt-get update \
    && apt-get install -y pandoc netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# preload embedding model
RUN python -c "import os; from chromadb.utils import embedding_functions; sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=os.environ['RAG_EMBEDDING_MODEL'], device=os.environ['RAG_EMBEDDING_MODEL_DEVICE_TYPE'])"
# preload tts model
RUN python -c "import os; from faster_whisper import WhisperModel; WhisperModel(os.environ['WHISPER_MODEL'], device='auto', compute_type='int8', download_root=os.environ['WHISPER_MODEL_DIR'])"


# copy backend files
COPY . .

CMD [ "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8080" ]

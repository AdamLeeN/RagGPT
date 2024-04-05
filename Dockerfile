# syntax=docker/dockerfile:1


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
# whisper TTS Settings
ENV WHISPER_MODEL="base"
ENV WHISPER_MODEL_DIR="/app/backend/data/cache/whisper/models"

# RAG Embedding Model Settings
# any sentence transformer model; models to use can be found at https://huggingface.co/models?library=sentence-transformers
# Leaderboard: https://huggingface.co/spaces/mteb/leaderboard 
# for better persormance and multilangauge support use "intfloat/multilingual-e5-large" (~2.5GB) or "intfloat/multilingual-e5-base" (~1.5GB)
# IMPORTANT: If you change the default model (all-MiniLM-L6-v2) and vice versa, you aren't able to use RAG Chat with your previous documents loaded in the WebUI! You need to re-embed them.
ENV RAG_EMBEDDING_MODEL="all-MiniLM-L6-v2"
# device type for whisper tts and embbeding models - "cpu" (default), "cuda" (nvidia gpu and CUDA required) or "mps" (apple silicon) - choosing this right can lead to better performance
ENV RAG_EMBEDDING_MODEL_DEVICE_TYPE="cpu"
ENV RAG_EMBEDDING_MODEL_DIR="/app/backend/data/cache/embedding/models"
ENV SENTENCE_TRANSFORMERS_HOME $RAG_EMBEDDING_MODEL_DIR


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

CMD [ "uvicorn", "main:app", "--host", "localhost", "--port", "8008" ]

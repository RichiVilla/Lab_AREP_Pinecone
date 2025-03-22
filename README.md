# Guía para Implementar RAG con Pinecone y LangChain

## Introducción

Este documento te guiará paso a paso en la implementación de un sistema **Retrieval-Augmented Generation (RAG)** utilizando **LangChain** y **Pinecone**. Con este enfoque, podrás mejorar la capacidad de generación de respuestas de modelos como **GPT-4** al proporcionarles contexto relevante desde una base de conocimientos vectorizada.

---

## Requisitos Previos

Antes de comenzar, asegúrate de contar con los siguientes requisitos:

- **Python** 3.8 o superior
- Un **entorno virtual**
- Una cuenta en **Pinecone** y su API Key
- Una **OpenAI API Key**

---

## Configuración de Credenciales
Para utilizar OpenAI y Pinecone, es necesario configurar las credenciales:

```
import getpass
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
```
Aquí, se solicita la API Key de OpenAI de manera segura utilizando getpass.getpass().

## Inicialización de Modelos
Cargar el modelo de lenguaje GPT-4
```
from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
```
Aquí inicializamos el modelo GPT-4o-mini, que servirá como el generador de respuestas del sistema.

Cargar el modelo de embeddings
```
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```
El modelo text-embedding-3-large se usa para convertir el texto en representaciones vectoriales.


## Creación de un nuevo índice
```
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="TU_API_KEY")

index_name = "quickstart"

pc.create_index(
    name=index_name,
    dimension=3072, # Reemplázalo con la dimensión de tu modelo
    metric="cosine", # Métrica utilizada para la comparación
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

index = pc.Index(index_name)
vector_store = PineconeVectorStore(embedding=embeddings, index=index)
```

![image](https://github.com/user-attachments/assets/4062d2eb-8e5b-404c-96dd-fd8ce35aa8e4)

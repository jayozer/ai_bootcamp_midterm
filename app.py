import os
import openai
import chainlit as cl
from llama_index.core import Settings, ServiceContext
from llama_index.core.callbacks import CallbackManager
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
import qdrant_client
# from llama_index.core import Settings
# from llama_index.core.indices.vector_store.base import VectorStoreIndex
# from llama_index.vector_stores.qdrant import QdrantVectorStore
# from llama_index.llms.openai import OpenAI
# from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
# from llama_index.core.callbacks import CallbackManager
# from llama_index.core.service_context import ServiceContext
# from llama_index.postprocessor.flag_embedding_reranker import (
#     FlagEmbeddingReranker,
# )

import qdrant_client

openai.api_key = os.environ.get("OPENAI_API_KEY")

# Connect to Qdrant vector database
client = qdrant_client.QdrantClient(
    url="https://24f997d6-02df-4889-a352-1eac83e0bd37.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key=os.environ["QDRANT_API_KEY"],
)

# Load the existing index from Qdrant
try:
    vector_store = QdrantVectorStore(client=client, collection_name="meta_10k_filingsV5")
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
except Exception as e:
    print(f"Failed to load the index from Qdrant: {e}")

@cl.on_chat_start
async def start():
    Settings.llm = OpenAI(model="gpt-3.5-turbo-0125", temperature=0, max_tokens=1024, streaming=True)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.context_window = 1536
    service_context = ServiceContext.from_defaults(callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]))
    
    #initialize reranker
    reranker = FlagEmbeddingReranker(
    top_n=5,
    model="BAAI/bge-reranker-large",
    )

    query_engine = index.as_query_engine(streaming=True,
        similarity_top_k=15, node_postprocessors=[reranker], verbose=True, service_context=service_context
    )

    cl.user_session.set("query_engine", query_engine)
    await cl.Message(author="Assistant", content="Hello! I'm an AI assistant. How may I help you?").send()

@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")  # type: RetrieverQueryEngine
    msg = cl.Message(content="", author="Assistant")
    res = await cl.make_async(query_engine.query)(message.content)
    for token in res.response_gen:
        await msg.stream_token(token)
    await msg.send()
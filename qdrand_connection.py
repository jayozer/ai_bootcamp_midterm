import os
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore

import qdrant_client

client = qdrant_client.QdrantClient(
    "https://24f997d6-02df-4889-a352-1eac83e0bd37.us-east4-0.gcp.cloud.qdrant.io:6333",
    #api_key="In5f621EE_6yUq2lI8pBArpMNGOzBDrfDyYHiKAQlFNjWoC1mUeK3A", # For Qdrant Cloud, None for local instance
    api_key = os.environ["QDRANT_API_KEY"]
)

vector_store = QdrantVectorStore(client=client, collection_name="meta_10k_filings")
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

query_engine = index.as_query_engine()
response = query_engine.query("What was the total value of 'Cash and cash equivalents' as of December 31, 2023?")
print(response)
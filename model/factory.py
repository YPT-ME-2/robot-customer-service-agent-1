from langchain_ollama import OllamaEmbeddings,ChatOllama

embed_model=OllamaEmbeddings(
    model="qwen3-embedding:4b",
    client_kwargs={"trust_env": False},
)

chat_model=ChatOllama(
    model="qwen3.5:9b",
    reasoning=False,
    client_kwargs={"trust_env": False},
)


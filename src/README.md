# Core RAG Implementation
```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

def initialize_rag_pipeline(persist_directory):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_to_directory=persist_directory, embedding_function=embeddings)
    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return qa_chain
```
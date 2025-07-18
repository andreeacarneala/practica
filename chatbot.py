from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA

loader = PyPDFLoader("learning.pdf") 
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
chunks = splitter.split_documents(documents)

embedding = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(chunks, embedding)

llm = OllamaLLM(model="mistral")  
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

while True:
    q = input("\nTu: ")
    if q.lower() in ["exit", "quit"]:
        break
    a = qa.invoke({"query": q})
    print(f"\nchatbot: {a['result']}")

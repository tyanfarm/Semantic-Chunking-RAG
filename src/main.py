from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate  
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load file
loader = PyPDFLoader("2305.07866v2.pdf")
documents = loader.load()

# Embedding Model
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Chunker
semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type="percentile")

semantic_chunks = semantic_chunker.create_documents([d.page_content for d in documents])

# Create Vectorstore from chunks & embed them
semantic_chunk_vectorstore = Chroma.from_documents(semantic_chunks, embedding=embed_model)

# Retrieval
# search_kwargs == topK ratio => return 1 element nearest
semantic_chunk_retriever =  semantic_chunk_vectorstore.as_retriever(search_kwargs={"k": 1})

# ---- Augmentation Step ----
# System Prompt
system_template = """\
Use the following context to answer the user's query. If you cannot answer, please respond with 'I don't know'.

User's Query:
{question}

Context:
{context}    
"""

system_prompt = ChatPromptTemplate.from_template(system_template)

# ---- LLM Generation Step ----
llm_model = Ollama(
    model="llama3.2:1b",
    temperature=0,
    base_url="http://localhost:11434" 
)

# ---- RAG Pipeline ----
semantic_rag_chain = (
    {"context" : semantic_chunk_retriever, "question" : RunnablePassthrough()}
    | system_prompt
    | llm_model
    | StrOutputParser()
)

response = semantic_rag_chain.invoke("What is the introduction of GPFedRec?")
print(response)
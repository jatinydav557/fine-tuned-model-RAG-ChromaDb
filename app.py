import os
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import json

# Load environemnt variable
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Chroma directory path
project_root = os.path.dirname(os.path.abspath(__file__))
chroma_path = os.path.join(project_root, "Chroma_db")

# Load fine-tuned embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name=os.path.join(project_root, "finetune", "fine-tuned-model-rag"),
    model_kwargs={"device": "cpu"} #We have to mention it to ignore warnings
)

# Load Chroma vector store
vectordb = Chroma(
    persist_directory=chroma_path,
    embedding_function=embedding_model
)

#  Create retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

#  Load Groq LLM
llm = ChatGroq(
    temperature=0,
    model_name="llama3-70b-8192",
    api_key=groq_api_key
)

# 🧾 Prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant that finds and recommends quotes.

Use ONLY the following context (quote + author + tags) to answer the user's question.

⚠️ STRICTLY output your answer in this JSON format, and NOTHING ELSE:

{{ 
  "quote": "The quote here", 
  "author": "Author Name", 
  "tags": ["tag1", "tag2"] 
}}

NO explanations, NO introductions, NO markdown, NO extra text.

Context:
{context}

Question: {question}
"""
)

# 🔗 Build RAG chain 
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

# Streamlit UI
st.set_page_config(page_title="📑  QuoteBot", layout="centered")
st.title("💬 QuoteBot – Ask for a Quote")

query = st.text_input("🔎Enter your quote request (e.g., 'Show me quotes about courage by women authors')")

show_sources = st.checkbox("📌 Show Retrieved Context & Similarity", value=True)

if st.button("Get Quote") and query:
    with st.spinner("🤖 Thinking..."):
        #Retrieve documents
        retrieved_docs = retriever.invoke(query)

        # Generate final response
        result = rag_chain.invoke(query)

        # Showing the result
        st.success("✅ Here's a quote based on relevant context from ChromaDB:")
        st.json(result['result'])  # Expecting LLM to output JSON
        # LangChain response is a dict	Reason -- Streamlit renders dicts as JSON

        #  Download button for JSON
        st.download_button(
            label="📥 Download Quote as JSON",
            data=json.dumps(result, indent=2),
            file_name="quote.json",
            mime="application/json"
        )

        # Showing the retrieved documents
        if show_sources:
            st.markdown("---")
            st.subheader("📑 Retrieved Quotes")
            for i, doc in enumerate(retrieved_docs, 1):
                st.markdown(f"**{i}.** {doc.page_content}")
                if doc.metadata:
                    st.caption(f"Tags: {doc.metadata.get('tags', 'N/A')}")



import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.title("RAG Book Assistant")
st.write("Upload a PDF and ask questions from the document")

# Upload PDF
uploaded_file = st.file_uploader("Upload the PDF book", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully!")

    # Button to create Vector DB
    if st.button("Create Vector Database"):
        with st.spinner("Processing the document..."):
            # 1. Load Document
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()

            # 2. Split Document into Chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)

            # 3. Embeddings Model
            embedding_model = MistralAIEmbeddings(model = "mistral-embed")

            # 4. Store in Chroma Vector DB
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory="vector-storage",
            )
            vectorstore.persist()

            st.success("Vector database created successfully!")

    # Ask Questions
    query = st.text_input("Ask questions from the book")

    if query:
        # Load the existing database
        embedding_model = MistralAIEmbeddings(model = "mistral-embed")
        vectorstore = Chroma(
            persist_directory="vector-storage", embedding_function=embedding_model
        )

        # MMR Retriever
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
        )

        # Retrieve relevant chunks
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Setup LLM & Prompt
        llm = ChatMistralAI(model="ministral-14b-2512")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context, say: "I could not find the answer in the document."
""",
                ),
                ("human", "Context:\n{context}\n\nQuestion: {question}"),
            ]
        )

        final_prompt = prompt.invoke({"context": context, "question": query})
        response = llm.invoke(final_prompt)

        st.markdown("### AI Answer:")
        st.write(response.content)
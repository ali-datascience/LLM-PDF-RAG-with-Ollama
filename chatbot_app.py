import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.agents import initialize_agent, Tool
from langchain.tools import tool
import logging
import re

local_path = "sample_policy_doc_AU1234.pdf"

# Load the PDF file
if local_path:
    loader = PyMuPDFLoader(local_path)
    data = loader.load()
else:
    print("Upload a PDF file for processing.")

# Split and chunk the data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
chunks = text_splitter.split_documents(data)
chroma_persist_directory = "chroma_db"

# Add the chunks to vector database
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
    collection_name="police-rag",
    persist_directory=chroma_persist_directory,
)

local_llm = "llama3.1"
llm = ChatOllama(model=local_llm)

# Define the query prompt template
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI assistant. Provide clear, concise, and direct answers that address the customer's question in no more than 300 words. 
    Retrieve relevant documents from a vector database and summarize the response briefly. 
    Ensure the text is well-structured, easy to understand, and includes proper punctuation. 
    Avoid unnecessary details, references to pages, sections, or any other metadata.
    Original question: {question}"""
)

# Multi-query retriever setup
retriever = MultiQueryRetriever.from_llm(vector_db.as_retriever(), llm, prompt=QUERY_PROMPT)

def clean_text(text: str) -> str:
    """Function to clean unwanted content like page numbers or section references."""
    # Remove references like "ADH 15.10a 2 Section Page Buildings"
    cleaned_text = re.sub(r"ADH \d+\.\d+[a-z]? \d+ Section Page.*?(\d{1,2}|\d+\.?\d*)", "", text)
    cleaned_text = re.sub(r"\d+ \w{1,2} \d+.*?(\d{1,2}|\d+\.?\d*)", "", cleaned_text)  # Remove page number patterns
    # Remove unwanted formatting, headers, or other irrelevant data
    cleaned_text = re.sub(r"^\s*[\d\-]+\s*[A-Za-z]+.*$", "", cleaned_text, flags=re.MULTILINE)
    return cleaned_text.strip()

@tool("ask_pdf")
def ask_pdf_tool(question: str) -> str:
    """A tool to query the PDF content for relevant answers."""
    # Perform the retrieval process directly here
    results = retriever.get_relevant_documents(question)
    
    # Clean the text to remove unwanted content
    cleaned_results = " ".join([clean_text(result.page_content) for result in results])

    # Truncate the response to 300 words
    cleaned_results = ' '.join(cleaned_results.split()[:300])

    return cleaned_results

tools = [
    Tool(name="PDF Query Tool", func=ask_pdf_tool, description="Tool to query PDF data")
]

logging.basicConfig(level=logging.CRITICAL)

# Chatbot with history
history = []

def chatbot(query: str):
    # Get the response from the tool
    response = ask_pdf_tool(query)
    
    # Add the response to the history
    history.append(f"Agent: {response}")
    
    # Return the response with the history
    return "\n".join(history)

# Streamlit Web Interface
def main():
    # Header Section
    st.title("Home Insurance")
    st.markdown(
        "Use this chatbot to ask questions about the Insurance. "
        "This application provides precise answers to your queries."
    )
    st.divider()

    # Sidebar Section
    st.sidebar.markdown("**Insurance Chat:**")
    st.sidebar.markdown("- Chat Bot")
    st.sidebar.markdown(
        "Built with ❤️"
    )

    # Chat interface
    st.write("## Chat with the Assistant")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Input form
    with st.form(key="chat_form", clear_on_submit=True):
        user_query = st.text_input("Ask a question:", placeholder="Type your question about the document...")
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_query:
        # Append user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Get chatbot response
        response = chatbot(user_query)

        # Append bot message to chat history
        st.session_state.chat_history.append({"role": "bot", "content": response})

    # Display chat messages
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(
                f"""
                <div style="text-align: right; background-color: #DCF8C6; padding: 10px; 
                border-radius: 10px; margin: 5px; display: inline-block;">
                    {chat['content']}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="text-align: left; background-color: #F1F0F0; padding: 10px; 
                border-radius: 10px; margin: 5px; display: inline-block;">
                    {chat['content']}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Footer Text
    st.divider()
    st.markdown(
        "<footer style='text-align: center; color: gray; font-size: small;'>"
        "© 2024 Powered by ❤️"
        "</footer>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()

import streamlit as st
import os 
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Set page config
st.set_page_config(page_title="PDF Q&A Assistant", page_icon="📚", layout="wide")

# function for extracting texts from pdfs
def pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# function for getting chunks
def text_chunks(text):
    if text:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        return chunks
    return []


import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# function for embeddings
def get_embeddings(text):
    if google_api_key:
        chunks = text_chunks(text)  # Fixed: properly call the function with text parameter
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)  # Fixed: use chunks instead of text_chunks
        # Remove local saving for cloud deployment - store only in memory/session state
        return vector_store
    return None

# function for chains
def get_chains():
    prompt = """ You are a helpful AI tutor designed to assist a 15-year-old student in understanding content from a school textbook (provided in context).

Your job is to **only use the given context** to answer the student's question.

Follow these strict rules:

###  Context Rules
- ONLY answer using the information provided in the context.
- If the answer is not found in the context, say:
   “The context for this question is not provided.”
- Do NOT make up answers or guess from general knowledge.

###  Style Guidelines
- Use **simple, age-appropriate language** (as if you're explaining to a 15-year-old).
- Use **day-to-day examples or analogies** for technical/conceptual explanations.
- Avoid jargons, slangs, or complex terms.
- If the topic is from **Social Science**, explain it in **detailed, structured paragraphs with 100 to 300 words in depth of the topic**.
- Keep answers structured: ** Introduction → Explanation →Example or Conclusion**.

###  Question Handling Instructions
- If asked to create **multiple choice questions (MCQs)**:
  - Provide exactly **4 answer options** for each question.
- If asked to **generate questions**, do NOT provide answers unless the user requests them.
- For **Maths-related questions**, only provide a solution **if the student explicitly asks**.
- Stick to the topic and do not drift from the original question.

---
<context>
{context}
</context>

---

💬 Question: {question}

🧠 Your Answer:

    

    context:\n {context}\n
    question:\n {question}\n

    Answer:
    """

    if google_api_key:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key, temperature=0.3)
        prompt_template = PromptTemplate(template=prompt, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt_template)
        return chain
    return None

# Main Streamlit App
def main():
    st.title("📚 PDF Q&A Assistant")
    st.write("Upload PDF files and ask questions about their content!")
    
    # Privacy notice for cloud deployment
    st.info("🔒 **Privacy Notice**: Your PDFs are processed temporarily in memory and are not stored permanently. Session data is cleared when you close the app.")
    
    # Check if API key is configured
    if not google_api_key:
        st.error("❌ Please set your GOOGLE_API_KEY in the .env file")
        st.stop()
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📄 Upload PDF Files")
        
        # File size warning
        st.warning("⚠️ **File Size Limit**: Keep PDFs under 50MB for optimal performance")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files", 
            type="pdf", 
            accept_multiple_files=True,
            help="Upload one or more PDF files to analyze (Max 50MB per file)"
        )
        
        if uploaded_files:
            # Check file sizes
            total_size = sum(len(file.read()) for file in uploaded_files)
            for file in uploaded_files:
                file.seek(0)  # Reset file pointer after reading
            
            total_size_mb = total_size / (1024 * 1024)
            
            if total_size_mb > 50:
                st.error(f"❌ Total file size ({total_size_mb:.1f}MB) exceeds 50MB limit. Please upload smaller files.")
            else:
                st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully! (Total: {total_size_mb:.1f}MB)")
                
                # Process PDFs button
                if st.button("🔄 Process PDFs", type="primary"):
                    with st.spinner("Processing PDFs... This may take a moment."):
                        
                        # Extract text from PDFs
                        raw_text = pdf_text(uploaded_files)
                        
                        if raw_text:
                            st.success("✅ Text extracted successfully!")
                            
                            # Create embeddings and store in vector database
                            vector_store = get_embeddings(raw_text)
                            
                            if vector_store:
                                # Store in session state (in-memory storage for cloud deployment)
                                st.session_state['vector_store'] = vector_store
                                st.session_state['processed'] = True
                                st.session_state['pdf_names'] = [file.name for file in uploaded_files]
                                st.success("✅ PDFs processed and stored in memory!")
                            else:
                                st.error("❌ Failed to create vector store")
                        else:
                            st.error("❌ No text could be extracted from the PDFs")
        
        # Clear memory button
        st.subheader("🧹 Memory Management")
        if st.button("Clear Session Data", help="Clear all processed data to free up memory"):
            st.session_state.clear()
            st.success("✅ Session data cleared!")
            st.rerun()
    
    # Main content area
    if 'processed' in st.session_state and st.session_state['processed']:
        st.header("💬 Ask Questions About Your PDFs")
        
        # Show processed files
        if 'pdf_names' in st.session_state:
            st.write("📋 **Processed Files:**", ", ".join(st.session_state['pdf_names']))
        
        # Initialize conversation history if it doesn't exist
        if 'conversation_history' not in st.session_state:
            st.session_state['conversation_history'] = []
        
        # Display conversation history
        if st.session_state['conversation_history']:
            st.subheader("📜 Conversation History")
            
            # Create a container for scrollable history
            history_container = st.container()
            
            with history_container:
                for i, (question, answer) in enumerate(st.session_state['conversation_history']):
                    # Question
                    st.write(f"**❓ Question {i+1}:** {question}")
                    
                    # Answer
                    st.write(f"**📝 Answer:** {answer}")
                    
                    # Add separator
                    st.write("---")
        
        # Question input at the bottom
        st.subheader("🔍 Ask a New Question")
        
        # Create a form for the question input
        with st.form("question_form", clear_on_submit=True):
            user_question = st.text_input(
                "Enter your question:",
                placeholder="e.g., What is the main topic of this document?",
                key="new_question"
            )
            
            submit_button = st.form_submit_button("Submit Question", type="primary")
            
            if submit_button and user_question:
                with st.spinner("Generating answer..."):
                    # Get the vector store from session state
                    vector_store = st.session_state['vector_store']
                    
                    # Perform similarity search
                    docs = vector_store.similarity_search(user_question, k=3)
                    
                    # Get the QA chain
                    chain = get_chains()
                    
                    if chain:
                        # Generate answer
                        response = chain({"input_documents": docs, "question": user_question}, 
                                       return_only_outputs=True)
                        
                        # Add to conversation history
                        st.session_state['conversation_history'].append((user_question, response["output_text"]))
                        
                        # Rerun to update the display
                        st.rerun()
                        
                    else:
                        st.error("❌ Failed to initialize the QA chain")
        
        # Clear conversation history button
        if st.session_state['conversation_history']:
            if st.button("🗑️ Clear Conversation History", help="Clear all previous questions and answers"):
                st.session_state['conversation_history'] = []
                st.rerun()
    
    else:
        st.info("👆 Please upload PDF files using the sidebar and click 'Process PDFs' to get started!")

if __name__ == "__main__":
    main()

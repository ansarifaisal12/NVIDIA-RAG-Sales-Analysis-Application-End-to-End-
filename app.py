import os
import streamlit as st
from llama_index.core import Settings
from llama_index.llms.nvidia import NVIDIA
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, Document
import fitz  # PyMuPDF for reading PDFs
import tempfile
from dotenv import load_dotenv
import atexit
from datetime import datetime
import time
import uuid
from rag_monitor import RAGMonitor, ResponseEvaluator, RAGMetrics

# Set page config first, before any other Streamlit commands
st.set_page_config(page_title="📊 Sales Expert Analysis RAG", layout="wide")

# Load environment variables from .env file
load_dotenv()

# Set NVIDIA API key from environment variable
nvidia_api_key = os.getenv("NVIDIA_API_KEY")

# Initialize RAG monitoring and evaluation
rag_monitor = RAGMonitor()
response_evaluator = ResponseEvaluator()

# Debugging: Check if the NVIDIA API key is loaded
if nvidia_api_key is None or not nvidia_api_key.startswith("nvapi-"):
    st.error("❌ NVIDIA_API_KEY not found or invalid in .env file. Please check your .env file.")
else:
    st.success("✅ NVIDIA_API_KEY loaded successfully.")

# Set the environment variable explicitly
os.environ["NVIDIA_API_KEY"] = nvidia_api_key

# Check if NVIDIA_API_KEY is still None (failsafe)
if not os.getenv("NVIDIA_API_KEY"):
    st.error("❌ NVIDIA_API_KEY is missing. Please set it in your .env file.")
    st.stop()

# Initialize NVIDIA LLM and Embedding Models
try:
    Settings.llm = NVIDIA(model="nvidia/llama-3.1-nemotron-70b-instruct")
    Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")
    Settings.text_splitter = SentenceSplitter(chunk_size=400)
except Exception as e:
    st.error(f"❌ Error initializing NVIDIA models: {e}")
    st.stop()

# Streamlit App Title and Description
st.title("🛒 Sales Analysis RAG Expert System")
st.markdown("Analyze your sales data using NVIDIA's LLM and Embeddings.")

# Function to extract text from a PDF with proper resource management
def extract_text_from_pdf(file_path):
    """Extract text content from a PDF file."""
    doc = None
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    finally:
        if doc:
            doc.close()

# Function to load and preprocess sales data from PDFs
def process_uploaded_files(uploaded_files):
    """Process uploaded PDF files and return a list of text documents wrapped in Document objects."""
    documents = []
    temp_files = []
    
    try:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                file_path = temp_file.name
                temp_files.append(file_path)
                
                # Extract text from the PDF
                try:
                    text = extract_text_from_pdf(file_path)
                    # Wrap the text in a Document object
                    document = Document(text=text)
                    documents.append(document)  # Add Document object to the list
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    st.warning(f"Could not delete temporary file {temp_file}: {str(e)}")
    
    return documents

# Main function to build the Vector Store Index and create a query engine
def build_index(documents):
    """Build Vector Store Index from text documents."""
    try:
        # Use the extracted Document objects directly
        index = VectorStoreIndex.from_documents(documents)
        return index.as_query_engine(similarity_top_k=10)
    except Exception as e:
        st.error(f"Error building index: {e}")
        return None

# Create two columns for the layout
col1, col2 = st.columns([2, 1])

with col1:
    # File uploader for PDFs
    uploaded_files = st.file_uploader(
        "Upload your Sales PDFs",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF files containing your sales data"
    )

with col2:
    # Display upload status and file count
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} PDF file(s) uploaded successfully.")
        for file in uploaded_files:
            st.info(f"📄 {file.name}")

# Process uploaded files and create a query engine
if uploaded_files:
    with st.spinner("⏳ Processing and indexing documents..."):
        documents = process_uploaded_files(uploaded_files)
        if documents:
            query_engine = build_index(documents)
        else:
            query_engine = None
            st.error("Failed to process documents.")
    
    if query_engine:
        # Create a container for the query section
        query_container = st.container()
        with query_container:
            st.markdown("### 🔍 Ask Questions About Your Sales Data")
            
            # Input box for user queries with a placeholder
            query = st.text_input(
                "Enter your question:",
                placeholder="Example: What were the top selling products last quarter?",
                key="query_input"
            )
            
            if query:
                st.markdown("### 📊 Analysis Result")
                with st.spinner("🤔 Analyzing your question..."):
                    try:
                        # Record start time
                        start_time = time.time()
                        
                        # Get response from query engine
                        response = query_engine.query(query)
                        
                        # Calculate response time
                        response_time = time.time() - start_time
                        
                        # Evaluate response
                        eval_scores = response_evaluator.evaluate_response(query, str(response))
                        
                        # Create metrics object
                        metrics = RAGMetrics(
                            query_id=str(uuid.uuid4()),
                            timestamp=datetime.now().isoformat(),
                            query=query,
                            response=str(response),
                            response_time=response_time,
                            token_count=len(str(response).split()),
                            context_quality=eval_scores['relevance'],
                            answer_relevancy=eval_scores['overall'],
                            source_documents=[str(doc) for doc in response.source_nodes] if hasattr(response, 'source_nodes') else None
                        )
                        
                        # Log interaction
                        rag_monitor.log_interaction(metrics)
                        
                        # Display response
                        st.success(response)
                        
                        # Display evaluation metrics
                        with st.expander("📊 Response Quality Metrics"):
                            st.json(eval_scores)
                            
                            # Add user feedback
                            feedback = st.slider(
                                "How helpful was this response?",
                                min_value=1,
                                max_value=5,
                                value=3,
                                help="1 = Not helpful at all, 5 = Extremely helpful"
                            )
                            
                            if st.button("Submit Feedback"):
                                metrics.user_feedback = feedback
                                rag_monitor.log_interaction(metrics)
                                st.success("Thank you for your feedback!")
                        
                        # Create columns for download buttons
                        btn_col1, btn_col2 = st.columns([1, 4])
                        with btn_col1:
                            st.download_button(
                                label="📥 Download Analysis",
                                data=str(response),
                                file_name="sales_analysis.txt",
                                mime="text/plain"
                            )
                    except Exception as e:
                        st.error(f"Error processing query: {e}")
else:
    st.info("👋 Welcome! Please upload your PDF files containing sales data to begin analysis.")
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        1. Upload one or more PDF files containing your sales data
        2. Wait for the files to be processed and indexed
        3. Ask questions about your sales data in natural language
        4. Download the analysis results if needed
        """)

# Add monitoring dashboard
with st.expander("📈 RAG System Metrics", expanded=False):
    st.markdown("### System Performance Metrics")
    metrics_summary = rag_monitor.get_metrics_summary()
    
    if metrics_summary:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Queries", metrics_summary["total_queries"])
            st.metric("Queries (Last 24h)", metrics_summary["queries_last_24h"])
            
        with col2:
            st.metric("Avg Response Time", f"{metrics_summary['avg_response_time']:.2f}s")
            st.metric("Avg Context Quality", f"{metrics_summary['avg_context_quality']:.2%}")
            
        with col3:
            st.metric("Avg Answer Relevancy", f"{metrics_summary['avg_answer_relevancy']:.2%}")
            if metrics_summary.get("avg_user_feedback"):
                st.metric("Avg User Rating", f"{metrics_summary['avg_user_feedback']:.1f}/5")
        
        # Show detailed metrics table
        if st.checkbox("Show Detailed Metrics"):
            st.dataframe(rag_monitor.generate_report())
    else:
        st.info("No metrics available yet. Start asking questions to generate metrics!")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.markdown("**Powered by NVIDIA LLM & Embeddings**")
with col2:
    st.markdown("Built with ❤️ by Faisal")
with col3:
    st.markdown("Version 1.0.0")

# Add debug information in an expander
with st.expander("🔧 Debug Information", expanded=False):
    st.markdown("### System Status")
    st.write("- NVIDIA API Key Status:", "✅ Connected" if nvidia_api_key else "❌ Not Connected")
    st.write("- PDF Processing:", "✅ Ready" if 'fitz' in globals() else "❌ Not Ready")
    if uploaded_files:
        st.write("- Number of Files Processed:", len(uploaded_files))
        st.write("- File Names:", [file.name for file in uploaded_files])
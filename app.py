import streamlit as st
import tempfile
from rag_utils import extract_text_from_pdf, chunk_text, build_faiss, retrieve_and_answer

# Page configuration
st.set_page_config(
    page_title="PDF Q&A Assistant",
    page_icon="📚",
    layout="wide"
)

# Sidebar section
with st.sidebar:
    st.header("📂 Upload Your PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )
    st.markdown("---")
    st.info("✅ After uploading, type your question in the main area below.")


# Main interface

# 🟦 Title & Subtitle (Centered)
st.markdown(
    """
    <h1 style='text-align: center; color: #0078FF;'>
        📚 Ask Your PDFs
    </h1>
    <h4 style='text-align: center; color: #5D6D7E;'>
        Your personal document-based Q&A assistant
    </h4>
    """,
    unsafe_allow_html=True
)


st.write("Upload PDFs from the sidebar and ask any question. Answers will be generated based on your uploaded documents.")

# Process uploaded PDFs
if uploaded_files:
    all_text = ""
    for uploaded_file in uploaded_files:
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name
        
        # Extract text from each uploaded PDF
        all_text += extract_text_from_pdf(pdf_path) + "\n"

    # Create chunks and FAISS index dynamically
    chunks = chunk_text(all_text)
    index, embeddings = build_faiss(chunks)

    # Question input
    query = st.text_input("💬 Ask your question here:", placeholder="e.g., Summarize the document in 3 lines")

    # Search and answer
    if st.button("🔍 Find Answer"):
        if query.strip():
            with st.spinner("⏳ Analyzing your documents..."):
                answer = retrieve_and_answer(query, index, chunks)
            st.markdown("### 📢 Response:")
            st.info(answer)
        else:
            st.error("⚠ Please enter a valid question.")
else:
    st.warning("📌 Please upload at least one PDF from the sidebar to start.")


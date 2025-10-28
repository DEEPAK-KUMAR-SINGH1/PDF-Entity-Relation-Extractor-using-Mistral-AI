import streamlit as st
import csv
import PyPDF2
from mistralai import Mistral
import os
import io


# CONFIGURATION

# Load the Mistral API key from environment variables
API_KEY = os.getenv("MISTRAL_API_KEY")

# Select the Mistral model to use
MODEL_NAME = "mistral-small-2501"

# Define how much text to send to the model at once
CHUNK_SIZE = 20000  # characters per chunk 


# FUNCTIONS

def extract_pdf_text(file):
    """Extracts text from a PDF file."""
    text = ""
    pdf_reader = PyPDF2.PdfReader(file)  # Read the uploaded PDF
    for page in pdf_reader.pages:
        page_text = page.extract_text()  # Extract text from each page
        if page_text:
            text += page_text + "\n"
    return text.strip()   # Return all extracted text


def extract_entities_relations(text):
    """Sends PDF text (in chunks) to Mistral API for entity extraction."""
    client = Mistral(api_key=API_KEY)  # Create Mistral client

    # Split text into smaller chunks
    chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    total_chunks = len(chunks)

    all_results = []    # Store results from all chunks
    progress = st.progress(0, text="⏳ Starting entity extraction...")

    # Process each chunk one by one
    for i, chunk in enumerate(chunks, start=1):
        progress.progress(i / total_chunks, text=f"🔍 Processing chunk {i}/{total_chunks}...")

        # Instruction prompt for the AI model
        prompt = f"""
        You are an information extraction assistant.
        From the following text, extract: 
        - Entities: Organisation, Name, PAN
        - Relation: PAN_Of (linking PAN to the Name)
        Output in CSV format with columns:
        Entity(PAN),Relation,Entity(Person),Organisation
        
        If no organisation, keep blank.
        Text (Part {i}/{total_chunks}):

        {chunk}
"""

        try:
            # Send request to Mistral model
            response = client.chat.complete(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}]
            )
            result = response.choices[0].message.content.strip()
            all_results.append(result)
        except Exception as e:
            # If an error occurs, show it and skip this chunk
            st.error(f"❌ Error in chunk {i}: {e}")
            continue
    
    # Update progress when all chunks are done
    progress.progress(1.0, text="✅ Completed all chunks!")
    return "\n".join(all_results)  # Combine all results


def convert_to_csv(data_str):
    """Converts extracted text into downloadable CSV format."""
    output = io.StringIO()
    writer = csv.writer(output)
    lines = data_str.strip().split("\n")  # Split text into lines
    for line in lines:
        row = [col.strip() for col in line.split(",")]  # Split columns by comma
        writer.writerow(row)
    return output.getvalue().encode("utf-8")   # Return encoded CSV data


# STREAMLIT UI

# Set Streamlit page details
st.set_page_config(page_title="PDF Entity Extractor", page_icon="📄", layout="centered")

# Title and instructions
st.title("📄 PDF Entity & Relation Extractor using Mistral AI")
st.write("Upload a PDF, and this app will extract entities (Organisation, Name, PAN) and their relationships using Mistral LLM.")

# File uploader for PDF input
uploaded_file = st.file_uploader("📤 Upload your PDF file", type=["pdf"])

# If file is uploaded
if uploaded_file:
    if st.button("🚀 Extract Entities"):
        with st.spinner("Extracting text from PDF..."):
            pdf_text = extract_pdf_text(uploaded_file)  # Step 1: Extract text

        # Warn user if PDF is too large
        if len(pdf_text) > 1_000_000:  # optional safeguard
            st.warning("⚠️ Your PDF is very large! Only the first part will be processed to avoid token overflow.")
            pdf_text = pdf_text[:1_000_000]

        st.success("✅ PDF text extracted successfully!")
        st.text_area("📜 Extracted PDF Text (Preview)", pdf_text[:1000], height=200)

        # Step 2: Send text to Mistral for entity extraction
        with st.spinner("🔍 Sending text to Mistral API..."):
            extracted_data = extract_entities_relations(pdf_text)

        st.success("✅ Entities extracted successfully!")
        st.text_area("🧾 Extracted Entities & Relations (CSV Format)", extracted_data, height=200)

        # Step 3: Convert and download extracted data as CSV
        csv_data = convert_to_csv(extracted_data)
        st.download_button(
            label="💾 Download CSV",
            data=csv_data,
            file_name="output_entities.csv",
            mime="text/csv"
        )
else:
    # If no file uploaded
    st.info("👆 Please upload a PDF file to start.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Mistral AI")

import os
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import docx
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from groq import Groq

# Initialize the embedding model and Groq client
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = Groq(api_key="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")


# Helper functions for text extraction
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text


def extract_text_from_image(image_path):
    return pytesseract.image_to_string(Image.open(image_path))


def extract_text_from_word(word_path):
    doc = docx.Document(word_path)
    return "\n".join(para.text for para in doc.paragraphs)


def extract_text_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return json.dumps(data, indent=4)


def split_text_into_chunks(text, chunk_size=512):
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def create_embeddings(chunks):
    return embedding_model.encode(chunks)


def retrieve_relevant_chunks(query, vector_db, top_k=5):
    query_embedding = embedding_model.encode([query])
    similarities = cosine_similarity(query_embedding, vector_db["embeddings"])
    most_similar_idx = similarities[0].argsort()[-top_k:][::-1]
    return [vector_db["chunks"][i] for i in most_similar_idx]


def generate_summary(system_prompt, user_prompt, relevant_chunks):
    context = "\n".join(relevant_chunks)
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuery: {user_prompt}"},
            ],
            model="llama3-8b-8192",
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"


# Streamlit app
st.title("📄 DocScanner: Document Query Assistant")
st.caption("🚀 A document-based assistant powered by Groq and Sentence Transformers")

# Sidebar info
st.sidebar.header("📜 Instructions")
st.sidebar.write("""
1. **Upload Your Document:**  
   - Supported formats: PDF, Word, JSON, PNG, JPG, JPEG.  
2. **Wait for Processing:**  
   - The app will process the document and prepare it for querying.  
3. **Enter Your Query:**  
   - Use the input box below the chat to ask specific questions about the document.    

📬 **Need Help?**  
If you need assistance, [Contact Us](mailto:adi.desh1734@gmail.com).
""")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I assist you with your document?"}
    ]
if "vector_db" not in st.session_state:
    st.session_state["vector_db"] = None
if "processed_file" not in st.session_state:
    st.session_state["processed_file"] = None

# File upload
uploaded_file = st.file_uploader(
    "Upload a document (PDF, Word, JSON, or image)", type=["pdf", "docx", "json", "png", "jpg", "jpeg"]
)

if uploaded_file:
    # Check if a new file is uploaded
    if st.session_state["processed_file"] != uploaded_file.name:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract text based on file type
        if file_extension == ".pdf":
            text_content = extract_text_from_pdf(uploaded_file.name)
            if not text_content.strip():
                images = convert_from_path(uploaded_file.name, 300)
                text_content = "".join(extract_text_from_image(image) for image in images)
        elif file_extension in [".png", ".jpg", ".jpeg"]:
            text_content = extract_text_from_image(uploaded_file.name)
        elif file_extension == ".docx":
            text_content = extract_text_from_word(uploaded_file.name)
        elif file_extension == ".json":
            text_content = extract_text_from_json(uploaded_file.name)
        else:
            text_content = "Unsupported file type."

        os.remove(uploaded_file.name)

        # Process text and create embeddings
        chunks = split_text_into_chunks(text_content)
        embeddings = create_embeddings(chunks)

        # Update session state
        st.session_state["vector_db"] = {"chunks": chunks, "embeddings": embeddings}
        st.session_state["processed_file"] = uploaded_file.name

        # Update chat history
        st.session_state["messages"].append(
            {"role": "assistant", "content": "Document processed. Please enter your query."}
        )

# Display chat history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle query input
if user_query := st.chat_input("Enter your query..."):
    if st.session_state["vector_db"] is None:
        st.error("Please upload and process a document first.")
    else:
        # Add user query to chat
        st.session_state["messages"].append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        # Use preprocessed embeddings and chunks
        relevant_chunks = retrieve_relevant_chunks(user_query, st.session_state["vector_db"])
        system_prompt = "You are an assistant specialized in summarizing and querying documents."
        response = generate_summary(system_prompt, user_query, relevant_chunks)

        # Add assistant response to chat
        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

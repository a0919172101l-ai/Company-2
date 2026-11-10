import streamlit as st
from pathlib import Path
import fitz  # PyMuPDF
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_TITLE = "Lyla | Company Investor AI"
KNOWLEDGE_FOLDER = Path(__file__).parent / "knowledge_base"
SUPPORTED_EXTENSIONS = [".pdf"]

st.set_page_config(page_title=APP_TITLE, page_icon="💡", layout="wide")

st.title("💡 Lyla — Company Investor AI")
st.caption("Making technology visible | Ask questions about our company, services, and business model.")

st.info(
    "Lyla searches the approved company documents provided in its knowledge base. "
    "It does not use an external AI API. Answers are limited to information found in the documents."
)

def split_into_chunks(text, max_chars=1100, overlap=180):
    """Split page text into readable overlapping chunks."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + max_chars // 2:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks

def extract_pdf(file_bytes, filename):
    """Extract text page-by-page so every result can show its PDF page."""
    records = []
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
        for page_index, page in enumerate(document):
            page_text = page.get_text("text")
            for chunk in split_into_chunks(page_text):
                records.append({
                    "text": chunk,
                    "source": filename,
                    "page": page_index + 1
                })
        document.close()
    except Exception as exc:
        st.warning(f"Could not read {filename}: {exc}")
    return records

@st.cache_data(show_spinner=False)
def load_knowledge_base(folder_string, file_signature):
    """Read all PDFs in knowledge_base. Cache refreshes when file names/sizes change."""
    folder = Path(folder_string)
    records = []
    if not folder.exists():
        return records
    for pdf_path in sorted(folder.rglob("*.pdf")):
        try:
            records.extend(extract_pdf(pdf_path.read_bytes(), pdf_path.name))
        except Exception as exc:
            st.warning(f"Could not open {pdf_path.name}: {exc}")
    return records

def get_folder_signature(folder):
    if not folder.exists():
        return ()
    return tuple(
        (str(p.relative_to(folder)), p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(folder.rglob("*.pdf"))
    )

def retrieve_relevant_chunks(question, records, top_k=4):
    """Use local TF-IDF similarity to find document passages related to the question."""
    if not records:
        return []
    texts = [item["text"] for item in records]
    # Character n-grams work better than English-only word tokenization
    # when the knowledge base contains both Chinese and English PDFs.
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        lowercase=True,
        sublinear_tf=True,
        max_features=250000
    )
    try:
        document_matrix = vectorizer.fit_transform(texts)
        question_vector = vectorizer.transform([question])
        scores = cosine_similarity(question_vector, document_matrix).flatten()
    except ValueError:
        return []
    ranked_indices = scores.argsort()[::-1][:top_k]
    results = []
    for idx in ranked_indices:
        score = float(scores[idx])
        if score > 0:
            result = dict(records[idx])
            result["score"] = score
            results.append(result)
    return results

# PDFs committed to GitHub in this folder are loaded automatically.
signature = get_folder_signature(KNOWLEDGE_FOLDER)
records = load_knowledge_base(str(KNOWLEDGE_FOLDER), signature)

# Optional session-only uploads for testing. These are not permanently saved to GitHub.
with st.sidebar:
    st.header("Knowledge Base")
    st.write(f"PDF passages loaded: **{len(records)}**")
    st.caption("Permanent PDFs belong in the `knowledge_base` folder in your GitHub repository.")
    uploaded_pdfs = st.file_uploader(
        "Or temporarily test with PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Uploaded files are available during this app session only. Add important PDFs to GitHub's knowledge_base folder for permanent use."
    )

    if uploaded_pdfs:
        st.caption("Temporary uploaded files:")
        for uploaded in uploaded_pdfs:
            records.extend(extract_pdf(uploaded.getvalue(), uploaded.name))

    st.divider()
    st.subheader("Example investor questions")
    examples = [
        "What does Lyla's company do?",
        "What services does the company provide?",
        "Who are the target customers?",
        "How does the company make money?",
        "What problems does the company solve?",
        "What is the investment plan?"
    ]
    for example in examples:
        st.write("• " + example)

if not records:
    st.warning(
        "No PDF information was found yet. Add your company PDFs to the "
        "`knowledge_base` folder in GitHub, or upload PDFs temporarily in the sidebar."
    )

st.subheader("Ask Lyla")
question = st.text_input(
    "Enter an investor question",
    placeholder="For example: What makes your services valuable to customers?"
)
top_k = st.slider("Number of source passages to retrieve", min_value=1, max_value=6, value=4)

if st.button("Search company information", type="primary", disabled=not question.strip()):
    if not records:
        st.error("There are no company documents to search. Please add PDF files first.")
    else:
        with st.spinner("Searching the company documents..."):
            matches = retrieve_relevant_chunks(question, records, top_k=top_k)

        if not matches or matches[0]["score"] < 0.035:
            st.warning(
                "I could not find reliable information about this question in the current company documents. "
                "This information is not currently available in the company data."
            )
        else:
            st.markdown("### Lyla's response")
            st.write(
                "I searched the approved company documents and found the following relevant information. "
                "The answer below is based on these sources:"
            )

            for number, item in enumerate(matches, start=1):
                with st.container(border=True):
                    st.markdown(f"**Relevant information {number}**")
                    st.write(item["text"])
                    st.caption(
                        f"Source: {item['source']} | Page: {item['page']} | "
                        f"Text-match score: {item['score']:.3f}"
                    )

            st.caption(
                "Reliability note: These are retrieved passages, not independently verified claims. "
                "Please check the cited PDF pages before using information in an investor presentation."
            )

st.divider()
st.caption("Lyla Company AI • Local document search • No external AI API")

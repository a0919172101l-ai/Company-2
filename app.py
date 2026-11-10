import streamlit as st
from pathlib import Path
import re
import fitz  # PyMuPDF

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. APP SETTINGS
# ==========================================

st.set_page_config(
    page_title="Lyla | Company Investor AI",
    page_icon="💡",
    layout="wide"
)

st.title("💡 Lyla")
st.header("Company Investor AI")

st.caption(
    "Making technology visible. "
    "Ask questions about our company, services, "
    "customers, and business model."
)

st.info(
    "Lyla searches approved company documents "
    "and provides evidence from the knowledge base. "
    "It does not use an external generative AI API."
)

ROOT = Path(__file__).parent
KNOWLEDGE_FOLDER = ROOT / "knowledge_base"


# ==========================================
# 2. FIND ALL PDF FILES
# ==========================================

def find_pdf_files():

    pdf_files = []

    # Read PDFs directly uploaded to the repository root
    pdf_files.extend(ROOT.glob("*.pdf"))

    # Read PDFs inside knowledge_base
    if KNOWLEDGE_FOLDER.exists():
        pdf_files.extend(KNOWLEDGE_FOLDER.rglob("*.pdf"))

    # Remove duplicates and sort files
    pdf_files = sorted(set(pdf_files))

    return pdf_files


# ==========================================
# 3. EXTRACT TEXT FROM PDF
# ==========================================

def split_into_chunks(text, max_chars=1000, overlap=150):

    text = re.sub(r"\s+", " ", text).strip()

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


def extract_pdf(pdf_path):

    records = []

    try:

        document = fitz.open(pdf_path)

        for page_number, page in enumerate(document, start=1):

            page_text = page.get_text("text")

            chunks = split_into_chunks(page_text)

            for chunk in chunks:

                records.append({
                    "text": chunk,
                    "source": pdf_path.name,
                    "page": page_number
                })

        document.close()

    except Exception as error:

        st.warning(
            f"Could not read {pdf_path.name}: {error}"
        )

    return records


# ==========================================
# 4. LOAD COMPANY KNOWLEDGE BASE
# ==========================================

@st.cache_data(show_spinner=False)
def load_all_pdfs(file_signature):

    records = []

    pdf_paths = find_pdf_files()

    for pdf_path in pdf_paths:

        records.extend(extract_pdf(pdf_path))

    return records


def get_file_signature():

    pdf_paths = find_pdf_files()

    return tuple(
        (
            str(path),
            path.stat().st_size,
            path.stat().st_mtime_ns
        )
        for path in pdf_paths
    )


# ==========================================
# 5. SEARCH ENGINE
# ==========================================

def search_company_data(question, records, top_k=4):

    if not records:
        return []

    documents = [
        record["text"]
        for record in records
    ]

    # Character-based TF-IDF supports Chinese and English text.
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        lowercase=True,
        sublinear_tf=True,
        max_features=250000
    )

    try:

        document_matrix = vectorizer.fit_transform(documents)

        question_vector = vectorizer.transform([question])

        scores = cosine_similarity(
            question_vector,
            document_matrix
        ).flatten()

    except ValueError:

        return []

    ranked_indices = scores.argsort()[::-1][:top_k]

    results = []

    for index in ranked_indices:

        score = float(scores[index])

        if score > 0:

            result = records[index].copy()

            result["score"] = score

            results.append(result)

    return results


# ==========================================
# 6. LOAD DATA
# ==========================================

signature = get_file_signature()

with st.spinner("Loading company documents..."):

    records = load_all_pdfs(signature)

pdf_files = find_pdf_files()


# ==========================================
# 7. SIDEBAR
# ==========================================

with st.sidebar:

    st.header("About Lyla")

    st.write(
        "Lyla helps investors understand "
        "our company by searching approved "
        "company information."
    )

    st.divider()

    st.metric(
        "PDF Documents",
        len(pdf_files)
    )

    st.metric(
        "Knowledge Records",
        len(records)
    )

    st.write("**Search Method:**")
    st.write("TF-IDF + Cosine Similarity")

    st.write("**External AI API:**")
    st.write("Not used")

    with st.expander("View loaded PDF files"):

        for pdf in pdf_files:
            st.write("📄 " + pdf.name)

    st.divider()

    st.subheader("Example Questions")

    examples = [
        "What does our company do?",
        "What services do we provide?",
        "Who are our target customers?",
        "How does our company make money?",
        "What problems do we solve?",
        "What is our investment plan?",
        "Who are our competitors?"
    ]

    for example in examples:
        st.write("• " + example)


# ==========================================
# 8. CHECK KNOWLEDGE BASE
# ==========================================

if not pdf_files:

    st.error(
        "No PDF files found. "
        "Please upload company PDFs to "
        "the repository root or knowledge_base folder."
    )

elif not records:

    st.warning(
        "PDF files were found, but no readable text "
        "was extracted. Some PDFs may contain scanned images."
    )


# ==========================================
# 9. INVESTOR QUESTION
# ==========================================

st.subheader("Ask Lyla")

question = st.text_input(
    "Investor Question",
    placeholder="What services does our company provide?"
)

top_k = st.slider(
    "Number of relevant passages",
    min_value=1,
    max_value=6,
    value=4
)

if st.button("Ask Lyla", type="primary"):

    if not question.strip():

        st.warning("Please enter a question.")

    elif not records:

        st.error(
            "The company knowledge base is empty."
        )

    else:

        with st.spinner(
            "Searching company information..."
        ):

            results = search_company_data(
                question,
                records,
                top_k
            )

        # ==================================
        # 10. DISPLAY ANSWER
        # ==================================

        if not results or results[0]["score"] < 0.015:

            st.warning(
                "I could not find sufficient information "
                "in our company knowledge base. "
                "This information is not currently "
                "available in the company data."
            )

        else:

            st.subheader("Lyla's Response")

            st.write(
                "I found the following information "
                "in the company's approved documents:"
            )

            for number, result in enumerate(
                results, start=1
            ):

                with st.container(border=True):

                    st.markdown(
                        f"### Relevant Information {number}"
                    )

                    st.write(result["text"])

                    st.caption(
                        f"Source: {result['source']} | "
                        f"Page: {result['page']} | "
                        f"Match score: {result['score']:.3f}"
                    )

            st.divider()

            st.caption(
                "Lyla retrieves information from company documents. "
                "Please verify the original PDF before making "
                "business or investment decisions."
            )


# ==========================================
# 11. FOOTER
# ==========================================

st.divider()

st.caption(
    "Lyla Company Investor AI | "
    "Local Document Search | No External AI API"
)

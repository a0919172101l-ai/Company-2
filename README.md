# Lyla — Company Investor AI

A no-external-API Streamlit app that searches approved company PDFs and returns relevant passages with source filenames and page numbers.

## 1. Add your PDFs

Put all approved company PDF files inside the `knowledge_base/` folder.

Example:

```text
lyla-company-investor-ai/
├── app.py
├── requirements.txt
├── README.md
└── knowledge_base/
    ├── company_information.pdf
    ├── business_model.pdf
    ├── services.pdf
    └── investment_plan.pdf
```

You can add as many PDFs as you need. You do not need to edit `app.py` when adding another PDF.

**Important:** For permanent use on Streamlit Community Cloud, upload/commit the PDFs into the GitHub repository's `knowledge_base` folder. The sidebar uploader is only for temporary testing during the current app session.

## 2. Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 3. Deploy with GitHub + Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, and the `knowledge_base` folder.
3. Upload your approved PDF files into `knowledge_base`.
4. In Streamlit Community Cloud, choose the repository.
5. Set the main file path to `app.py`.
6. Deploy.

## 4. How it works

- PyMuPDF extracts text from each PDF page.
- The app divides text into overlapping passages.
- TF-IDF and cosine similarity rank passages related to the investor's question.
- The app displays the most relevant passages, PDF filename, and page number.
- If no relevant passage is found, Lyla says the information is not currently available.

This is a document retrieval system, not a generative language model. It does not invent a fluent answer or verify whether a claim in a PDF is true. Make sure your PDFs contain accurate, approved company information.

## 5. Suggested PDF categories

- Company overview and mission
- Products and services
- Customer groups and customer problems
- Business model and pricing assumptions
- Competitors and market research
- Investment plan and use of funds
- Training courses and facilities
- Company risks and solutions
- Revenue model and proposed payment methods

Do not include private personal information, passwords, or confidential information that should not be shown to exhibition visitors.

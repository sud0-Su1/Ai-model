# Ai-model 📚

An intelligent document question-answering system that works entirely on your **local system** — no external AI model or API required. Upload up to **10 PDFs** using a sleek **Trikeler UI**, ask questions based on the documents, and get relevant answers with key point summaries.

---

## ⚙️ Features

- 🧠 **Local-only processing** — No OpenAI, no internet dependency
- 📄 **Supports up to 10 PDFs** at once
- ❓ **Ask questions** based on uploaded PDFs
- 📌 **Extracts key points** and summarizes core ideas
- 🧑‍💻 **Custom Trikeler UI** for smooth interaction

---

## 🚀 Getting Started

### Requirements

- Python 3.8+
- pip
- No GPU or external ML models needed

### Installation

```bash
git clone https://github.com/sud0-Su1/Ai-model.git
cd Ai-model
python -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## 🧩 How It Works

1. Launch the application
2. Use **Trikeler UI** to upload up to **10 PDF files**
3. The backend extracts and processes text from PDFs
4. Enter your question in the UI
5. System matches and returns:
   - ✅ Answer based on PDF content
   - 📌 Key points summarized from relevant sections

---

## 💡 Example Usage

- **Upload PDFs**: `report1.pdf`, `summary2.pdf`, ...
- **Ask**: "What are the advantages of using method X?"
- **Output**:
  - **Answer**: "Method X is beneficial because..."
  - **Keypoints**:
    - Easy to implement
    - Reduces cost
    - Scales well

---

## 🔧 Configuration

No model or API configuration needed — everything runs locally.

Optional tweaks in `config.py`:
- `MAX_PDF_UPLOAD = 10`
- `CHUNK_SIZE = 500`
- `MAX_KEYPOINTS = 5`

---

## 🧪 Tech Stack

- Python (text processing, backend logic)
- PDFMiner / PyMuPDF for PDF text extraction
- Basic NLP logic (e.g., keyword matching, ranking)
- Trikeler UI (custom HTML/CSS interface)

---

## 🛠️ Contributing

Want to improve the summarization, add filtering, or support more file types?

1. Fork the repository
2. Create a new branch (`feature/improve-summary`)
3. Commit your changes
4. Open a Pull Request 🚀

---

# Pagewise

Upload a PDF and ask it questions in plain language. Pagewise answers using only your document and shows the pages each answer came from.

**Live demo:** (https://pagewise-ask-questions.streamlit.app/)

## Features

- **Ask your own PDF.** Upload a document and start asking. There are no settings to configure.
- **Grounded answers.** Replies come only from the PDF. If the answer isn't in the document, Pagewise says so instead of guessing.
- **Page references.** Every answer lists the pages it was found on.
- **Conversation memory.** Follow-up questions such as "explain that more simply" work, and you can ask what you asked earlier.
- **Question history.** A sidebar lists every question you've asked, newest first.
- **Math support.** Formulas in the document are shown as properly rendered math.
- **Friendly small talk.** Greetings and questions about the assistant itself (who it is, which model it uses, how it works) get a natural reply.

## How it works

Pagewise uses retrieval-augmented generation (RAG):

1. **Load.** The uploaded PDF is read page by page.
2. **Split.** The text is cut into overlapping chunks.
3. **Embed.** Each chunk is converted into a vector with a sentence-transformer model and stored in an in-memory Chroma database.
4. **Retrieve.** When you ask a question, the most relevant and diverse chunks are found using MMR search.
5. **Answer.** The language model writes a reply from those chunks, using the recent chat for follow-ups.

## Tech stack

| Part | Tool |
|---|---|
| Interface | Streamlit with custom HTML and CSS |
| Language model | `openai/gpt-oss-120b` through Groq |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database | Chroma (in memory) |
| Framework | LangChain |
| PDF reading | pypdf |

## Project structure

```
.
├── app.py                  Streamlit interface and question answering
├── create_database.py      Loads the PDF, splits it and builds the search index
├── requirements.txt        Python dependencies
├── .streamlit/
│   └── config.toml         Theme and upload settings
└── .gitignore
```

## Run it locally

1. Clone the repository and open the folder.
2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```
   On macOS or Linux, use `source .venv/bin/activate` instead.
3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a file named `.env` next to `app.py` with your Groq API key:
   ```
   GROQ_API_KEY=your-key-here
   ```
   You can get a key from the Groq console.
5. Start the app:
   ```
   streamlit run app.py
   ```
6. Open http://localhost:8501, upload a PDF and ask a question.

The first run downloads the embedding model, so the first upload takes a little longer.


## Limitations

- Scanned PDFs without selectable text are not supported, because there is no OCR.
- One PDF at a time. Uploading a new one replaces the current document and clears the chat.
- Uploaded documents and chat history are kept in memory only and disappear when the app restarts.
- Tables and complex layouts may not be extracted perfectly from some PDFs.
- Answers depend on the language model and can still contain mistakes, so check important details against the source pages.

## License

Add a license of your choice, for example MIT.

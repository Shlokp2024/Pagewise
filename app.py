import html
import os
import re
import traceback
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import streamlit as st

APP_NAME = "Pagewise"

st.set_page_config(page_title=APP_NAME, page_icon="📄", layout="wide")

from dotenv import load_dotenv

load_dotenv()

try:
    os.environ.setdefault("GROQ_API_KEY", st.secrets["GROQ_API_KEY"])
except Exception:
    pass

from groq import BadRequestError
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from create_database import create_database, create_embeddings

NOT_FOUND = "Unable to find answer"
ABOUT_TAG = "[ABOUT]"
MEMORY_TURNS = 6
MODEL_ID = "groq:openai/gpt-oss-120b"
ASSISTANT_FACTS = (
    f"This assistant is called {APP_NAME}. "
    "It is an AI assistant that answers questions about a PDF that the user uploads. "
    "The user uploads a PDF, asks questions in plain language, and gets answers with the page numbers they came from. "
    "It also remembers the last few questions of the current chat, so follow-up questions work. "
    "If the answer is not in the PDF it says so instead of guessing. "
    "The language model is gpt-oss-120b (openai/gpt-oss-120b), an open-weight model from OpenAI, served through Groq. "
    "Relevant passages are found with the embedding model all-MiniLM-L6-v2 and stored in a Chroma vector database in memory. "
    "This approach is called retrieval-augmented generation (RAG): the app first retrieves the most relevant passages "
    "from the PDF, then the language model writes the answer from them."
)

answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a friendly, helpful AI assistant for answering questions about an uploaded PDF. "
        "There are two kinds of messages.\n"
        "1. Questions about the document. Only use the provided context to answer. "
        f"If the answer is not in the context, reply with exactly '{NOT_FOUND}'.\n"
        "2. Conversation about the assistant itself: greetings, thanks, goodbyes, small talk, "
        "'who are you', 'what can you do', 'how do you work', 'which model or LLM are you', "
        "and short questions about the AI technology behind this app (LLMs, embeddings, RAG). "
        "Answer these naturally and briefly using the facts below, without using the context, "
        f"and begin that reply with {ABOUT_TAG}. Never reply '{NOT_FOUND}' to this kind of message.\n"
        "Questions about the conversation itself, such as 'what did I ask before' or 'what was my last question', "
        "also belong here: answer them from the list of earlier questions given in the user message. "
        "The last question asked means the final item in that list, never the current question. "
        "If the list says None, say there is no earlier question. Begin that reply with "
        f"{ABOUT_TAG} too.\n"
        "Use the earlier conversation only to understand follow-up questions. "
        "Write math formulas in LaTeX, using double dollar signs for a formula on its own line and single dollar signs inside a sentence. "
        "Never use backslash brackets or plain square brackets around a formula. "
        "Write in plain text. You may use short lists and **bold**, but do not use tables or headings.\n"
        f"Facts: {ASSISTANT_FACTS}",
    ),
    MessagesPlaceholder("history"),
    (
        "human",
        "Earlier questions in this chat, oldest first, not including the current one:\n{recent}\n\n"
        "Context:\n{context}\n\nCurrent question: {question}",
    ),
])

rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Rewrite the final question so it can be understood without the conversation. "
        "Return only the rewritten question. If it already stands alone, or it is a greeting, thanks or small talk, return it unchanged.",
    ),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&display=swap');

html, body, [class*="css"] { font-family: 'Instrument Sans', system-ui, sans-serif; }

footer, .stDeployButton, #MainMenu { visibility: hidden; }

.block-container { max-width: 820px; padding-top: 2.4rem; padding-bottom: 6rem; }

[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] li,
section.main [data-testid="stMarkdownContainer"] p,
section.main [data-testid="stMarkdownContainer"] li {
  font-family: 'Newsreader', Georgia, serif;
  font-size: 1.14rem;
  line-height: 1.7;
}

.brand {
  font-family: 'Newsreader', Georgia, serif;
  font-size: 1.7rem;
  font-weight: 500;
  letter-spacing: -0.01em;
  margin-bottom: 1rem;
}

.hero { text-align: center; padding: 3vh 0 1.2rem; }
.hero h1 {
  font-family: 'Newsreader', Georgia, serif;
  font-weight: 500;
  font-size: clamp(2rem, 4.5vw, 2.9rem);
  line-height: 1.12;
  letter-spacing: -0.02em;
  margin: 1rem 0 0.7rem;
  padding: 0;
}
.hero p { color: #5d6a84; font-size: 1.05rem; margin: 0 auto; max-width: 44ch; }

.q {
  font-family: 'Newsreader', Georgia, serif;
  font-size: 1.5rem;
  line-height: 1.5;
  font-weight: 500;
  letter-spacing: -0.01em;
  margin: 1.4rem 0 0.6rem;
  padding: 0;
  overflow-wrap: anywhere;
}
.q span {
  background: linear-gradient(transparent 52%, #ffe45c 52%, #ffe45c 92%, transparent 92%);
  padding: 0 0.12em;
  -webkit-box-decoration-break: clone;
  box-decoration-break: clone;
}

.unfound {
  font-family: 'Newsreader', Georgia, serif;
  font-style: italic;
  font-size: 1.14rem;
  color: #5d6a84;
}

.chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 0.5rem 0 0; }
.chip {
  font-size: 0.8rem;
  font-weight: 500;
  color: #2f4ea2;
  border: 1px solid #b8c5e6;
  background: #f6f8fd;
  padding: 3px 10px;
  border-radius: 999px;
}

.sep { border: 0; border-top: 1px solid #d9e0ec; margin: 1.6rem 0 0; }

.blank h2 {
  font-family: 'Newsreader', Georgia, serif;
  font-weight: 500;
  font-size: 2rem;
  letter-spacing: -0.015em;
  margin: 4vh 0 0.4rem;
  padding: 0;
}
.blank p { color: #5d6a84; margin: 0; }

.doc-card {
  background: #ffffff;
  border: 1px solid #d9e0ec;
  border-radius: 12px;
  padding: 12px 14px;
  margin-bottom: 0.8rem;
}
.doc-name { font-weight: 600; overflow-wrap: anywhere; }
.doc-meta { font-size: 0.82rem; color: #5d6a84; margin-top: 2px; }

.side-title { font-weight: 600; margin: 1.2rem 0 0.5rem; }
.hist-item {
  display: flex;
  gap: 10px;
  align-items: flex-start;
  padding: 7px 4px;
  font-size: 0.92rem;
  line-height: 1.4;
}
.hist-dot {
  flex: 0 0 9px;
  height: 9px;
  border-radius: 50%;
  border: 1.5px solid #5d6a84;
  margin-top: 6px;
}
.hist-item.latest .hist-dot { background: #ffe45c; border-color: #b89b00; }
.hist-item.latest { font-weight: 500; }
.hist-tag { display: block; font-size: 0.76rem; color: #8a7400; font-weight: 600; margin-top: 2px; }
.hist-empty { color: #5d6a84; font-size: 0.9rem; }
</style>
"""

PAGE_ART = """
<svg viewBox="0 0 160 200" width="120" height="150" aria-hidden="true">
  <rect x="6" y="6" width="148" height="188" rx="8" fill="#fff" stroke="#16213A" stroke-width="2.5"/>
  <rect x="26" y="34" width="78" height="14" rx="2" fill="#16213A" opacity=".85"/>
  <rect x="26" y="62" width="108" height="9" rx="2" fill="#C9D2E3"/>
  <rect x="24" y="82" width="96" height="16" rx="2" fill="#FFE45C"/>
  <rect x="26" y="86" width="92" height="8" rx="2" fill="#16213A" opacity=".85"/>
  <rect x="26" y="108" width="108" height="9" rx="2" fill="#C9D2E3"/>
  <rect x="26" y="126" width="64" height="9" rx="2" fill="#C9D2E3"/>
  <rect x="24" y="146" width="84" height="16" rx="2" fill="#FFE45C"/>
  <rect x="26" y="150" width="80" height="8" rx="2" fill="#16213A" opacity=".85"/>
</svg>
"""

st.markdown(STYLE, unsafe_allow_html=True)

DEFAULTS = {"store": None, "filename": None, "signature": None, "pages": 0, "chunks": 0, "history": []}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


@st.cache_resource(show_spinner="Loading the search model. This takes a moment the first time.")
def get_embeddings():
    return create_embeddings()


@st.cache_resource(show_spinner=False)
def get_llm():
    return init_chat_model(MODEL_ID, temperature=0)


def safe_invoke(prompt_value, attempts=3):
    last_error = None
    for _ in range(attempts):
        try:
            response = get_llm().invoke(prompt_value)
        except BadRequestError as error:
            last_error = error
            continue
        if response.content and response.content.strip():
            return response
    if last_error is not None:
        raise last_error
    raise ValueError("The model returned an empty reply.")


def prepare_markdown(text):
    text = re.sub(r"\\\[(.+?)\\\]", lambda m: "\n\n$$" + m.group(1).strip() + "$$\n\n", text, flags=re.S)
    text = re.sub(r"\\\((.+?)\\\)", lambda m: "$" + m.group(1).strip() + "$", text, flags=re.S)
    text = re.sub(r"(?<!\$)\$(?=\d)", r"\\$", text)
    return text


def history_messages():
    messages = []
    for item in st.session_state.history[-MEMORY_TURNS:]:
        messages.append(HumanMessage(content=item["question"]))
        messages.append(AIMessage(content=item["answer"]))
    return messages


def answer_question(question):
    messages = history_messages()

    standalone = question
    if messages:
        try:
            rewritten = safe_invoke(rewrite_prompt.invoke({"history": messages, "question": question}))
            standalone = rewritten.content.strip()
        except (BadRequestError, ValueError):
            standalone = question

    retriever = st.session_state.store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )
    docs = retriever.invoke(standalone)
    context = "\n\n".join(doc.page_content for doc in docs)

    earlier = [item["question"] for item in st.session_state.history[-MEMORY_TURNS:]]
    recent = "\n".join(f"{number}. {text}" for number, text in enumerate(earlier, 1)) or "None"

    response = safe_invoke(
        answer_prompt.invoke(
            {"history": messages, "recent": recent, "context": context, "question": question}
        )
    )
    answer = response.content.strip()

    if answer.startswith(ABOUT_TAG):
        answer = answer[len(ABOUT_TAG):].strip()
        sources = []
    elif re.sub(r"[^a-z ]", "", answer.lower()).strip() == NOT_FOUND.lower():
        sources = []
    else:
        sources = sorted({int(doc.metadata.get("page", 0)) + 1 for doc in docs})

    return {"question": question, "answer": answer, "sources": sources}


def load_document(uploaded):
    signature = f"{uploaded.name}:{uploaded.size}"
    if signature == st.session_state.signature:
        return

    try:
        embeddings = get_embeddings()
        with st.spinner(f"Reading {uploaded.name}"):
            store, pages, chunks = create_database(uploaded.getvalue(), embeddings)
    except ValueError as error:
        st.error(str(error))
        return
    except Exception:
        traceback.print_exc()
        st.error("This PDF could not be processed. Try another file.")
        return

    previous = st.session_state.store
    if previous is not None:
        try:
            previous.delete_collection()
        except Exception:
            traceback.print_exc()

    st.session_state.update(
        store=store,
        filename=uploaded.name,
        signature=signature,
        pages=pages,
        chunks=chunks,
        history=[],
    )
    st.rerun()


def render_exchange(item):
    st.markdown(f'<h3 class="q"><span>{html.escape(item["question"])}</span></h3>', unsafe_allow_html=True)

    unfound = not item["sources"] and NOT_FOUND.lower() in item["answer"].lower()
    if unfound:
        st.markdown(f'<p class="unfound">{html.escape(item["answer"])}</p>', unsafe_allow_html=True)
    else:
        st.markdown(prepare_markdown(item["answer"]))

    if item["sources"]:
        chips = "".join(f'<span class="chip">Page {page}</span>' for page in item["sources"])
        st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="sep">', unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown(f'<div class="brand">{html.escape(APP_NAME)}</div>', unsafe_allow_html=True)

        if st.session_state.store is not None:
            page_word = "page" if st.session_state.pages == 1 else "pages"
            st.markdown(
                f'<div class="doc-card"><div class="doc-name">{html.escape(st.session_state.filename)}</div>'
                f'<div class="doc-meta">{st.session_state.pages} {page_word} ready to search</div></div>',
                unsafe_allow_html=True,
            )

            replacement = st.file_uploader("Replace PDF", type=["pdf"], key="replace_upload")
            if replacement is not None:
                load_document(replacement)

            if st.button("Clear chat", use_container_width=True, disabled=not st.session_state.history):
                st.session_state.history = []
                st.rerun()

        st.markdown('<div class="side-title">Your questions</div>', unsafe_allow_html=True)

        history = st.session_state.history
        if not history:
            st.markdown(
                '<div class="hist-empty">Questions you ask will be listed here, newest first.</div>',
                unsafe_allow_html=True,
            )
        for position in range(len(history) - 1, -1, -1):
            latest = position == len(history) - 1
            tag = '<span class="hist-tag">Asked last</span>' if latest else ""
            st.markdown(
                f'<div class="hist-item{" latest" if latest else ""}"><span class="hist-dot"></span>'
                f'<span>{html.escape(history[position]["question"])}{tag}</span></div>',
                unsafe_allow_html=True,
            )


def render_empty_state():
    st.markdown(
        f'<div class="hero">{PAGE_ART}<h1>Upload a PDF and ask it questions</h1>'
        "<p>Answers come only from your document, with the pages they were found on.</p></div>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], key="main_upload", label_visibility="collapsed")
    if uploaded is not None:
        load_document(uploaded)


def render_chat():
    history = st.session_state.history

    if not history:
        st.markdown(
            f'<div class="blank"><h2>Ask anything about {html.escape(st.session_state.filename)}</h2>'
            "<p>Follow-up questions work too. The assistant remembers what you asked.</p></div>",
            unsafe_allow_html=True,
        )

    for item in history:
        render_exchange(item)

    question = st.chat_input("Ask a question about your PDF", max_chars=2000)
    if not question:
        return

    question = question.strip()
    st.markdown(f'<h3 class="q"><span>{html.escape(question)}</span></h3>', unsafe_allow_html=True)

    try:
        with st.spinner("Searching your document"):
            item = answer_question(question)
    except Exception:
        traceback.print_exc()
        st.error("The AI service could not answer right now. Check your GROQ_API_KEY and try again.")
        return

    st.session_state.history.append(item)
    st.rerun()


render_sidebar()

if st.session_state.store is None:
    render_empty_state()
else:
    render_chat()
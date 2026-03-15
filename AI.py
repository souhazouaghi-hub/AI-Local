import streamlit as st
import re
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from io import BytesIO
from urllib.parse import quote
from sklearn.feature_extraction.text import TfidfVectorizer

# ----------------------------------
# CONFIG
# ----------------------------------
MAX_NOTE_CHARS = 1500
st.set_page_config(page_title="🔒 Local Notes Assistant", layout="wide")

# ----------------------------------
# FILE PARSERS
# ----------------------------------
@st.cache_data
def parse_html(file_bytes):
    soup = BeautifulSoup(file_bytes, "html.parser")
    return soup.get_text(separator="\n")

@st.cache_data
def parse_pdf(file_bytes):
    reader = PdfReader(BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

@st.cache_data
def parse_txt(file_bytes):
    return file_bytes.decode("utf-8")

# ----------------------------------
# NOTE EXTRACTION
# ----------------------------------
NAME_REGEX = re.compile(r"^[A-ZÉÈÀ][a-zéèà\-]+ [A-ZÉÈÀ][a-zéèà\-]+")  # détecte "Prénom Nom"

@st.cache_data
def extract_notes(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    notes = []

    current_header = None
    current_body = []

    FORBIDDEN_STARTS = (
        "best regards",
        "kind regards",
        "regards",
        "next steps",
        "action",
        "actions",
        "meeting",
        "meetings",
        "discussion",
        "opp",
    )

    def is_contact_line(line):
        if len(line) > 120:
            return False
        if line.lower().startswith(FORBIDDEN_STARTS):
            return False
        return bool(NAME_REGEX.match(line))

    for line in lines:
        if is_contact_line(line):
            if current_header:
                notes.append({
                    "header": current_header,
                    "body": "\n".join(current_body)
                })
            current_header = line
            current_body = []
        else:
            current_body.append(line)

    if current_header:
        notes.append({
            "header": current_header,
            "body": "\n".join(current_body)
        })

    return notes

# ----------------------------------
# NOTE HELPERS
# ----------------------------------
def format_note_for_select(note):
    return note["header"]

def extract_contact(note):
    header = note["header"]
    if " - " in header:
        name, company = header.split(" - ", 1)
    else:
        name, company = header, ""
    return name.strip(), company.strip()

# ----------------------------------
# SAFE SENTENCE TOKENIZER (no NLTK)
# ----------------------------------
def safe_sent_tokenize(text):
    """Découpe le texte en phrases sans utiliser NLTK."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

# ----------------------------------
# CONTEXT EXTRACTION
# ----------------------------------
def extract_context_sentence(note_body):
    """
    Returns a context sentence for LinkedIn messages:
    - If the note contains a National Instruments product, mention it.
    - Otherwise, return a generic phrase.
    """
    # Expanded National Instruments product list
    ni_products = [
        'pxi', 'cdaq', 'crio', 'myrio', 'compactrio', 'fielddaq', 'usb daq',
        'scxi', 'scc', 'sb', 'pci', 'gpib',
        'lv', 'labview', 'labwindows', 'multisim', 'veristand',
        'signalexpress', 'diadem', 'teststand', 'flexlogger'
    ]

    note_lower = note_body.lower()
    found = [p for p in ni_products if p.lower() in note_lower]

    if found:
        # Take the first product found and capitalize for clarity
        return f"I remember our last interactions about {found[0].upper()}"
    else:
        return "I remember our last interactions"

# ----------------------------------
# GENERATION
# ----------------------------------
def generate_for_note(note, full_name, sales_manager, region):
    name, company = extract_contact(note)
    first_name = name.split()[0]

    note_body = note["body"].replace("\n", " ").strip()
    if len(note_body) > MAX_NOTE_CHARS:
        note_body = note_body[:MAX_NOTE_CHARS] + "..."

    context_sentence = extract_context_sentence(note_body)
    region_text = f"NI {region.strip()}" if region else ""

    # Message LinkedIn plus naturel
    message = f"""Hello {first_name},
I hope you're doing well! I wanted to reconnect. {region_text} is focusing more on local customers, and we've recently added a new Sales Area Manager, {sales_manager}. {context_sentence}. Would you be open to a quick chat with {sales_manager} and/or me sometime in the next couple of weeks?
Best regards,
{full_name}
"""
    linkedin_url = f"https://www.google.com/search?q={quote(name)}+{quote(company)}+linkedin"

    return {
        "name": name,
        "company": company,
        "message": message,
        "linkedin": linkedin_url
    }

# ----------------------------------
# UI
# ----------------------------------
st.title("🔒 Local Personalised Notes Assistant")

with st.sidebar:
    st.header("👤 User")
    first_name = st.text_input("First name")
    last_name = st.text_input("Last name")
    full_name = f"{first_name} {last_name}".strip()
    sales_manager = st.text_input("Sales Area Manager (Name)")
    region = st.text_input("Region")

st.header("📂 Upload client notes")
uploaded_files = st.file_uploader(
    "Upload HTML, PDF or TXT files",
    type=["html", "pdf", "txt"],
    accept_multiple_files=True
)

all_notes = []

if uploaded_files:
    for file in uploaded_files:
        data = file.read()
        if file.type == "text/html":
            text = parse_html(data)
        elif file.type == "application/pdf":
            text = parse_pdf(data)
        else:
            text = parse_txt(data)
        all_notes.extend(extract_notes(text))

st.session_state["notes"] = all_notes
notes = st.session_state.get("notes", [])
st.success(f"{len(notes)} notes detected")

st.header("📒 Select notes")
all_options = ["📌 Select All"] + notes
selected_notes = st.multiselect(
    "Choose notes to process",
    all_options,
    format_func=lambda n: format_note_for_select(n) if n != "📌 Select All" else n
)
if "📌 Select All" in selected_notes:
    selected_notes = notes

st.header("⚙️ Action")
action = st.radio("What do you want to generate?", ["LinkedIn messages"])

if st.button("🚀 Generate") and selected_notes and full_name and sales_manager:
    st.divider()
    results = [generate_for_note(n, full_name, sales_manager, region) for n in selected_notes]

    st.header("📬 Generated messages")
    for r in results:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"### {r['name']}")
            st.markdown(f"*{r['company']}*")
            st.markdown(f"[🔗 Find on LinkedIn]({r['linkedin']})")

        with col2:
            st.text_area("Message", value=r["message"], height=200, key=f"msg_{r['name']}")
        st.divider()

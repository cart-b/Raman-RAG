import json
import unicodedata
import ollama
import pymupdf
import numpy as np
from collections import Counter

PDF_PATH = "lipid.pdf"
CHAT_MODEL = "qwen3:14b"
EMBED_MODEL = "nomic-embed-text"

QUESTION = "What does an increase in the 1650 cm^-1 peak mean for older worms?"

# Settings for each specificity level. These numbers are starting guesses to test.
LEVELS = {
    # label:      chunk words, overlap words, chunks to retrieve
    "extreme":   {"chunk_words": 100, "overlap_words": 25,  "top_k": 2},
    "very high": {"chunk_words": 150, "overlap_words": 40,  "top_k": 3},
    "high":      {"chunk_words": 200, "overlap_words": 50,  "top_k": 4},
    "medium":    {"chunk_words": 300, "overlap_words": 75,  "top_k": 6},
    "low":       {"chunk_words": 500, "overlap_words": 125, "top_k": 8},
}

SPECIFICITY_PROMPT = """Classify how specific the question below is. Reply with exactly one of these labels and nothing else:
low, medium, high, very high, extreme

Definitions:
- extreme: asks for one exact value, name, or wavenumber stated in a single sentence.
  Example: "What wavenumber was used to quantify intestinal lipid signal?"
- very high: asks for one fact, such as an assignment or a single result.
  Example: "What is the ~1,650 cm-1 band assigned to?"
- high: asks about one result plus its context or conditions, usually within one paragraph.
  Example: "How did BCARS and 2p-FLIM compare in classifying daf-2 particles?"
- medium: needs to combine facts from several parts of the paper, or to infer.
  Example: "What would a relatively higher ~1,650 cm-1 peak signify in older worms?"
- low: asks about the paper as a whole.
  Example: "What is this paper about?"

Question: {question}

Label:"""


def ask(prompt, fmt=None):
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        think=False,
        format=fmt,  # forces JSON output matching a schema when given
        options={"num_ctx": 8192},
    )
    return response["message"]["content"]


def classify(question):
    reply = ask(SPECIFICITY_PROMPT.format(question=question)).strip().lower()
    # Check "very high" before "high", since "high" is inside "very high"
    for label in ["extreme", "very high", "high", "medium", "low"]:
        if label in reply:
            return label
    return "high"  # safe default if the reply is unexpected


# ---------- Extract text once, removing repeated headers/footers ----------
def extract_words(path):
    doc = pymupdf.open(path)
    pages = [page.get_text().splitlines() for page in doc]
    line_counts = Counter(line.strip() for page in pages for line in set(page))
    words = []  # each item is (word, page_number)
    for page_num, lines in enumerate(pages, start=1):
        stripped = [l.strip() for l in lines]
        done = "References" in stripped
        if done:
            lines = lines[: stripped.index("References")]
        for line in lines:
            if line.strip() and line_counts[line.strip()] < 5:
                clean = unicodedata.normalize("NFKC", line)  # fixes ligatures like "ﬂ"
                words += [(w, page_num) for w in clean.split()]
        if done:
            break
    return words

WORDS = extract_words(PDF_PATH)


# ---------- Build chunks + embeddings once per level, then reuse ----------
_index_cache = {}

def get_index(label):
    if label in _index_cache:
        return _index_cache[label]
    s = LEVELS[label]
    chunks = []
    step = s["chunk_words"] - s["overlap_words"]
    for start in range(0, len(WORDS), step):
        piece = WORDS[start : start + s["chunk_words"]]
        chunks.append({"text": " ".join(w for w, _ in piece), "page": piece[0][1]})
        if start + s["chunk_words"] >= len(WORDS):
            break
    vecs = ollama.embed(
        model=EMBED_MODEL,
        input=["search_document: " + c["text"] for c in chunks],
    )["embeddings"]
    matrix = np.array(vecs)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    _index_cache[label] = (chunks, matrix)
    return _index_cache[label]


def search(query):
    label = classify(query)
    chunks, matrix = get_index(label)
    q = np.array(ollama.embed(
        model=EMBED_MODEL,
        input=["search_query: " + query],
    )["embeddings"][0])
    scores = matrix @ (q / np.linalg.norm(q))
    best = np.argsort(scores)[::-1][: LEVELS[label]["top_k"]]
    return label, [(chunks[i], float(scores[i])) for i in best]


# ---------- Stage 1: break the question into facts (JSON) ----------
DECOMPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string"},
                    "search_query": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["fact", "search_query", "why"],
            },
        }
    },
    "required": ["facts"],
}

decompose_prompt = f"""Break the question below into the separate facts that could be necesary to answer it.
For each fact give:
- fact: what needs to be established, phrased neutrally without assuming the answer
- search_query: an open question that could be looked up in a scientific paper to find that fact
- why: one sentence on why the fact is needed
- for questions related to raman peaks, keep in mind that raman peaks are always associated with the same chemical information. That is to say that a peak at a certain wavenumber is always coorelated with the same chemical identity regardless of what is being imaging
Return 3 to 6 facts. Respond in JSON.

Question: {QUESTION}"""

facts = json.loads(ask(decompose_prompt, fmt=DECOMPOSE_SCHEMA))["facts"]

print("--- Subquestions needed ---")
for n, f in enumerate(facts, 1):
    print(f"{n}. {f['fact']}\n   search: {f['search_query']}\n   why: {f['why']}")


# ---------- Stage 2: search and answer each fact separately ----------
FACT_PROMPT = """Answer the question using only the text below. Be brief (1-3 sentences). Report wavenumbers exactly as written. When the question mentions a peak or band:
- Look for that wavenumber in the texts. Report the value exactly as the text states it.
- Wavenumbers may be written in different forms, such as "1650", "1,650", "~1650 cm-1", or "1650 cm−1".

Inference: if the answer is not directly stated but can be inferred, give your best answer here. Do not quote. Instead, tag this answer as an assumption

Text:
{context}

Question: {query}"""

fact_results = []
for n, f in enumerate(facts, 1):
    label, hits = search(f["search_query"])
    context = "\n\n".join(f"[page {c['page']}]\n{c['text']}" for c, _ in hits)
    answer = ask(FACT_PROMPT.format(context=context, query=f["search_query"])).strip()
    fact_results.append({
        **f,
        "answer": answer,
        "level": label,
        "pages": sorted({c["page"] for c, _ in hits}),
    })
    print(f"\n[{n}] ({label}, pages {fact_results[-1]['pages']}) {f['search_query']}\n -> {answer}")


# ---------- Stage 3: combine findings into a final answer ----------
facts_block = "\n\n".join(
    f"Fact {n}: {r['fact']}\n"
    f"Why needed: {r['why']}\n"
    f"Finding: {r['answer']}"
    for n, r in enumerate(fact_results, 1)
)

final_prompt = f"""You are given findings that were each looked up separately in a scientific paper.

Findings:
{facts_block}

Question: {QUESTION}

Base your answer on the findings first. Findings may or may not be included in your answer.
Never invent results, measurements, or conclusions specific to this paper. Only say the
question cannot be answered if the findings and general knowledge together give no reasonable basis.

Format your reply as:
Answer: the direct answer, citing which facts (e.g. Fact 1, Fact 3) support each claim.
Assumptions: any general knowledge you used to fill gaps.
Caveats: overlapping bands (within about 30 cm-1) or other factors that could change the conclusion.
Confidence: high, medium, or low, with one sentence on why."""

print("\n--- Final answer ---")
print(ask(final_prompt))

from IPython.display import display
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import arxiv
import pandas as pd
import json
import urllib.request
import time
from pathlib import Path
from pypdf import PdfReader

Path("pdfs").mkdir(exist_ok=True)

topics = {
    "Machine Translation": '"machine translation"',
    "Question Answering": '"question answering"',
    "Summarization": "summarization",
    "Dialogue Systems": '"dialogue systems"'
}

date_range = "submittedDate:[202301010000 TO 202512312359]"

# Load previously collected papers
if Path("corpus.csv").exists():
    df = pd.read_csv("corpus.csv", dtype={"arxiv_id": str}).fillna("")
    data = df.to_dict("records")
else:
    data = []

seen = set(str(row["arxiv_id"]) for row in data)
queries = {}

client = arxiv.Client(
    page_size=100,
    delay_seconds=3,
    num_retries=3
)

for topic, keyword in topics.items():

    count = sum(row["sub_topic"] == topic for row in data)

    query = f"cat:cs.CL AND {keyword} AND {date_range}"
    queries[topic] = query

    print("Starting:", topic, count, "/100")

    search = arxiv.Search(
        query=query,
        max_results=600,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    try:
        for paper in client.results(search):

            if count >= 100:
                break

            paper_id = paper.get_short_id()

            if paper_id in seen:
                continue

            path = Path("pdfs") / (
                paper_id.replace("/", "_") + ".pdf"
            )

            try:
                if not path.exists():
                    urllib.request.urlretrieve(
                        paper.pdf_url, str(path)
                    )

                reader = PdfReader(str(path))
                pages = len(reader.pages)

                if pages == 6:
                    continue

                content = ""

                for page in reader.pages:
                    content += (page.extract_text() or "") + "\n"

                content = content.encode(
                    "utf-8", errors="replace"
                ).decode("utf-8")

                if not content.strip():
                    continue

                if pages < 6:
                    length = "Short"
                elif pages <= 9:
                    length = "Medium"
                else:
                    length = "Long"

                data.append({
                    "title": paper.title,
                    "content": content,
                    "arxiv_id": paper_id,
                    "arxiv_link": paper.entry_id,
                    "sub_topic": topic,
                    "pages": pages,
                    "length": length
                })

                seen.add(paper_id)
                count += 1

                pd.DataFrame(data).to_csv(
                    "corpus.csv",
                    index=False,
                    encoding="utf-8",
                    errors="replace"
                )

                print(topic, count)

            except Exception as e:
                print("Skipping", paper_id, ":", e)
                time.sleep(2)

    except Exception as e:
        print("Search interrupted:", e)

with open("collection_info.json", "w", encoding="utf-8") as file:
    json.dump({
        "category": "cs.CL",
        "date_range": date_range,
        "queries": queries,
        "target_per_topic": 100
    }, file, indent=4)

if data:
    df = pd.DataFrame(data)

    print("\nTotal papers:", len(df))

    print("\nPapers by topic:")
    print(df["sub_topic"].value_counts())

    print("\nPapers by length:")
    print(df["length"].value_counts())

else:
    print("No papers collected yet.")


# In[7]:


import pandas as pd
df = pd.read_csv("corpus.csv")

print("Total papers:", len(df))
print(df["sub_topic"].value_counts())
print(df["length"].value_counts())

display(df.head())


# In[8]:


import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# 1. Load dataset
df = pd.read_csv("corpus.csv")
df = df.fillna("")

# 2. Prepare documents
documents = (
    df["title"] + " " + df["content"].str[:2000]
).tolist()

# 3. BM25 (Sparse Retrieval)
tokenized_docs = [
    doc.lower().split() for doc in documents
]

bm25 = BM25Okapi(tokenized_docs)

# 4. Dense Retrieval
model = SentenceTransformer("all-MiniLM-L6-v2")

document_embeddings = model.encode(
    documents,
    normalize_embeddings=True,
    show_progress_bar=True
)

# 5. Hybrid Search using RRF
def hybrid_search(query, top_k=10):

    # BM25 scores
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)

    # Dense scores
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    dense_scores = np.dot(
        document_embeddings,
        query_embedding
    )

    # Rank both models
    bm25_ranking = np.argsort(-bm25_scores)
    dense_ranking = np.argsort(-dense_scores)

    # Reciprocal Rank Fusion
    rrf_scores = np.zeros(len(documents))

    k = 60

    for rank, index in enumerate(bm25_ranking):
        rrf_scores[index] += 1 / (k + rank + 1)

    for rank, index in enumerate(dense_ranking):
        rrf_scores[index] += 1 / (k + rank + 1)

    # Final fused ranking
    final_ranking = np.argsort(-rrf_scores)[:top_k]

    results = df.iloc[final_ranking].copy()
    results["score"] = rrf_scores[final_ranking]

    return results


# 6. Test the search engine
query = "machine translation using transformers"

results = hybrid_search(query)

# 7. Display top 10 results
for rank, (_, paper) in enumerate(results.iterrows(), 1):

    print("\nRank:", rank)
    print("Title:", paper["title"])
    print("Link:", paper["arxiv_link"])
    print("Snippet:", paper["content"][:200])
    print("RRF Score:", round(paper["score"], 5))
    print("-" * 50)


# In[12]:


import re
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from rapidfuzz import process, fuzz
from collections import Counter
from html import escape

# 1. Build vocabulary from the corpus
word_counts = Counter(
    word
    for doc in tokenized_docs
    for word in doc
)

vocabulary = [
    word for word, count in word_counts.items()
    if count >= 2 and word.isalpha()
]

# 2. Abbreviations
abbreviations = {
    "nlp": "natural language processing",
    "ai": "artificial intelligence",
    "ml": "machine learning",
    "mt": "machine translation",
    "qa": "question answering",
    "llm": "large language model"
}

# 3. Query processing
def process_query(query):

    query = query.lower().strip()

    # Expand abbreviations
    for short, full in abbreviations.items():
        query = re.sub(
            r"\b" + short + r"\b",
            full,
            query
        )

    # Correct spelling outside quoted phrases
    parts = re.split(r'("[^"]*")', query)
    corrected_parts = []

    for part in parts:

        if part.startswith('"') and part.endswith('"'):
            corrected_parts.append(part)
            continue

        words = re.findall(r"\w+|[^\w\s]", part)
        corrected = []

        for word in words:

            if (word.isalpha()
                    and len(word) >= 4
                    and word not in word_counts):

                match = process.extractOne(
                    word,
                    vocabulary,
                    scorer=fuzz.ratio,
                    score_cutoff=85
                )

                if match:
                    word = match[0]

            corrected.append(word)

        corrected_parts.append(" ".join(corrected))

    return " ".join(corrected_parts).strip()


# 4. Search function
def search_papers(query, method="Hybrid (RRF)"):

    processed = process_query(query)

    if not processed:
        return processed, []

    # BM25 scores
    bm25_scores = bm25.get_scores(
        processed.replace('"', '').split()
    ).copy()

    # Dense scores
    query_embedding = model.encode(
        processed.replace('"', ''),
        normalize_embeddings=True
    )

    dense_scores = np.dot(
        document_embeddings,
        query_embedding
    )

    # Exact phrase matching
    phrases = re.findall(r'"([^"]+)"', processed)

    phrase_matches = np.zeros(len(documents))

    for phrase in phrases:
        for i, doc in enumerate(documents):
            if phrase in doc.lower():
                phrase_matches[i] += 1

    # Phrase boost for BM25
    bm25_scores += phrase_matches * 5

    # Model rankings
    bm25_ranking = np.argsort(-bm25_scores)
    dense_ranking = np.argsort(-dense_scores)

    # Select model or fuse rankings
    if method == "BM25":
        scores = bm25_scores

    elif method == "Dense":
        scores = dense_scores + phrase_matches * 0.1

    else:
        # Reciprocal Rank Fusion
        scores = np.zeros(len(documents))

        for rank, index in enumerate(bm25_ranking):
            scores[index] += 1 / (60 + rank + 1)

        for rank, index in enumerate(dense_ranking):
            scores[index] += 1 / (60 + rank + 1)

        # Prioritize exact phrase matches
        scores += phrase_matches * 0.005

    ranking = np.argsort(-scores)[:10]

    results = []

    for index in ranking:
        paper = df.iloc[index]

        results.append({
            "title": paper["title"],
            "link": paper["arxiv_link"],
            "snippet": paper["content"][:200],
            "score": scores[index]
        })

    return processed, results


# 5. GUI
import ipywidgets as widgets
from IPython.display import display

search_box = widgets.Text(
    description="Search:",
    placeholder="Enter your query",
    value="nlp"
)

model_selector = widgets.Dropdown(
    options=["Hybrid (RRF)", "BM25", "Dense"],
    description="Model:"
)

button = widgets.Button(
    description="Search",
    button_style="success"
)

output = widgets.Output()

def run_search(b):
    output.clear_output()

    with output:
        try:
            print("Searching...")

            processed, results = search_papers(
                search_box.value,
                model_selector.value
            )

            print("Processed Query:", processed)
            print()

            for i, paper in enumerate(results, 1):
                print("Rank:", i)
                print("Title:", paper["title"])
                print("Link:", paper["link"])
                print("Snippet:", paper["snippet"])
                print("Score:", round(float(paper["score"]), 5))
                print("-" * 40)

        except Exception as e:
            print("Error:", str(e))

button.on_click(run_search)

display(search_box, model_selector, button, output)




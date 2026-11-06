
import streamlit as st
from search_engine import search_papers

st.title("Information Retrieval System")

query = st.text_input("Enter your search query")

model = st.selectbox(
    "Select Retrieval Model",
    ["BM25", "Dense", "Hybrid (RRF)"]
)

if st.button("Search"):
    if query:
        processed, results = search_papers(query, model)

        st.write("Processed Query:", processed)

        for i, paper in enumerate(results, 1):
            st.subheader(f"{i}. {paper['title']}")
            st.write("Score:", paper["score"])
            st.write(paper["snippet"])
            st.write(paper["link"])
    else:
        st.warning("Please enter a search query")

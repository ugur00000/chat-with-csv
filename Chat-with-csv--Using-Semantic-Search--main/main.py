import pandas as pd
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from llama_index.core.query_pipeline import QueryPipeline as QP, Link, InputComponent
from llama_index.experimental.query_engine.pandas import PandasInstructionParser
from llama_index.llms.ollama import Ollama
from llama_index.core.prompts import PromptTemplate
import os
import matplotlib.pyplot as plt
import re

# Start Streamlit in English
st.title("Chat with CSV and Visualize!")
st.write("Upload a CSV file and start asking questions in English!")

# Model selection (English)
model_type = st.selectbox("Choose your model", ["Llama3.1:8B (Local)"])
llm = Ollama(model="llama3.1:8b", request_timeout=120.0)

# File uploader (English)
uploaded_file = st.file_uploader("Upload a CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded File Preview:")
    st.write(df.head())

    # Embedding model (English)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Vectorize column names (English)
    column_embeddings = model.encode(df.columns.tolist())
    faiss.normalize_L2(column_embeddings)
    d = column_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(column_embeddings)

    # User query (English)
    query_str = st.text_input("Type your question here :", value="")

    if query_str:
        query_embedding = model.encode([query_str])
        faiss.normalize_L2(query_embedding)
        k = 3
        distances, indices = index.search(query_embedding, k)
        relevant_columns = [df.columns[i] for i in indices[0]]
        st.write("Most relevant columns:", relevant_columns)

        if relevant_columns:
            selected_columns = relevant_columns
            st.write(f"Selected columns for the query: {selected_columns}")

            # English prompt engineering
            instruction_str = (
                f"Translate the query into Python code that can be executed with Pandas, using only the columns: {', '.join(selected_columns)}.\n"
                "If complex operations are needed, consider using functions like grouping, aggregation, merging, or reshaping.\n"
                "If a plot is requested, consider creating the appropriate plot using matplotlib.\n"
                "Handle missing data appropriately.\n"
                "End the code with a Python expression that can be executed with the `eval()` function.\n"
                "PRINT ONLY THE EXPRESSION.\n"
                "Do not enclose the expression in quotes.\n"
                "Use only the English language.\n"
            )

            pandas_prompt_str = (
                "You are working with a pandas dataframe in Python.\n"
                "The name of the dataframe is `df`.\n"
                "This is the result of `print(df.head())`:\n"
                "{df_str}\n\n"
                "Follow these instructions:\n"
                "{instruction_str}\n"
                "Query: {query_str}\n\n"
                "Expression:"
            )

            pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
                instruction_str=instruction_str, df_str=df[selected_columns].head(5)
            )

            class CleanedPandasInstructionParser(PandasInstructionParser):
                def parse(self, output: str):
                    # Remove markdown code block markers and language identifiers
                    cleaned = re.sub(r"^```[a-zA-Z]*\\n?", "", output.strip())
                    cleaned = re.sub(r"```$", "", cleaned.strip())
                    return super().parse(cleaned)

            pandas_output_parser = CleanedPandasInstructionParser(df[selected_columns])
            response_synthesis_prompt = PromptTemplate(
                "Generate a detailed response from the query results based on your input.\n"
                "Query: {query_str}\n\n"
                "Pandas Instructions:\n{pandas_instructions}\n\n"
                "Pandas Output: {pandas_output}\n\n"
                "Response: "
            )

            qp = QP(
                modules={
                    "input": InputComponent(),
                    "pandas_prompt": pandas_prompt,
                    "llm1": llm,
                    "pandas_output_parser": pandas_output_parser,
                    "response_synthesis_prompt": response_synthesis_prompt,
                    "llm2": llm,
                },
                verbose=True,
            )
            qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
            qp.add_links([
                Link("input", "response_synthesis_prompt", dest_key="query_str"),
                Link("llm1", "response_synthesis_prompt", dest_key="pandas_instructions"),
                Link("pandas_output_parser", "response_synthesis_prompt", dest_key="pandas_output"),
            ])
            qp.add_link("response_synthesis_prompt", "llm2")
            fig, ax = plt.subplots()
            response = qp.run(query_str=query_str)
            st.write("Response:")
            st.write(response.message.content)

            if "visualize" in response.message.content.lower():
                st.session_state['fig'] = fig
                st.pyplot(fig=fig)
                plt.close(fig)

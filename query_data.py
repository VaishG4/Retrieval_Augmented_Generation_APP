import argparse
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from transformers import pipeline, AutoTokenizer


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Paraphrase function using HuggingFace transformers (T5-base paraphraser)
def generate_paraphrases(query, num_return_sequences=3):
    # Use the slow tokenizer to avoid tiktoken/blobfile dependency
    tokenizer = AutoTokenizer.from_pretrained(
        "ramsrigouthamg/t5_paraphraser",
        use_fast=False,
    )
    paraphraser = pipeline(
        "text2text-generation",
        model="ramsrigouthamg/t5_paraphraser",
        tokenizer=tokenizer,
        framework="pt",
        device=-1,
    )
    input_text = f"paraphrase: {query}"
    outputs = paraphraser(
        input_text,
        max_length=64,
        num_return_sequences=num_return_sequences,
        num_beams=10,
        do_sample=True
    )
    paraphrases = list(set([o['generated_text'].strip() for o in outputs]))
    # Always include the original query
    if query not in paraphrases:
        paraphrases.insert(0, query)
    return paraphrases



def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},  # or "cuda" if you have GPU
        encode_kwargs={"normalize_embeddings": True}
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Generate paraphrased queries (multi-query)
    print("Generating paraphrased queries...")
    queries = generate_paraphrases(query_text, num_return_sequences=3)
    print(f"Queries used for search: {queries}")

    # Search the DB for each query and aggregate results
    all_results = []
    for q in queries:
        results = db.similarity_search_with_relevance_scores(q, k=3)
        all_results.extend(results)

    # Deduplicate documents by page_content (or use doc.metadata['source'] if available)
    seen = set()
    unique_results = []
    for doc, score in all_results:
        key = (doc.page_content, doc.metadata.get("source", None))
        if key not in seen:
            seen.add(key)
            unique_results.append((doc, score))

    if len(unique_results) == 0:
        print(f"Unable to find matching results.")
        return

    # Optionally, sort by score (lower is better if using distance, higher is better if using similarity)
    unique_results = sorted(unique_results, key=lambda x: -x[1])

    # Build context from top N unique results (e.g., top 3)
    top_results = unique_results[:3]
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in top_results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOllama(model="llama3:8b", temperature=0)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in top_results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()

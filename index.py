import math
import re
from pathlib import Path
from collections import Counter

# Documents folder
DOC_DIR = Path("documents")


# Convert text into words
def tokenize(text):
    return re.findall(r"[a-zA-Z]+", text.lower())


# Calculate Term Frequency (TF)
def calculate_tf(tokens):
    counts = Counter(tokens)
    total_words = len(tokens)

    return {
        word: count / total_words
        for word, count in counts.items()
    }


# Read all documents
documents = {}

for file in sorted(DOC_DIR.glob("*.txt")):
    documents[file.name] = file.read_text(encoding="utf-8")


# Convert documents into tokens
document_tokens = {
    name: tokenize(text)
    for name, text in documents.items()
}

number_of_documents = len(document_tokens)


# Calculate Document Frequency (DF)
document_frequency = Counter()

for tokens in document_tokens.values():
    document_frequency.update(set(tokens))


# Calculate Inverse Document Frequency (IDF)
def calculate_idf(word):
    return math.log(
        (number_of_documents + 1)
        / (document_frequency.get(word, 0) + 1)
    ) + 1


# Create TF-IDF vectors for documents
document_vectors = {}

for name, tokens in document_tokens.items():

    tf_values = calculate_tf(tokens)

    document_vectors[name] = {
        word: tf_values[word] * calculate_idf(word)
        for word in tf_values
    }


# Display project title
print("=" * 70)
print("          INFORMATION RETRIEVAL SYSTEM")
print("       TF-IDF WEIGHTAGE AND DOCUMENT RANKING")
print("=" * 70)


# Take input query
while True:

    query = input("\nEnter your query (or type 'exit' to stop): ")

    if query.lower() == "exit":
        print("\nProgram ended.")
        break

    query_tokens = tokenize(query)

    if not query_tokens:
        print("Please enter a valid query.")
        continue


    # Calculate query TF
    query_tf = calculate_tf(query_tokens)


    # Calculate query TF-IDF
    query_vector = {
        word: query_tf[word] * calculate_idf(word)
        for word in query_tf
    }


    # Calculate cosine similarity
    scores = {}

    for document_name, document_vector in document_vectors.items():

        common_words = (
            set(query_vector)
            & set(document_vector)
        )

        dot_product = sum(
            query_vector[word] * document_vector[word]
            for word in common_words
        )

        query_length = math.sqrt(
            sum(value ** 2 for value in query_vector.values())
        )

        document_length = math.sqrt(
            sum(value ** 2 for value in document_vector.values())
        )

        if query_length != 0 and document_length != 0:
            similarity = (
                dot_product
                / (query_length * document_length)
            )
        else:
            similarity = 0

        scores[document_name] = similarity


    # Rank documents from highest score to lowest
    ranked_documents = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )


    # Display results
    print("\nDOCUMENT RANKING")
    print("-" * 70)

    for rank, (document_name, score) in enumerate(
        ranked_documents,
        start=1
    ):

        print(
            f"Rank {rank:<2} | "
            f"{document_name:<10} | "
            f"Weightage / Similarity: {score:.4f}"
        )

    print("-" * 70)
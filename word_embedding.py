"""
Implementation for https://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec/#3
"""

from collections import Counter

import nltk
import numpy as np
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

# Sample text corpus (replace this with your corpus)
text_corpus = (
    "Natural language processing enables computers to understand human language. "
    "Language is a core part of many machine learning applications."
)

# Ensure necessary NLTK data is available
nltk.download("punkt")

# Tokenize the text
tokens = word_tokenize(text_corpus.lower())

# Calculate Unigram Probability
freq_dist = FreqDist(tokens)
total_words = len(tokens)
unigram_probability = {word: count / total_words for word, count in freq_dist.items()}

# Generate Skipgrams
window_size = 2  # Define the window size for the skipgrams
skipgrams = []
for i, target_word in enumerate(tokens):
    start_index = max(0, i - window_size)
    end_index = min(len(tokens), i + window_size + 1)
    for j in range(start_index, end_index):
        if i != j:
            context_word = tokens[j]
            skipgrams.append((target_word, context_word))

# Calculate Skipgram Probability
skipgram_count = Counter(skipgrams)
total_skipgrams = len(skipgrams)
skipgram_probability = {
    pair: count / total_skipgrams for pair, count in skipgram_count.items()
}

# Number of words and Initializing PMI matrix
words = list(unigram_probability.keys())
num_words = len(words)
pmi_matrix = np.zeros((num_words, num_words))

# Calculate PMI and store in the matrix
for i, word1 in enumerate(words):
    for j, word2 in enumerate(words):
        prob_word1 = unigram_probability.get(word1, 0)
        prob_word2 = unigram_probability.get(word2, 0)
        prob_word_pair = skipgram_probability.get((word1, word2), 0)
        if prob_word1 > 0 and prob_word2 > 0 and prob_word_pair > 0:
            pmi = np.log2(prob_word_pair / (prob_word1 * prob_word2))
            pmi_matrix[i, j] = max(pmi, 0)  # Use max to avoid negative PMI values

# Convert PMI Matrix to Sparse Matrix and Apply SVD
pmi_sparse_matrix = csr_matrix(pmi_matrix)
u, s, vt = svds(pmi_sparse_matrix, k=5)  # k is the number of dimensions
word_vectors = u


# Function to find similar words
def find_similar_words(target_word, word_vectors, word_index, words, top_n=5):
    if target_word not in word_index:
        return "Target word not in vocabulary."

    target_idx = word_index[target_word]
    target_word_vector = word_vectors[target_idx].reshape(1, -1)
    similarities = cosine_similarity(target_word_vector, word_vectors).flatten()
    similar_indices = (-similarities).argsort()[1 : top_n + 1]
    similar_words = [(words[idx], similarities[idx]) for idx in similar_indices]

    return similar_words


# Create a word index mapping
word_index = {word: i for i, word in enumerate(words)}

# Example: Find words similar to 'language'
similar_words = find_similar_words("language", word_vectors, word_index, words)
print("Words similar to 'language':", similar_words)

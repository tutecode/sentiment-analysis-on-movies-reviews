from typing import List

import numpy as np
from gensim.models.word2vec import Word2Vec
from typing import List
from gensim.models import KeyedVectors


def vectorizer(
    corpus: List[List[str]], model: Word2Vec, num_features: int = 100
) -> np.ndarray:
    """
    This function takes a list of tokenized text documents (corpus) and a pre-trained
    Word2Vec model as input, and returns a matrix where each row represents the
    vectorized form of a document.

    Args:
        corpus : list
            A list of text documents that needs to be vectorized.

        model : Word2Vec
            A pre-trained Word2Vec model that will be used to vectorize the corpus.

        num_features : int
            The size of the vector representation of each word. Default is 100.

    Returns:
        corpus_vectors : numpy.ndarray
            A 2D numpy array where each row represents the vectorized form of a
            document in the corpus.
    """
    # Get the number of documents in the corpus
    num_docs = len(corpus)

    # Initialize an empty matrix to store the document vectors
    corpus_vectors = np.zeros((num_docs, num_features))

    # Iterate over each document in the corpus
    for i, document in enumerate(corpus):
        # Initialize an empty vector to store the document's vector representation
        doc_vector = np.zeros(num_features)

        # Initialize a counter to keep track of the number of words in the document
        num_words = 0

        # Iterate over each word in the document
        for word in document:
            # Check if the word is present in the Word2Vec model's vocabulary
            if word in model.wv:
                # Add the word's vector representation to the document vector
                doc_vector += model.wv[word]
                num_words += 1

        # Calculate the average vector by dividing the sum of word vectors by the number of words
        if num_words > 0:
            doc_vector /= num_words

        # Assign the document vector to the corresponding row in the corpus_vectors matrix
        corpus_vectors[i] = doc_vector

    return corpus_vectors


def vectorizer_pre_trained(
    corpus: List[List[str]], model: KeyedVectors, num_features: int = 300
) -> np.ndarray:
    """
    This function takes a list of tokenized text documents (corpus) and a pre-trained
    Word2Vec model as input, and returns a matrix where each row represents the
    vectorized form of a document.

    Args:
        corpus : list
            A list of text documents that needs to be vectorized.

        model : KeyedVectors
            A pre-trained Word2Vec model that will be used to vectorize the corpus.

        num_features : int
            The size of the vector representation of each word. Default is 100.

    Returns:
        corpus_vectors : numpy.ndarray
            A 2D numpy array where each row represents the vectorized form of a
            document in the corpus.
    """
    corpus_vectors = []
    for document in corpus:
        document_vector = np.zeros(num_features)
        word_count = 0
        for word in document:
            if word in model:
                document_vector += model[word]
                word_count += 1
        if word_count > 0:
            document_vector /= word_count
        corpus_vectors.append(document_vector)
    return np.array(corpus_vectors)
from datasets import load_dataset
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

from rouge_score import rouge_scorer


# loading dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Taking a small subset for analysis
subset = dataset[:100]  # First 100 entries

#print(type(subset))

texts = [text for text in subset['text'] if text.strip()]

# texts is in the type -> list



def fixed_length_chunking(texts, chunk_size=256):
    chunks = []
    for text in texts:
        tokens = text.split()  # Tokenize by whitespace
        for i in range(0, len(tokens), chunk_size):
            chunks.append(" ".join(tokens[i:i + chunk_size]))
    return chunks

fixed_chunks = fixed_length_chunking(texts)



def sentence_based_chunking(texts):
    chunks = []
    for text in texts:
        sentences = sent_tokenize(text)
        chunks.extend(sentences)  
    return chunks

sentence_chunks = sentence_based_chunking(texts)


def paragraph_based_chunking(texts):
    chunks = []
    for text in texts:
        paragraphs = text.split("\n\n")  # Split by double newline
        for para in paragraphs:
            if para.strip():  # Ignore empty paragraphs
                chunks.append(para.strip())
    return chunks

paragraph_chunks = paragraph_based_chunking(texts)




def information_density(chunks):
    return [len(set(chunk.split())) / len(chunk.split()) for chunk in chunks]

fixed_density = information_density(fixed_chunks)
sentence_density = information_density(sentence_chunks)
paragraph_density = information_density(paragraph_chunks)




def semantic_coherence(chunks):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(chunks)
    coherence = []
    for i in range(len(chunks) - 1):
        coherence.append(cosine_similarity(vectors[i], vectors[i + 1])[0, 0])
    return coherence

fixed_coherence = semantic_coherence(fixed_chunks)
sentence_coherence = semantic_coherence(sentence_chunks)
paragraph_coherence = semantic_coherence(paragraph_chunks)


def context_overlap(chunks):
    overlaps = []
    for i in range(len(chunks) - 1):
        tokens_a = set(chunks[i].split())
        tokens_b = set(chunks[i + 1].split())
        overlap = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)  # Jaccard similarity
        overlaps.append(overlap)
    return overlaps

fixed_overlap = context_overlap(fixed_chunks)
sentence_overlap = context_overlap(sentence_chunks)
paragraph_overlap = context_overlap(paragraph_chunks)


# Token Preservation
def token_preservation(original_text, chunks):
    if isinstance(original_text, list):
        original_text = ' '.join(original_text)
    original_tokens = set(word_tokenize(original_text))
    chunked_tokens = set(word for chunk in chunks for word in word_tokenize(chunk))
    return len(original_tokens & chunked_tokens) / len(original_tokens)

fixed_token_preservation =  token_preservation(texts, fixed_chunks)
sentence_token_preservation = token_preservation(texts, sentence_chunks)
paragraph_token_preservation = token_preservation(texts, paragraph_chunks)




# Function to calculate cosine similarity
def semantic_preservation_cosine(original_text, chunks):
    if isinstance(original_text, list):
        original_text = ' '.join(original_text)
    
    reconstructed_text = " ".join(chunks)
    texts = [original_text, reconstructed_text]

    # Convert texts to vectors using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return similarity[0][0]

# Apply to both chunking methods
cosine_fixed = semantic_preservation_cosine(texts, fixed_chunks)
cosine_sentence = semantic_preservation_cosine(texts, sentence_chunks)
cosine_paragraph = semantic_preservation_cosine(texts, paragraph_chunks)


# reconstructing text from chunks
def reconstruct_text(chunks):
    return " ".join(chunks)

# calulating ROUGE score
def semantic_preservation(original_text, chunks):
    
    if isinstance(original_text, list):
        original_text = ' '.join(original_text)
    reconstructed_text = reconstruct_text(chunks)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(original_text, reconstructed_text)
    return scores


rouge_fixed = semantic_preservation(texts, fixed_chunks)
rouge_sentence = semantic_preservation(texts, sentence_chunks)
rouge_paragraph = semantic_preservation(texts, paragraph_chunks)


# Token Efficiency Calculation
def token_efficiency(original_text, chunks):
    if isinstance(original_text, list):
        original_text = ' '.join(original_text)
    original_tokens = word_tokenize(original_text)
    chunked_tokens = [word for chunk in chunks for word in word_tokenize(chunk)]
    return len(chunked_tokens) / len(original_tokens)


efficiency_fixed = token_efficiency(texts, fixed_chunks)
efficiency_sentence = token_efficiency(texts, sentence_chunks)
efficiency_paragraph = token_efficiency(texts, paragraph_chunks)



print("Fixed-Length Chunking:")
print("  Avg. Information Density:", np.mean(fixed_density))
print("  Avg. Semantic Coherence:", np.mean(fixed_coherence))


print("Sentence-Based Chunking:")
print("  Avg. Information Density:", np.mean(sentence_density))
print("  Avg. Semantic Coherence:", np.mean(sentence_coherence))

print("Paragraph Based Chunking")
print("  Avg. Information Density:", np.mean(paragraph_density))
print("  Avg. Semantic Coherence:", np.mean(paragraph_coherence))


print("Context Overlap Patterns:")
print("  Fixed-Length Chunking Avg. Overlap:", np.mean(fixed_overlap))
print("  Sentence-Based Chunking Avg. Overlap:", np.mean(sentence_overlap))
print("  Paragraph-Based Chunking Avg. Overlap:", np.mean(paragraph_overlap))

print("Cosine Similarity:")
print("Cosine Similarity - Fixed-Size:", cosine_fixed)
print("Cosine Similarity - Sentence:", cosine_sentence)
print("Cosine Similarity - Paragraph:", cosine_paragraph)

print("ROUGE Scores:")
print("ROUGE Scores - Fixed-Size:", rouge_fixed)
print("ROUGE Scores - Sentence:", rouge_sentence)
print("ROUGE Scores - Paragraph:", rouge_paragraph)


print("Token Efficiency - Fixed-Size:", efficiency_fixed)
print("Token Efficiency - Sentence:", efficiency_sentence)
print("Token Efficiency - Paragraph:", efficiency_paragraph)


'''
Fixed-Length Chunking:
  Avg. Information Density: 0.7206336435170604
  Avg. Semantic Coherence: 0.1596200734299183
Sentence-Based Chunking:
  Avg. Information Density: 0.8815712426052792
  Avg. Semantic Coherence: 0.09510924638631618
Paragraph Based Chunking
  Avg. Information Density: 0.7063421175665089
  Avg. Semantic Coherence: 0.17183122839030074
Context Overlap Patterns:
  Fixed-Length Chunking Avg. Overlap: 0.09378667994940455
  Sentence-Based Chunking Avg. Overlap: 0.10745108836414907
  Paragraph-Based Chunking Avg. Overlap: 0.09885226335435049
Cosine Similarity:
Cosine Similarity - Fixed-Size: 1.000000000000001
Cosine Similarity - Sentence: 1.000000000000001
Cosine Similarity - Paragraph: 1.000000000000001
ROUGE Scores:
ROUGE Scores - Fixed-Size: {'rouge1': Score(precision=1.0, recall=1.0, fmeasure=1.0), 'rougeL': Score(precision=1.0, recall=1.0, fmeasure=1.0)}
ROUGE Scores - Sentence: {'rouge1': Score(precision=1.0, recall=1.0, fmeasure=1.0), 'rougeL': Score(precision=1.0, recall=1.0, fmeasure=1.0)}
ROUGE Scores - Paragraph: {'rouge1': Score(precision=1.0, recall=1.0, fmeasure=1.0), 'rougeL': Score(precision=1.0, recall=1.0, fmeasure=1.0)}
Token Efficiency - Fixed-Size: 1.0
Token Efficiency - Sentence: 1.0
Token Efficiency - Paragraph: 1.0
'''
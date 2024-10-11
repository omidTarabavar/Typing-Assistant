import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
from sklearn.model_selection import train_test_split
import joblib

def load_data():
    with open("./data/en_US.blogs.txt", "r", encoding = "utf8") as f:
        data = f.read()
    return data

def tokenize(data):
    sentences = data.split("\n")
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 0]
    tokenized_sentences = []
    for sentence in sentences:
        tokenized_sentences.append(nltk.word_tokenize(sentence.lower()))
    return tokenized_sentences

def count_words(tokenized_sentences):
    word_counts = {}
    for sentence in tokenized_sentences:
        for token in sentence:
            if token not in word_counts.keys():
                word_counts[token] = 0
            word_counts[token] += 1
    return word_counts

def create_vocab(tokenized_sentences, count_threshold):
    vocabulary = []
    word_counts = count_words(tokenized_sentences)
    for word, count in word_counts.items():
        if count >= count_threshold:
            vocabulary.append(word)
    return vocabulary

def replace_oov_words(tokenized_sentences, vocabulary, unknown_token = "<unk>"):
    vocabulary = set(vocabulary)
    replaced_tokenized_sentences = []
    for sentence in tokenized_sentences:
        replaced_sentence = []
        for token in sentence:
            if token in vocabulary:
                replaced_sentence.append(token)
            else:
                replaced_sentence.append(unknown_token)
        replaced_tokenized_sentences.append(replaced_sentence)
    return replaced_tokenized_sentences

def count_n_grams(tokenized_sentences, n, start_token = "<s>", end_token = "<e>"):
    n_grams = {}
    for sentence in tokenized_sentences:
        sentence = [start_token] * n + sentence + [end_token]
        sentence = tuple(sentence) # n_grams are immutable so we use tuple
        m = len(sentence) if n == 1 else len(sentence) - n + 1
        for i in range(m):
            n_gram = sentence[i:i+n]
            if n_gram not in n_grams.keys():
                n_grams[n_gram] = 0
            n_grams[n_gram] += 1
    return n_grams

def get_n_grams(tokenized_sentences):
    n_gram_count_list = []
    for n in range(1,6):
        n_model_counts = count_n_grams(tokenized_sentences,n)
        n_gram_count_list.append(n_model_counts)
    return n_gram_count_list

def estimate_probability(word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    previous_n_gram = tuple(previous_n_gram)
    previous_n_gram_count = n_gram_counts[previous_n_gram] if previous_n_gram in n_gram_counts else 0
    n_plus1_gram = previous_n_gram + (word,)
    n_plus1_gram_count = n_plus1_gram_counts[n_plus1_gram] if n_plus1_gram in n_plus1_gram_counts else 0
    numerator = n_plus1_gram_count + k
    denominator = previous_n_gram_count + k*vocabulary_size
    probability = numerator / denominator
    return probability

def probability_of_words(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, end_token = "<e>", unknown_token = "<unk>", k = 1.0):
    previous_n_gram = tuple(previous_n_gram)
    vocabulary += [end_token, unknown_token]
    vocabulary_size = len(vocabulary)
    probabilities = {}
    for word in vocabulary:
        probability = estimate_probability(word,previous_n_gram,n_gram_counts,n_plus1_gram_counts,vocabulary_size,k)
        probabilities[word] = probability
    return probabilities

def suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, end_token = "<e>", unknown_token = "<unk>", k = 1.0):
    n = len(list(n_gram_counts.keys())[0])
    previous_tokens = ['<s>'] * n + previous_tokens
    previous_n_gram = previous_tokens[-n:]
    probabilities = probability_of_words(previous_n_gram,n_gram_counts,n_plus1_gram_counts,vocabulary,end_token,unknown_token,k)
    suggestion = None
    max_prob = 0.0
    for word, prob in probabilities.items():
        if prob > max_prob:
            suggestion = word
            max_prob = prob
    return suggestion,max_prob

def get_multiple_suggestions(previous_tokens, n_grams_count_list,vocabulary,k = 1.0):
    model_counts = len(n_grams_count_list)
    suggestions = []
    for i in range(model_counts-1):
        n_gram_counts = n_grams_count_list[i]
        n_plus1_gram_counts = n_grams_count_list[i+1]
        suggestion = suggest_a_word(previous_tokens,n_gram_counts,n_plus1_gram_counts,vocabulary,k)
        suggestions.append(suggestion)
    return suggestions

def main():
    data = load_data()
    train = tokenize(data)
    count_threshold = 2
    unknown_token = "<unk>"
    vocabulary = create_vocab(train,count_threshold)
    train_replaced = replace_oov_words(train,vocabulary,unknown_token)
    n_grams_count_list = get_n_grams(train_replaced)

    while True:
        str_input = input(f"\nEnter a string: ")
        tokenized_input = nltk.word_tokenize(str_input.strip())
        print(get_multiple_suggestions(tokenized_input,n_grams_count_list,vocabulary))
        print()

if __name__ == "__main__":
    main()
import re
import unicodedata
import string
import nltk
nltk.download('punkt')
import unicodedata
from nltk.probability import ConditionalFreqDist, ConditionalProbDist, MLEProbDist
import collections

def load_data():
    with open("./data/en_US.blogs.txt", "r", encoding = "utf8") as f:
        data = f.read()
    return data

def filter(text):
    # normalize text
    text = (unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore'))
    # replace html chars with ' '
    text = re.sub('<.*?>', ' ', text)
    # remove punctuation
    text = text.translate(str.maketrans(' ', ' ', string.punctuation))
    # only alphabets and numerics
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    # replace newline with space
    text = re.sub("\n", " ", text)
    # lower case
    text = text.lower()
    # split and join the words
    text = ' '.join(text.split())

    return text

def clean(text):
    tokens = nltk.word_tokenize(text)
    return tokens

def create_vocabulary(tokenized_text):
    return set(tokenized_text)


def pyNgrams(text,n):
    ngrams = zip(*[text[i:] for i in range(n)])
    counts = collections.Counter(ngrams)
    return counts

def get_ngrams(text):
    ngrams = {}
    for n in range(1,5):
        n_gram = pyNgrams(text,n)
        ngrams[n] = n_gram
        print(f"N = {n} created.")
    return ngrams

def estimate_probability(word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=0.1):
    previous_n_gram_count = n_gram_counts[previous_n_gram] if previous_n_gram in n_gram_counts else 0
    n_plus1_gram = previous_n_gram + (word,)
    n_plus1_gram_count = n_plus1_gram_counts[n_plus1_gram] if n_plus1_gram in n_plus1_gram_counts else 0
    numerator = n_plus1_gram_count + k
    denominator = previous_n_gram_count + k*vocabulary_size
    probability = numerator / denominator
    return probability

def predict(n,previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, end_token = "<e>", unknown_token = "<unk>", k = 0.1):
    previous_tokens = ['<s>'] * n + previous_tokens
    previous_tokens = previous_tokens[-n:]
    previous_n_gram = tuple(previous_tokens)
    vocabulary_size = len(vocabulary)
    suggestion = None
    max_prob = 0.0
    for word in vocabulary:
        probability = estimate_probability(word,previous_n_gram,n_gram_counts,n_plus1_gram_counts,vocabulary_size,k)
        if probability > max_prob:
            suggestion = word
            max_prob = probability
    return suggestion, max_prob

def get_multiple_suggestions(previous_tokens, n_gram_list,vocabulary,k = 0.1):
    model_counts = len(n_gram_list)
    suggestions = []
    for n in range(1,model_counts):
        n_gram_counts = n_gram_list[n]
        n_plus1_gram_counts = n_gram_list[n+1]
        suggestion = predict(n,previous_tokens,n_gram_counts,n_plus1_gram_counts,vocabulary,k)
        suggestions.append(suggestion)
    return suggestions


def main():
    data = load_data()
    data = "Hello\n my name is omid\n I am a window hello my name but they call me human, lol!\nI'm filling this so it would be a big text\nhahaha!!!"
    print("Filtering...")
    data = filter(data)
    print("Cleaning...")
    data = clean(data)
    print("Creating Vocabulary...")
    vocabulary = create_vocabulary(data)
    vocab_size = len(vocabulary)
    print("Creating probability distribution for N-Grams...")
    n_gram_list = get_ngrams(data)
    while True:
        print("Enter a phrase: ")
        previous_tokens = clean(input())
        print(get_multiple_suggestions(previous_tokens,n_gram_list,vocabulary))
        print()

main()
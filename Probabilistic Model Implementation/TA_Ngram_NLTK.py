import re
import unicodedata
import string
import nltk
nltk.download('punkt')
import unicodedata
from nltk.probability import ConditionalFreqDist, ConditionalProbDist, MLEProbDist

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

def ngram_probDist(text,n):
    grams = list(nltk.ngrams(text, n, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))  
    # make conditional frequencies dictionary
    gram_pairs = [(tuple(gram[:n-1]), gram[n-1]) for gram in grams]
    cfdist = ConditionalFreqDist(gram_pairs)
    return cfdist

def get_ngrams(text):
    ngrams = {}
    for n in range(1,5):
        n_gram = ngram_probDist(text,n)
        ngrams[n] = n_gram
        print(f"N = {n} created.")
    return ngrams

def predict(freq_dists, input_text, vocabulary_size, k = 0.1):
    input_text = filter(input_text)
    input_text = input_text.split()
    predictions = {}
    for n in range(2,5):
        if len(input_text) >= n-1:
            indx_start = len(input_text) - n + 1
            prev_words = tuple(input_text[indx_start:])

            if prev_words in freq_dists[n]:
                next_word_counts = freq_dists[n][prev_words]
                total_count = sum(next_word_counts.values())
                probabilities = {word: count / total_count for word, count in next_word_counts.items()}
                top_predicted_word = max(probabilities, key=probabilities.get)
                predictions[n] = top_predicted_word
            else:
                smoothed_probs = {}

                if n > 2 and prev_words[-1:] in freq_dists[n-1]:
                    lower_order_counts = freq_dists[n - 1][prev_words[-1:]]
                    total_count = sum(lower_order_counts.values())

                    for word in lower_order_counts:
                        smoothed_probs[word] = (lower_order_counts[word] + k ) / (total_count + k * vocabulary_size)

                    top_predicted_word = max(smoothed_probs, key=smoothed_probs.get)
                    predictions[n] = top_predicted_word
                else:
                    predictions[n] = None
        else:
            predictions[n] = None
    return predictions

def main():
    data = load_data()
    print("Filtering...")
    data = filter(data)
    print("Cleaning...")
    data = clean(data)
    print("Creating Vocabulary...")
    vocabulary = create_vocabulary(data)
    vocab_size = len(vocabulary)
    print("Creating probability distribution for N-Grams...")
    prob_dists = get_ngrams(data)
    while True:
        print("Enter a phrase: ")
        input_text = input()
        print(predict(prob_dists,input_text,vocab_size))
        print()

main()
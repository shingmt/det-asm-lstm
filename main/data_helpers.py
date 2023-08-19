import numpy as np
import re
import itertools
from collections import Counter
import json

"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""

def clean_str_(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(train_pos_txt_path, train_neg_txt_path, test_pos_txt_path, test_neg_txt_path):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load (train) data from files
    print('train_pos_txt_path', train_pos_txt_path)
    positive_examples = list(open(train_pos_txt_path, encoding='latin-1').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(train_neg_txt_path, encoding='latin-1').readlines())
    negative_examples = [s.strip() for s in negative_examples]

    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)



    # Load (test) data from files
    print('test_pos_txt_path', test_pos_txt_path)
    _positive_examples = list(open(test_pos_txt_path, encoding='latin-1').readlines())
    _positive_examples = [s.strip() for s in _positive_examples]
    _negative_examples = list(open(test_neg_txt_path, encoding='latin-1').readlines())
    _negative_examples = [s.strip() for s in _negative_examples]
    # Split by words
    _x_text = _positive_examples + _negative_examples
    _x_text = [clean_str(sent) for sent in _x_text]
    _x_text = [s.split(" ") for s in _x_text]
    # Generate labels
    _positive_labels = [[0, 1] for _ in _positive_examples]
    _negative_labels = [[1, 0] for _ in _negative_examples]
    _y = np.concatenate([_positive_labels, _negative_labels], 0)

    return [x_text, y, _x_text, _y]


def pad_sentences(sentences, padding_word="<PAD/>", sequence_length=None):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    if sequence_length is None:
        sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    # print('[pad_sentences] sequence_length', sequence_length)
    for i in range(len(sentences)):
        sentence = sentences[i]
        # print('[pad_sentences] sentence', sentence, sequence_length, len(sentence))
        if sequence_length > len(sentence):
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
            # print('[pad_sentences] (pad) new_sentence', new_sentence, len(new_sentence))
            padded_sentences.append(new_sentence)
        else:
            new_sentence = sentence[:sequence_length]
            # print('[pad_sentences] (trim) new_sentence', new_sentence, len(new_sentence))
            padded_sentences.append(new_sentence)
    return padded_sentences, sequence_length


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    # x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    x = []
    for sentence in sentences:
        xs = []
        for word in sentence:
            if word in vocabulary:
                xs.append(vocabulary[word])
            else:
                xs.append(0)
        x.append(xs)
    x = np.array(x)
    y = np.array(labels)
    return [x, y]


def build_input_data_from_sentences(sentences, vocabulary):
    """
    Maps sentencs to vectors based on a vocabulary.
    """
    # x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    x = []
    for sentence in sentences:
        xs = []
        for word in sentence:
            if word in vocabulary:
                xs.append(vocabulary[word])
            else:
                xs.append(0)
        x.append(xs)
    x = np.array(x)
    return x
def load_data_x(sentences, sequence_length, vocabulary_inv_path):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    vocabulary_inv = json.load(open(vocabulary_inv_path, 'r'))
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    sentences_padded, _ = pad_sentences(sentences, sequence_length=sequence_length)
    x = build_input_data_from_sentences(sentences_padded, vocabulary)
    return x


def load_data(train_pos_txt_path, train_neg_txt_path, test_pos_txt_path, test_neg_txt_path, sequence_length=None):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels, sentences_test, labels_test = load_data_and_labels(train_pos_txt_path, train_neg_txt_path, test_pos_txt_path, test_neg_txt_path)
    sentences_padded, sequence_length = pad_sentences(sentences, sequence_length=sequence_length)
    sentences_test_padded, _ = pad_sentences(sentences_test, sequence_length=sequence_length)

    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    x_test, y_test = build_input_data(sentences_test_padded, labels_test, vocabulary)

    # x = np.array(x)
    # x_test = np.array(x_test)

    print('x', len(x[0]), y.shape)
    print('x_test', x_test.shape, y_test.shape)

    return [x, y, x_test, y_test, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

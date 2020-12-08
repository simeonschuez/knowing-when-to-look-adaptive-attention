import nltk
import pickle
from collections import Counter
import configparser
import string
import os
import sys

utils_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(utils_path)
import data_utils

nltk.download('punkt')

config = configparser.ConfigParser()
config.read(os.path.join(utils_path, 'config.ini'))


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.unknown_token = '<unk>'

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unknown_token]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_word_vocab(df, threshold):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    ids = df.index
    for i, id in enumerate(ids):
        caption = str(df.loc[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    # discard word if word frequency is less than threshold
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def char_vocab():
    """ set up vocabulary wrapper for character level decoding """
    vocab = Vocabulary()
    # add start, stop and padding tokens
    vocab.add_word('^')
    vocab.add_word('$')
    vocab.add_word('&')
    vocab.unknown_token = '&'

    # add digits and lowercase letters
    for char in ' ' + string.digits + string.ascii_lowercase:
        vocab.add_word(char)
    return vocab


def main(threshold):

    vocab_path = os.path.join(os.path.dirname(utils_path), 'data', 'vocab')
    refcoco_path = config['REFCOCO']['data-path']
    coco_path = config['MSCOCO']['data-path']

    # create vocab directory if it doesn't exist
    if not os.path.isdir(vocab_path):
        print('create vocab directory: {}'.format(vocab_path))
        os.makedirs(vocab_path)

    # COCO Captions
    out_path = os.path.join(vocab_path, 'coco_vocab.pkl')
    # retrieve caption df
    captions = preprocess_data.get_coco_captions(coco_path)
    # build vocab
    vocab = build_word_vocab(df=captions, threshold=threshold)
    # save vocab to file
    with open(out_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("COCO Captions:")
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(out_path))

    # RefCOCO Referring Expressions
    out_path = os.path.join(vocab_path, 'refcoco_vocab.pkl')
    # retrieve caption df
    captions = preprocess_data.get_refcoco_captions(refcoco_path)
    # build vocab
    vocab = build_word_vocab(df=captions, threshold=threshold)
    # save vocab to file
    with open(out_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("RefCOCO:")
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(out_path))


if __name__ == '__main__':

    main(
        threshold=4
    )

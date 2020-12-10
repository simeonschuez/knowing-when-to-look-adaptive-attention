import nltk
import pickle
from collections import Counter
import string
import os

from data_utils import get_karpathy_split, get_refcoco_captions

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


def build_vocab(caption_list, threshold):
    """Build a simple vocabulary wrapper."""

    vocab = Vocabulary()

    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word(vocab.unknown_token)

    tokens = []
    for c in caption_list:
        c = c.casefold()
        c = c.translate(str.maketrans('', '', string.punctuation))
        tokens += nltk.word_tokenize(c)

    counter = Counter(tokens)

    words = [w for w, cnt in counter.items() if cnt >= threshold]

    for w in words:
        vocab.add_word(w)

    return vocab


def main(coco_threshold, refcoco_threshold, splits_path, caps_path, refcoco_path, out_dir):

    # create vocab directory if it doesn't exist
    if not os.path.isdir(out_dir):
        print('create vocab directory: {}'.format(out_dir))
        os.makedirs(out_dir)

    print('generate vocab for coco captions')

    caps_df = get_karpathy_split(splits_path=splits_path, caps_path=caps_path)
    train_caps = caps_df.loc[caps_df.split == 'train'].caption.to_list()
    vocab = build_vocab(train_caps, coco_threshold)
    out_path = os.path.join(out_dir, 'coco_vocab.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(vocab, f)
        print('saved vocab with size {} to {}.'.format(len(vocab), out_path))

    print('generate vocab for refcoco')
    reg_df = get_refcoco_captions(refcoco_path)
    train_caps = reg_df.loc[reg_df.split == 'train'].caption.to_list()
    vocab = build_vocab(train_caps, refcoco_threshold)
    out_path = os.path.join(out_dir, 'refcoco_vocab.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(vocab, f)
        print('saved vocab with size {} to {}.'.format(len(vocab), out_path))


if __name__ == '__main__':

    main(
        coco_threshold=5,
        refcoco_threshold=3,
        splits_path='/home/vu48pok/.data/compling/data/corpora/external/MSCOCO/COCO/splits/karpathy/caption_datasets/',
        caps_path='/home/vu48pok/.data/compling/data/corpora/external/MSCOCO/COCO/',
        refcoco_path='/home/vu48pok/.data/compling/data/corpora/external/COCO_ReferIt/licheng/refcoco/',
        out_dir='/home/vu48pok/Dokumente/Projekte/reg/knowing-when-to-look-adaptive-attention/data/'
    )

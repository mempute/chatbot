"""
Author: Lilian Weng (lilian.wengweng@gmail.com)
        http://lilianweng.github.io/lil-log
        Oct 2018
"""

import os
import sys
import urllib.parse
import numpy as np
import random

# IDs of special characters.
PAD_ID = 0
UNKNOWN_ID = 1
START_ID = 2
END_ID = 3


class DatasetManager:
    """
    Download data files and prepare the train and test data.

    """
    dataset_config_dict = {
        'iwslt15': {
            'source_lang': 'en',
            'target_lang': 'vi',
            'url': "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/",
            'files': ['train.en', 'train.vi',
                      'tst2012.en', 'tst2012.vi',
                      'tst2013.en', 'tst2013.vi',
                      'vocab.en', 'vocab.vi'],
            'train': 'train',
            'test': ['tst2012', 'tst2013'],
            'vocab': 'vocab',
        },
        'chatbot': {
            'source_lang': 'q',
            'target_lang': 'a',
            'url': "",
            'files': ['train.q', 'train.a',
                      'test.q', 'test.a',
                      'tst2013.en', 'tst2013.vi',
                      'vocab', 'vocab'],
            'train': 'train',
            'test': ['test'],
            'vocab': 'vocab',
        },
        'chatbot_16': {
            'source_lang': 'q',
            'target_lang': 'a',
            'url': "",
            'files': ['train.q', 'train.a',
                      'test.q', 'test.a',
                      'tst2013.en', 'tst2013.vi',
                      'vocab', 'vocab'],
            'train': 'train',
            'test': ['test'],
            'vocab': 'vocab',
        },
        'chatbot_32': {
            'source_lang': 'q',
            'target_lang': 'a',
            'url': "",
            'files': ['train.q', 'train.a',
                      'test.q', 'test.a',
                      'tst2013.en', 'tst2013.vi',
                      'vocab', 'vocab'],
            'train': 'train',
            'test': ['test'],
            'vocab': 'vocab',
        },
        'kor_eng': {
            'source_lang': 'q',
            'target_lang': 'a',
            'url': "",
            'files': ['train.q', 'train.a',
                      'test.q', 'test.a',
                      'tst2013.en', 'tst2013.vi',
                      'vocab', 'vocab'],
            'train': 'train',
            'test': ['test'],
            'vocab': 'vocab',
        },
        'wmt14': {
            'source_lang': 'en',
            'target_lang': 'de',
            'url': "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/",
            'files': ['train.en', 'train.de', 'train.align',
                      'newstest2012.en', 'newstest2012.de',
                      'newstest2013.en', 'newstest2013.de',
                      'newstest2014.en', 'newstest2014.de',
                      'newstest2015.en', 'newstest2015.de',
                      'vocab.50K.en', 'vocab.50K.de', 'dict.en-de'],
            'train': 'train',
            'test': ['newstest2012', 'newstest2013', 'newstest2014', 'newstest2015'],
            'vocab': 'vocab.50K',
        },
        'wmt15': {
            'source_lang': 'en',
            'target_lang': 'cs',
            'url': "https://nlp.stanford.edu/projects/nmt/data/wmt15.en-cs/",
            'files': ['train.en', 'train.cs',
                      'newstest2013.en', 'newstest2013.cs',
                      'newstest2014.en', 'newstest2014.cs',
                      'newstest2015.en', 'newstest2015.cs',
                      'vocab.1K.en', 'vocab.1K.cs',
                      'vocab.10K.en', 'vocab.10K.cs',
                      'vocab.20K.en', 'vocab.20K.cs',
                      'vocab.50K.en', 'vocab.50K.cs'],
            'train': 'train',
            'test': ['newstest2013', 'newstest2014', 'newstest2015'],
            'vocab': 'vocab.50K',

        }
    }

    def __init__(self, name, base_data_dir='./chatbot_data/'):
        assert name in self.dataset_config_dict

        self.name = name
        self.config = self.dataset_config_dict[name]
        self.source_lang = self.config['source_lang']
        self.target_lang = self.config['target_lang']

        self.data_dir = os.path.join(base_data_dir, name)
        os.makedirs(self.data_dir, exist_ok=True)

        self.source_word2id = None
        self.source_id2word = None
        self.target_word2id = None
        self.target_id2word = None
        self.line_count = 0
    def _download_data_from_url(self, download_url):
        filename = download_url.split('/')[-1]
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            # If the file does not exist, download it.
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                    filename, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(download_url, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

        return filepath

    def maybe_download_data_files(self):
        """Download and extract the file from Stanford NLP website.
        """
        for filename in self.config['files']:
            self._download_data_from_url(urllib.parse.urljoin(self.config['url'], filename))

        print("Downloaded Files:", os.listdir(self.data_dir))

    def _load_vocab_file(self, filename):
        # The first three words in both vocab files are special characters:
        # <unk>: unknown word.
        # <s>: start of a sentence.
        # </s>: # end of a sentence.
        # In addition, we add <pad> as a place holder for a padding space.
        vocab_file = os.path.join(self.data_dir, filename)
        words = list(map(lambda w: w.strip().lower(), open(vocab_file, encoding='utf-8')))
        #fp = open(vocab_file, encoding='utf-8')
        #words = []
        #for w in fp:
        #   words.append(w.strip().lower())

        #words.insert(0, '<pad>')
        #words = words[:4] + list(set(words[4:]))  # Keep the special characters on top.
        
        word2id = {word: i for i, word in enumerate(words)}
        id2word = words
        #print(id2word)
        #print(word2id)

        assert id2word[PAD_ID] == '<pad>'
        assert id2word[UNKNOWN_ID] == '<unk>'
        assert id2word[START_ID] == '<s>'
        assert id2word[END_ID] == '</s>'
        #print(word2id)
        return word2id, id2word

    def load_vocab(self):
        prefix = self.config['vocab']
        if self.name == 'chatbot' or self.name == 'chatbot_16' or self.name == 'chatbot_32' or self.name == 'kor_eng':
            self.source_word2id, self.source_id2word = self._load_vocab_file(prefix)
            self.target_word2id = self.source_word2id
            self.target_id2word = self.source_id2word
            print(f"vocabulary size:", len(self.source_word2id))
        else:
            self.source_word2id, self.source_id2word = self._load_vocab_file(
                prefix + '.' + self.source_lang)
            self.target_word2id, self.target_id2word = self._load_vocab_file(
                prefix + '.' + self.target_lang)

            print(f"'{self.source_lang}' vocabulary size:", len(self.source_word2id))
            print(f"'{self.target_lang}' vocabulary size:", len(self.target_word2id))

    def _sentence_pair_iterator(self, file1, file2, seq_len):
        """
        The sentence is discarded if it is longer than `seq_len`; otherwise we pad it with
        '<pad>' to make it to have the exact length `seq_len`.

        Args:
            file1 (str): training data in source language.
            file2 (str): training data in target language. Lines should match lines in `file1`.
            seq_len (int): max sequence length.

        Returns: a tuple of (a list of word id for language 1,
                             a list of word id for language 2)
        """

        def line_count(filename):
            return int(os.popen(f'wc -l {filename}').read().strip().split()[0])

        def parse_line(line, word2id):
            line = line.strip().lower().split()
            word_ids = [word2id.get(w, UNKNOWN_ID) for w in line]
            # If the sentence is not long enough, extend with '<pad>' symbols.
            #word_ids = [START_ID] + word_ids + [END_ID] 
            word_ids = word_ids + [END_ID] #auto regression 실행시 go토큰으로 START_ID을 사용
            leng = len(word_ids)
            if leng <= self.underline: return word_ids #최소 제한 길이보다 작은 것은 스킵되게 한다.
            word_ids += [PAD_ID] * max(0, seq_len - len(word_ids))
            #word_ids = [START_ID] + word_ids + [END_ID] + [PAD_ID]
            #word_ids += [UNKNOWN_ID] * max(0, seq_len - len(word_ids))
 
            return word_ids

        #print(f"Num. lines in '{file1}': {line_count(file1)}")
        #assert line_count(file1) == line_count(file2)
        line_pairs = list(zip(open(file1, encoding='utf-8'), open(file2, encoding='utf-8')))
        random.shuffle(line_pairs)

        self.line_count = 0
        for l1, l2 in line_pairs:
            sent1 = parse_line(l1, self.source_word2id)
            sent2 = parse_line(l2, self.target_word2id)
            if len(sent1) == len(sent2) == seq_len:
                #if len(sent1) < seq_len:
                #    print('11111')
                #if len(sent2) < seq_len:
                #    print('22222222')
                self.line_count += 1
                yield sent1, sent2

    def data_generator(self, batch_size, seq_len, data_type='train', file_prefix=None, epoch=None, under_line=None):
        """
        A generator yields a pair of two sentences, (source, target).
        Each sentence is a list of word ids. Sentences with more than `seq_len` words are
        discarded. Shorter ones are padded with <pad> symbol at the end to have exact
        `seq_len` words.

        Args:
            batch_size (int): size of one mini-batch.
            seq_len (int): desired sentence length.
            data_type (str): 'train' or 'test'
            file_prefix (str)
            epoch (int): if None, repeat the dataset infinitely.

        Returns:
            yields a pair of word ids.
        """
        if under_line is None: self.underline = 0 
        else: self.underline = under_line

        assert data_type in ('train', 'test')
        # Load vocabulary
        if self.source_id2word is None:
            self.load_vocab()

        # Use the expected set of files
        if file_prefix is None:
            prefixes = self.config[data_type]
            if not isinstance(prefixes, list):
                prefixes = [prefixes]
        else:
            prefixes = [file_prefix]

        batch_src, batch_tgt = [], []
        ep = 0
        while epoch is None or ep < epoch:
            for prefix in prefixes:
                for ids_src, ids_tgt in self._sentence_pair_iterator(
                        os.path.join(self.data_dir, prefix + '.' + self.source_lang),
                        os.path.join(self.data_dir, prefix + '.' + self.target_lang),
                        seq_len
                ):
                    batch_src.append(ids_src)
                    batch_tgt.append(ids_tgt)

                    if len(batch_src) == batch_size:
                        a = np.array(batch_src)
                        a = np.expand_dims(a, axis=2)
                        b = np.array(batch_tgt)
                        b = np.expand_dims(b, axis=2)
                        yield a.copy(), b.copy(), ep
                        batch_src, batch_tgt = [], []

            ep += 1

        # leftover
        #if len(batch_src) > 0:
        #    yield np.array(batch_src).copy(), np.array(batch_tgt).copy()


def recover_sentence(sent_ids, id2word):
    """Convert a list of word ids back to a sentence string.
    """
    words = list(map(lambda i: id2word[i] if 0 <= i < len(id2word) else '<unk>', sent_ids))

    # Then remove tailing <pad>
    i = len(words) - 1
    while i >= 0 and words[i] == '<pad>':
        i -= 1
    words = words[:i + 1]
    return ' '.join(words)

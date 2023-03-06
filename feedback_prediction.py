import pandas as pd
import numpy as np
import string
import os
import pickle
from random import randrange
from datetime import datetime
import time
import multiprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import nlpaug.augmenter.word as naw
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.corpora.wikicorpus import WikiCorpus
from spellchecker import SpellChecker
from xgboost import XGBClassifier
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import torch
import torch.nn as nn
import transformers
from transformers import AutoModel, BertTokenizerFast
from sklearn.model_selection import StratifiedKFold
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from scipy import sparse
import joblib
from tqdm import tqdm
nltk.download('omw-1.4')


def set_dir_get_path(dir_parent, dir_child):
    p_new_dir = dir_parent + '/' + dir_child
    if not os.path.exists(p_new_dir):
        os.makedirs(p_new_dir)
    return p_new_dir


def get_missing_data(df):
    for curr_col in df.columns.tolist():
        i_null = df[curr_col].isnull().sum()
        i_full = df.shape[0]
        i_exist = i_full - i_null
        f_perc = (i_exist / i_full) * 100
        f_missing = (1 - (i_exist / i_full)) * 100
        print('%s, Data Exists: %.2f%%, Missing: %.2f%%' % (curr_col, f_perc, f_missing))


def get_ratio(df):
    data_size = int(df.shape[0])
    y_unique = df['discourse_effectiveness'].unique()
    l_targets = list()
    for label in y_unique:
        curr_label = df[df['discourse_effectiveness'] == label]
        l_targets.append(label)
        label_size = curr_label.shape[0]
        print(str(label) + 'count: %d, ratio: %.2f%%' % (label_size, (label_size/data_size)*100))
    unique_discourse = df_train['discourse_id'].unique().size
    unique_essay = df_train['essay_id'].unique().size
    unique_type = df_train['discourse_type'].unique().size
    print(f'train size: {int(df_train.shape[0])}')
    print(
        f'unique discourse: {int(unique_discourse)} \n unique essay: {int(unique_essay)} \n unique type: {int(unique_type)}')


def spelling_apply(value, m_spell):
    s_spell = value
    if s_spell != 'False' and s_spell != '':
        l_sentence = s_spell.split()
        s_spell = ''
        for term in l_sentence:
            correct = m_spell.correction(term)
            candidate = m_spell.candidates(term)
            s_spell += correct + ' '
    s_spell = s_spell.lower()
    return s_spell


def spell_checker(curr_df, col_text):
    _distance = 1
    m_spell = SpellChecker(distance=_distance)
    srs_text = curr_df[col_text]
    tqdm.pandas()
    srs_text = srs_text.progress_apply(spelling_apply, args=(m_spell,))
    df_spell = srs_text.to_frame(col_text)
    return df_spell


def normalize_text_series(curr_df, col_text):
    df = curr_df.copy()
    df[col_text] = df[col_text].str.replace('[{}]'.format(string.punctuation), '', regex=True)  # v1
    # df[col_text] = df[col_text].str.replace('[^\w\s]', '')  # v2
    df[col_text] = df[col_text].str.lower()
    # df[col_text] = spell_checker(df, col_text)  # check spelling
    # df[col_text] = ''.join(filter(str.isalnum, df[col_text]))  # remove non alphabets
    # df[col_text] = ''.join([i for i in s if i.isalpha()])
    return df


def normalize_text_string(s_text):
    d_punc = {
              "\"": None, '\"': None, ',': None, '"': None, '|': None, '?': None,
              '-': None, '_': None, '*': None, '`': None, '/': None, '@': None,
              ';': None, "'": None, '[': None, ']': None, '(': None, ')': None,
              '{': None, '}': None, '<': None, '>': None, '~': None, '^': None,
              '&': None, '!': None, '=': None, '+': None, '#': None, '$': None,
              '%': None, ':': None, '.': None
              }
    s_text = s_text.translate(str.maketrans(d_punc))
    s_text = s_text.lower()
    s_text = s_text.strip()
    return s_text


def normalize_text_params(s_text):
    d_punc = {
              "\"": None, '\"': None, ',': None, '"': None, '|': None, '-': None, '`': None, '/': None, ';': None,
              "'": None, '[': None, ']': None, '(': None, ')': None, '{': None, '}': None, ':': None,
              }
    s_text = s_text.translate(str.maketrans(d_punc))
    s_text = s_text.lower()
    s_text = s_text.strip()
    return s_text


def df_to_string(df, col_text):
    srs_text = df[col_text].copy()
    df_text = pd.DataFrame({col_text: [' '.join(srs_text)]})
    s_text = df_text.at[0, col_text]
    return s_text


def generate_glove_emb(p_glove_input, p_glove_emb):
    d_glove_emb = dict()
    with open(p_glove_input, 'r') as f:
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], 'float32')
            d_glove_emb[word] = vectors
    f.close()
    with open(p_glove_emb, 'wb') as output:
        pickle.dump(d_glove_emb, output, pickle.HIGHEST_PROTOCOL)
    return d_glove_emb


def generate_word_index(corpus, p_glove_emb, d_glove_emb):
    # function create arrays of token indices for glove
    MAX_LEN = 50
    tokenizer_obj = Tokenizer()
    tokenizer_obj.fit_on_texts(corpus)
    sequences = tokenizer_obj.texts_to_sequences(corpus)
    pad = pad_sequences(sequences, maxlen=MAX_LEN, truncating='post', padding='post')
    word_index = tokenizer_obj.word_index  # <word, word_index>
    print('Number of unique words:', len(word_index))
    with open(p_glove_emb, 'wb') as output:
        pickle.dump(d_glove_emb, output, pickle.HIGHEST_PROTOCOL)
    return word_index


def append_list(sim_words, words):
    l_terms = list()
    for i in range(len(sim_words)):
        sim_words_list = list(sim_words[i])
        sim_words_list.append(words)
        sim_words_tuple = tuple(sim_words_list)
        l_terms.append(sim_words_tuple)
    return l_terms


def glove_apply(value, m_glove):
    i_modulo = 3
    s_glove = value
    if s_glove != 'False' and s_glove != '':
        l_glove = s_glove.split()
        s_glove = ''
        for i_term in range(len(l_glove)):  # v1
            term = l_glove[i_term]
            if i_term % i_modulo == 0:
                l_cands = m_glove.most_similar(value, topn=1)
                cand = l_cands[0]
                s_glove += cand + ' '
            else:
                s_glove += term + ' '
    s_glove = s_glove.lower()
    return s_glove


def aug_glove(srs_text, p_resource, p_glove_train):
    p_glove_input = p_resource + '/glove.6B.100d.txt'
    p_glove_emb = p_resource + '/d_glove_emb.pkl'
    p_glove_word_index = p_resource + '/d_word_index.pkl'
    p_m_glove = p_resource + '/m_glove.sav'

    if not os.exists(p_glove_emb):
        d_glove_emb = generate_glove_emb(p_glove_input, p_glove_emb)
    else:
        with open(p_glove_emb, 'rb') as curr_input:
            d_glove_emb = pickle.load(curr_input)

    if not os.exists(p_glove_word_index):
        d_word_index = generate_word_index(p_glove_word_index, p_glove_emb, d_glove_emb)
    else:
        with open(p_glove_word_index, 'rb') as curr_input:
            d_word_index = pickle.load(curr_input)

    num_words = len(d_word_index) + 1  # assigns glove vectors for the words obtained from tokenization
    embedding_matrix = np.zeros((num_words, 100))

    for word, i in tqdm(d_word_index.items()):
        if i > num_words:
            continue

        emb_vec = d_glove_emb.get(word)
        if emb_vec is not None:
            embedding_matrix[i] = emb_vec

    if not os.path.exists(p_m_glove):
        print('Glove model loading.')
        m_glove = KeyedVectors.load_word2vec_format(p_glove_input, binary=False, no_header=True)
        pickle.dump(m_glove, open(p_m_glove, 'wb'))


def get_synonym(term, m_syn):
    try:
        l_candidates = wordnet.synset(term + '.n.01')
        chosen_term = term
        for candidate in l_candidates._lemma_names:
            if candidate not in term:
                chosen_term = candidate
                return chosen_term
    except Exception as word_net_error:
        chosen_term = term
    return chosen_term


def synonym_apply(value, m_syn):
    # i_modulo = 5
    i_modulo = 3
    if isinstance(value, pd.Series) and value.shape[0] == 1:
        value = value.values[0]
    s_syn = value
    if s_syn != 'False' and s_syn != '':
        l_sequence = s_syn.split()
        s_syn = ''
        for i_term in range(len(l_sequence)):
            term = l_sequence[i_term]
            if i_term % i_modulo == 0:
                new_term = get_synonym(term, m_syn)
                s_syn += new_term + ' '
            else:
                s_syn += term + ' '
    s_syn = normalize_text_string(s_syn)
    return s_syn


def aug_synonym(srs_text, p_output, p_corpus, b_chunks=True):
    m_syn = naw.SynonymAug(aug_src='wordnet')
    if not b_chunks:
        tqdm.pandas()
        srs_text = srs_text.progress_apply(synonym_apply, args=(m_syn,))
        df_syn = srs_text.to_frame(col_text)
        # df_syn.to_csv(path_or_buf=p_output, mode='w', index=False, na_rep='', header=True, encoding='utf-8-sig')
    else:
        df_syn = pd.DataFrame(srs_text, columns=[col_text])
        df_syn.to_csv(path_or_buf=p_output, mode='w', index=False, na_rep='', header=True, encoding='utf-8-sig')
        j = 0
        rounds = get_rounds(p_corpus)  # 74
        with open(p_corpus) as read_chunk:
            chunk_iter = pd.read_csv(read_chunk, chunksize=500)
            tqdm.pandas()
            for df_curr_chunk in tqdm(chunk_iter):
                if not df_curr_chunk.empty:
                    curr_srs_txt = df_curr_chunk[col_text]
                    srs_syn = curr_srs_txt.progress_apply(synonym_apply, args=(m_syn,))
                    df_syn = srs_syn.to_frame(col_text)
                    df_syn.to_csv(path_or_buf=p_output, mode='a', index=False, na_rep='', header=False,
                                  encoding='utf-8-sig')
                j += 1
                # if j % 10000 == 0:
                #     init()
                print(f'Finished Part: {j} / {rounds}')
        print('Finished Augmentation: Synonym.')
    return df_syn


def word2vec_sentence_apply(value, m_w2v):
    if isinstance(value, pd.Series):
        value = value.values[0]
    s_w2v = value
    if s_w2v != 'False' and s_w2v != '':
        s_w2v = m_w2v.augment(s_w2v)
    s_w2v = normalize_text_string(s_w2v)
    return s_w2v


def word2vec_word_apply(value, m_w2v):
    s_w2v = value
    i_modulo = 3
    # i_modulo = 5
    i_top = 10
    if value != 'False' and value != '':
        s_w2v = ''
        l_w2v_terms = value.split(' ')
        for i_term in range(len(l_w2v_terms)):  # key vectors
            term = l_w2v_terms[i_term]
            if i_term % i_modulo == 0 and term != '':
                try:
                    l_preds = m_w2v.most_similar(term, topn=i_top)
                    if len(l_preds) >= 10:
                        i_random = randrange(i_top)
                    else:
                        i_random = randrange(len(l_preds))
                    curr_tuple = l_preds[i_random]
                    chosen_candidate = curr_tuple[0]
                except Exception:  # KeyError
                    chosen_candidate = term
                s_w2v += chosen_candidate + ' '
            else:
                s_w2v += term + ' '
    s_w2v = normalize_text_string(s_w2v)
    return s_w2v


def get_rounds(p_corpus):
    df_original = pd.read_csv(p_corpus)
    length = df_original.shape[0]
    rounds = (length//500)+1
    print(f'Number of rounds: {rounds}.')
    del df_original
    return rounds


def train_word2vec(p_emb, p_resource):
    p_model = p_resource + '/m_w2v.model'
    if not os.path.exists(p_model):
        wiki = WikiCorpus(p_emb, lemmatize=False, dictionary={})
        sentences = list(wiki.get_texts())
        params = {'size': 300,
                  'window': 10,
                  'min_count': 10,
                  'workers': max(1, multiprocessing.cpu_count() - 1),
                  'sample': 1E-3, }
        m_w2v = Word2Vec(sentences, **params)
        m_w2v.save(p_model)
        # m_w2v = KeyedVectors.save_word2vec_format(fname=p_model)
    else:
        m_w2v = Word2Vec.load(p_model)
        # m_w2v = KeyedVectors.load_word2vec_format(p_model, binary=True)
    return m_w2v


def init():
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        print(
            '\n\nThis error most likely means that this notebook is not '
            'configured to use a GPU.  Change this in Notebook Settings via the '
            'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
        raise SystemError('GPU device not found')

    with tf.device('/device:GPU:0'):
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        random_image_gpu = tf.random.normal((100, 100, 100, 3))
        net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
        return tf.math.reduce_sum(net_gpu)


def aug_w2v(srs_text, p_emb, p_output, p_corpus, b_chunks=True):
    df_w2v = None
    if 'bert' in p_emb:
        m_w2v = naw.ContextualWordEmbsAug(model_path=p_emb)
    else:
        # m_w2v = train_word2vec(p_emb, p_resource)
        # m_w2v = naw.WordEmbsAug(model_type='fasttext', model_path=p_emb)
        # m_w2v = naw.WordEmbsAug(model_type='word2vec', model_path=p_emb)
        m_w2v = KeyedVectors.load_word2vec_format(p_emb, binary=False, unicode_errors='ignore', encoding='utf8')

    if not b_chunks:
        tqdm.pandas()
        if 'bert' in p_emb:
            srs_text = srs_text.progress_apply(word2vec_sentence_apply, args=(m_w2v,))  # loads by model
        else:
            srs_text = srs_text.progress_apply(word2vec_word_apply, args=(m_w2v,))  # loads by key vectors
        df_w2v = srs_text.to_frame(col_text)
        # df_w2v.to_csv(path_or_buf=p_output, mode='w', index=False, na_rep='', header=True, encoding='utf-8-sig')
    else:
        j = 0
        rounds = get_rounds(p_corpus)  # 74
        with open(p_corpus) as read_chunk:
            chunk_iter = pd.read_csv(read_chunk, chunksize=500)
            tqdm.pandas()
            for df_curr_chunk in tqdm(chunk_iter):
                if not df_curr_chunk.empty:
                    curr_srs_txt = df_curr_chunk[col_text]
                    srs_w2v = curr_srs_txt.progress_apply(word2vec_word_apply, args=(m_w2v,))
                    df_w2v = srs_w2v.to_frame(col_text)
                    df_w2v.to_csv(path_or_buf=p_output, mode='a', index=False, na_rep='', header=False,
                                  encoding='utf-8-sig')
                j += 1
                # if j % 10000 == 0:
                #     init()
                print(f'Finished Part: {j} / {rounds}')
        print('Finished Augmentation: w2v.')
    return df_w2v
      
      
def set_pickle(d, path, filename):
    p_write = path + '/' + filename + '.pkl'
    if '\\' in p_write:
        p_write = p_write.replace('\\', '/')
    if '//' in p_write:
        p_write = p_write.replace('//', '/')
    with open(p_write, 'wb') as output:
        pickle.dump(d, output, pickle.HIGHEST_PROTOCOL)


def get_pickle(path, filename):
    p_read = path + '/' + filename + '.pkl'
    if '\\' in p_read:
        p_read = p_read.replace('\\', '/')
    if '//' in p_read:
        p_read = p_read.replace('//', '/')
    with open(p_read, 'rb') as curr_input:
        return pickle.load(curr_input)


def read_csv_sample(p_csv, shape):
    df = pd.read_csv(p_csv, chunksize=shape)
    for chunk in df:
        if len(chunk.columns.tolist()) > 3:
            return chunk.iloc[:, :shape]
        else:
            return chunk


def get_filename(path):
    l_substring = path.split('/')
    l_subtitle = l_substring[len(l_substring) - 1].split('.')
    return l_subtitle[0]


def load_params(p_output):
    df_params = pd.read_csv(p_output)
    s_params = df_params.iloc[1, 0]
    l_params = s_params.split(',')
    d_params = dict()
    # d_params = get_pickle(p_project, filename)
    for curr_param in l_params:
        key = curr_param.split(':')[0]
        s_param = normalize_text_params(key)
        value = curr_param.split(':')[1]
        value = normalize_text_params(value)
        f_value = round(float(value), 3)
        if f_value > 1:
            d_params[s_param] = int(f_value)
        else:
            d_params[s_param] = f_value
    return d_params


def optimize_params(x, y, curr_label, p_opt):
    iterations = 3
    filename = '/' + 'opt_'+curr_label + '.csv'
    p_output = p_opt + filename
    if not os.path.exists(p_output):
        print('Optimizing label: ' + curr_label)
        df_x = x.copy()
        col_text = 'discourse_text'
        l_columns = df_x.columns.tolist()
        l_columns.remove(col_text)
        l_columns.remove(col_id)

        df_text = pd.DataFrame(df_x[col_text], columns=[col_text])
        df_text = normalize_text_series(df_text, col_text)
        srs_text = df_text[col_text]
        df_general = pd.DataFrame(df_x, columns=l_columns)

        df_general = onehot_encoder(df_general)
        df_y = onehot_encoder(y)
        df_y_curr = pd.DataFrame(df_y[curr_label], columns=[curr_label])
        df_y_curr.reset_index(drop=True, inplace=True)

        _max_features = 10000
        _stopwords = list(stopwords.words('english'))

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=_stopwords,
                                     max_features=_max_features, analyzer='word',
                                     encoding='utf-8', decode_error='strict',
                                     lowercase=True, norm='l2', smooth_idf=True,
                                     sublinear_tf=False)

        x_text = vectorizer.fit_transform(srs_text)
        feature_names = vectorizer.get_feature_names_out()
        df_text = pd.DataFrame(x_text.toarray(), columns=feature_names)

        df_x = combine_features(df_text, df_general)

        tqdm.pandas()
        optimize_model(df_x, df_y_curr, p_output, iterations)
        print('Finished optimizing label: ' + curr_label)
        # init()
    print('Loading pretrained params for label: ' + curr_label)
    return load_params(p_output)


def optimize_cv(n_estimators, max_depth, eta, x, y):
    _cv = 3
    # _cv = 5
    o_model = XGBClassifier(max_depth=max_depth, eta=eta, n_estimators=n_estimators,
                            random_state=5, verbosity=0, use_label_encoder=False)
    cval = cross_val_score(o_model, x, y, scoring='roc_auc', cv=_cv)
    return cval.mean()


def optimize_model(x, y, p_output, iterations):

    def crossval(n_estimators, max_depth, eta):

        return optimize_cv(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            eta=float(eta),
            x=x,
            y=y,
        )

    optimizer = BayesianOptimization(
        f=crossval,
        pbounds={
            "n_estimators": (100, 500),
            "max_depth": (2, 12),
            "eta": (0.00001, 0.3),
        },
        random_state=9,
        verbose=2
    )
    optimizer.maximize(n_iter=iterations)  # -np.squeeze(res.fun) instead of -res.fun[0] (scipy==1.7.0)
    d_results = optimizer.max
    print("Final result:", optimizer.max)
    # d_params = d_results['params']
    # filename = get_filename(p_output)
    # set_pickle(d_params, p_project, filename)
    # df_results = pd.DataFrame.from_dict(d_params, orient='index')
    df_results = pd.DataFrame.from_dict(d_results, orient='index')
    df_results.to_csv(path_or_buf=p_output, mode='w', index=False, na_rep='', header=True, encoding='utf-8-sig')


def load_model(p_model, curr_label):
    p_load = p_model + '/' + curr_label
    p_load_joblib = p_load + '.' + 'pkl'
    o_model = joblib.load(p_load_joblib)
    filename = get_filename(p_load)
    print(f'Model {filename} has been loaded.')
    return o_model


def preprocess(df_curr_train, df_curr_test=None):
    col_text = 'discourse_text'
    df_train = df_curr_train.copy()

    l_columns = df_train.columns.tolist()
    l_columns.remove(col_text)
    l_columns.remove(col_id)

    df_text_train = pd.DataFrame(df_train[col_text], columns=[col_text])
    df_text_train = normalize_text_series(df_text_train, col_text)
    df_general_train = pd.DataFrame(df_train, columns=l_columns)
    df_general_train.reset_index(drop=True, inplace=True)

    if df_curr_test is not None:
        df_test = df_curr_test.copy()
        df_text_test = pd.DataFrame(df_test[col_text], columns=[col_text])
        df_text_test = normalize_text_series(df_text_test, col_text)
        df_general_test = pd.DataFrame(df_test, columns=l_columns)
        df_general_test.reset_index(drop=True, inplace=True)
        return df_text_train, df_general_train, df_text_test, df_general_test
    else:
        return df_text_train, df_general_train


def set_column_names(l_org_cols, ohe):
    train_feature_names = ohe.get_feature_names_out(l_org_cols)
    # train_feature_names = ohe.feature_names_in_.tolist()
    for i in range(len(train_feature_names)):
        feature = train_feature_names[i]
        if '_' in feature:
            i_del = feature.rfind('_')
            train_feature_names[i] = feature[i_del+1:]
    return train_feature_names


def onehot_encoder_infer(df_curr, p_onehot, curr_label):
    df_test_ohe = df_curr.copy()
    l_org_cols = df_test_ohe.columns.tolist()

    s_dict = 'd_onehots_'+curr_label
    d_onehots = get_pickle(p_onehot, s_dict)
    l_train_ohe_cols = d_onehots['l_original']

    for curr_col, o_ohe in d_onehots.items():
        if curr_col == 'l_original':
            continue
        if curr_col in df_test_ohe.columns.tolist():  # (1) one hot encoding test features that are in train
            test_col_ohe = o_ohe.transform(df_test_ohe[curr_col].values.reshape(-1, 1))
            l_test_ohe_cols = set_column_names([curr_col], o_ohe)
            df_col_test_ohe = pd.DataFrame(test_col_ohe.toarray(), columns=l_test_ohe_cols)
            df_test_ohe.drop(curr_col, axis=1, inplace=True)
            df_test_ohe = df_test_ohe.merge(df_col_test_ohe, left_index=True, right_index=True)
        else:  # (2) adding dummy category features from train that werent in test
            for new_train_ohe_col in l_train_ohe_cols:
                df_test_ohe[new_train_ohe_col] = 0

    for curr_col in df_test_ohe.columns.tolist():  # removes test categorical features that are not in train
        if df_test_ohe[curr_col].dtype == object or '_id' in curr_col and curr_col not in l_train_ohe_cols:
            df_test_ohe.drop(curr_col, axis=1, inplace=True)

    return df_test_ohe


def onehot_encoder(df_curr, df_test_curr=None):
    df_test_ohe = None
    df_train_ohe = df_curr.copy()
    if df_test_curr is not None:
        df_test_ohe = df_test_curr.copy()
    l_org_train_cols = df_train_ohe.columns.tolist()
    d_onehots = dict()
    d_onehots['l_original'] = l_org_train_cols

    for curr_col in df_train_ohe.columns.tolist():
        if df_train_ohe[curr_col].dtype == object or '_id' in curr_col:
            ohe = OneHotEncoder(handle_unknown='ignore')
            train_col_ohe = ohe.fit_transform(df_train_ohe[curr_col].values.reshape(-1, 1))
            l_train_ohe_cols = set_column_names([curr_col], ohe)
            df_col_train_ohe = pd.DataFrame(train_col_ohe.toarray(), columns=l_train_ohe_cols)
            df_train_ohe.drop(curr_col, axis=1, inplace=True)
            df_train_ohe = df_train_ohe.merge(df_col_train_ohe, left_index=True, right_index=True)
            d_onehots[curr_col] = ohe

            if df_test_ohe is not None:  # (1) one hot encoding test features that are in train
                if curr_col in df_test_ohe.columns.tolist():
                    test_col_ohe = ohe.transform(df_test_ohe[curr_col].values.reshape(-1, 1))
                    l_test_ohe_cols = set_column_names([curr_col], ohe)
                    df_col_test_ohe = pd.DataFrame(test_col_ohe.toarray(), columns=l_test_ohe_cols)
                    df_test_ohe.drop(curr_col, axis=1, inplace=True)
                    df_test_ohe = df_test_ohe.merge(df_col_test_ohe, left_index=True, right_index=True)
                else:  # (2) adding dummy category features from train that werent in test
                    for new_train_ohe_col in l_train_ohe_cols:
                        df_test_ohe[new_train_ohe_col] = 0

    if df_test_ohe is not None:
        for curr_col in df_test_ohe.columns.tolist():  # removes test categorical features that are not in train
            if df_test_ohe[curr_col].dtype == object or '_id' in curr_col and curr_col not in l_org_train_cols:
                df_test_ohe.drop(curr_col, axis=1, inplace=True)

    if df_test_ohe is not None:
        return df_train_ohe, df_test_ohe, d_onehots
    else:
        return df_train_ohe


def apply_tfidf(vectorizer, df_curr_text_test, df_curr_text_train=None):
    if df_curr_text_train is not None:
        srs_text_train = df_curr_text_train[col_text].copy()
        x_text_train = vectorizer.fit_transform(srs_text_train)
        train_feature_names = vectorizer.get_feature_names_out()
        df_text_train = pd.DataFrame(x_text_train.toarray(), columns=train_feature_names)

    srs_text_test = df_curr_text_test[col_text].copy()
    x_text_test = vectorizer.transform(srs_text_test)
    test_feature_names = vectorizer.get_feature_names_out()
    df_text_test = pd.DataFrame(x_text_test.toarray(), columns=test_feature_names)

    if df_curr_text_train is not None:
        return df_text_train, df_text_test
    else:
        return df_text_test


def combine_features(curr_text, curr_general):
    text = curr_text.copy()
    general = curr_general.copy()
    general.reset_index(drop=True, inplace=True)
    df = general.merge(text, left_index=True, right_index=True)
    return df


def evaluate_model(y_test, y_preds, y_probs, label, s_model, i_fold, i_train, i_val):
    acc_test = round(accuracy_score(y_test, y_preds), 3)
    precision = round(precision_score(y_test, y_preds), 3)
    recall = round(recall_score(y_test, y_preds), 3)
    f1 = round(f1_score(y_test, y_preds), 3)
    fpr, tpr, threshold = roc_curve(y_test, y_preds)
    auc_score = round(roc_auc_score(y_test, y_probs), 3)
    arr_precision, arr_recall, threshold_pr_auc = precision_recall_curve(y_test, y_probs)
    pr_auc_score = round(auc(arr_recall, arr_precision), 3)
    print(f'Label: {label}, AUC: {auc_score}, precision: {precision}, recall: {recall}, f1: {f1}, PRAUC: {pr_auc_score}')
    return {'Model': s_model, 'Class': label, 'Fold': i_fold, 'AUC': auc_score, 'PRAUC': pr_auc_score,
            'Precision': precision, 'Recall': recall, 'F1': f1, 'Train': i_train, 'Test': i_val}


def init_submit(p_test):
    l_cols_submit = ['discourse_id', 'Ineffective', 'Adequate', 'Effective']
    df_submit = pd.DataFrame(columns=l_cols_submit)
    p_submit = '/home/edoli/PycharmProjects/Feedback-Prediction/' + 'submission' + '.csv'
    df_test_original = pd.read_csv(p_test)
    df_submit['discourse_id'] = df_test_original['discourse_id']
    del df_test_original
    return df_submit, p_submit


def init_models(df_test, p_project):
    col_text = 'discourse_text'
    p_resource = p_project + '/resource'
    df_test = normalize_text_series(df_test, col_text)
    df_test_corpus = df_test[col_text].to_frame('Text')
    p_write = p_project + '/test_corpus.csv'
    df_test_corpus.to_csv(path_or_buf=p_write, mode='w', index=False, na_rep='', header=True, encoding='utf-8-sig')

    df_syn = aug_synonym(df_test[col_text], None, None, False)

    # p_emb = p_resource + '/GoogleNews-vectors-negative300.bin'
    # p_emb = p_resource + '/cc.en.300.vec'
    # p_emb = 'xlnet'
    # p_emb = 'roberta'
    # p_emb = 'bert-base-uncased'
    # df_w2v = aug_w2v(df_test[col_text], p_resource + '/GoogleNews-vectors-negative300.bin', None, None)
    df_w2v = aug_w2v(df_test[col_text], p_resource + '/cc.en.300.vec', None, None, False)

    # df_bert = aug_w2v(df_test[col_text], 'bert-base-uncased', None, None, False)
    df_bert = pd.read_csv(p_project + '/bert.csv')

    return [df_syn, df_w2v, df_bert]


def infer_model(df_test, l_labels, p_project, p_model, p_onehot, p_vectorizer, p_test):
    print('Inferring Model.')
    df_x = df_test.copy()
    l_augs = init_models(df_test, p_project)
    df_submit, p_submit = init_submit(p_test)
    tqdm.pandas()
    for i in tqdm(range(len(l_labels))):
        x = df_x.copy()
        curr_label = l_labels[i]
        s_filename = 'vectorizer_'+curr_label

        df_text_test, df_general_test = preprocess(x)
        df_general_test = onehot_encoder_infer(df_general_test, p_onehot, curr_label)

        vectorizer = get_pickle(p_vectorizer, s_filename)

        df_text_test = apply_tfidf(vectorizer, df_text_test)
        x_test = combine_features(df_text_test, df_general_test)
        o_model = load_model(p_model, curr_label)

        l_test_aug = list()
        for j in range(len(l_augs)):
            x_aug_curr = l_augs[j]
            df_text_test_aug = apply_tfidf(vectorizer, x_aug_curr)
            x_test_aug = combine_features(df_text_test_aug, df_general_test)
            l_test_aug.append(x_test_aug)

        y_preds = o_model.predict(x_test).reshape(-1, 1)
        y_probs = o_model.predict_proba(x_test)[:, 1]
        l_preds = [y_preds]
        l_probs = [y_probs]

        l_preds_aug, l_probs_aug = list(), list()
        for m in range(len(l_test_aug)):
            x_test_aug_curr = l_test_aug[m]
            y_preds_aug_curr = o_model.predict(x_test_aug_curr).reshape(-1, 1)
            l_preds_aug.append(y_preds_aug_curr)
            y_probs_aug_curr = o_model.predict_proba(x_test_aug_curr)
            y_probs_aug_curr = y_probs_aug_curr[:, 1]
            l_probs_aug.append(y_probs_aug_curr)
            
        l_preds.extend(l_preds_aug)
        l_probs.extend(l_probs_aug)

        y_aug_preds, y_aug_probs = 0, 0
        for n in range(len(l_test_aug)+1):
            curr_y_probs = l_probs[n]
            curr_y_preds = l_preds[n]
            if n == 0:  # original
                y_aug_probs += curr_y_probs / 2
                y_aug_preds += curr_y_preds / 2
            else:
                y_aug_probs += curr_y_probs / 6
                y_aug_preds += curr_y_preds / 6

        curr_class = curr_label[curr_label.rfind('_') + 1:]
        # df_submit[curr_class] = y_aug_preds
        df_submit[curr_class] = y_aug_probs

    df_submit.to_csv(path_or_buf=p_submit, mode='w', index=False, na_rep='', header=True, encoding='utf-8-sig')


def save_model(o_model, curr_label, i_top_fold, p_output):
    s_model = curr_label + '.' + 'pkl'
    p_output_model = p_output + '/' + s_model
    joblib.dump(o_model, p_output_model)
    print(f'Saved new top score model for class {curr_label} (fold: {i_top_fold}).')


def save_results(d_fold_results, t_top_fold, score, o_model, curr_label, d_onehots, vectorizer, p_model, p_onehot, p_vectorizer):
    i_top_fold = t_top_fold[0]
    f_top_score = t_top_fold[1]
    curr_class_fold_index = d_fold_results['Fold']
    curr_class_fold_score = d_fold_results[score]
    if curr_class_fold_score > f_top_score:
        f_top_score = curr_class_fold_score
        i_top_fold = curr_class_fold_index
        save_model(o_model, curr_label, i_top_fold, p_model)
        set_pickle(d_onehots, p_onehot, 'd_onehots_'+curr_label)
        set_pickle(vectorizer, p_vectorizer, 'vectorizer_'+curr_label)
    return tuple((i_top_fold, f_top_score))


def init_results():
    s_filename = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
    p_result = '/home/edoli/PycharmProjects/Feedback-Prediction/' + s_filename + '.csv'
    l_cols_results = ['Model', 'Class', 'Fold', 'AUC', 'PRAUC', 'Precision', 'Recall', 'F1']
    df_results = pd.DataFrame(columns=l_cols_results)
    df_results.to_csv(path_or_buf=p_result, mode='w', index=False, na_rep='', header=True, encoding='utf-8-sig')
    return df_results, p_result


def set_average(df_results, curr_label):
    df_class_results = df_results[df_results['Class'] == curr_label]
    d_avg_results = {'Model': 'AVG', 'Class': curr_label, 'Fold': -1, 'AUC': df_class_results['AUC'].mean(),
                     'PRAUC': df_class_results['PRAUC'].mean(), 'Precision': df_class_results['Precision'].mean(),
                     'Recall': df_class_results['Recall'].mean(), 'F1': df_class_results['F1'].mean(),
                     'Train': -1, 'Test': -1}
    df_avg_results = pd.DataFrame.from_dict(d_avg_results, orient='index').T
    df_results = pd.concat([df_results, df_avg_results], ignore_index=True, sort=False)
    return df_results


def train_model(df_train, l_labels, col_label, p_model, p_onehot, p_vectorizer, p_opt):
    print('Training Model.')
    # init()

    df_x = df_train.copy()
    df_x.drop(col_label, axis=1, inplace=True)

    df_y = df_train[col_label].copy()
    df_y = pd.DataFrame(df_y, columns=[col_label])
    df_y = onehot_encoder(df_y)

    df_results, p_result = init_results()
    score = 'AUC'

    vectorizer = get_vectorizer()

    tqdm.pandas()
    for i in tqdm(range(len(l_labels))):
        x = df_x.copy()

        curr_label = l_labels[i]
        y_curr = df_y[curr_label].copy()
        y_curr = pd.DataFrame(y_curr, columns=[curr_label])

        d_params = optimize_params(x, y_curr, curr_label, p_opt)
        _n_estimators = d_params['n_estimators']
        _max_depth = d_params['max_depth']
        _eta = d_params['eta']
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
        t_top_fold = tuple((-1, float('-inf')))
        
        for i_fold, (i_train, i_val) in enumerate(skf.split(x, y_curr)):
            x_train, x_test = x.iloc[i_train], x.iloc[i_val]
            y_train, y_test = y_curr.iloc[i_train], y_curr.iloc[i_val]

            df_text_train, df_general_train, df_text_test, df_general_test = preprocess(x_train, x_test)

            df_general_train, df_general_test, d_onehots = onehot_encoder(df_general_train, df_general_test)
            y_train.reset_index(drop=True, inplace=True)
            y_test.reset_index(drop=True, inplace=True)

            df_text_train, df_text_test = apply_tfidf(vectorizer, df_text_test, df_text_train)

            x_train = combine_features(df_text_train, df_general_train)
            x_test = combine_features(df_text_test, df_general_test)

            o_model = XGBClassifier(max_depth=_max_depth, eta=_eta,
                                    n_estimators=_n_estimators, verbosity=0,
                                    random_state=3, use_label_encoder=False)

            o_model.fit(x_train, y_train)

            y_preds = o_model.predict(x_test).reshape(-1, 1)

            y_probs = o_model.predict_proba(x_test)[:, 1]

            d_fold_results = evaluate_model(y_test, y_preds, y_probs, curr_label, 'stopwords', i_fold, i_train, i_val)

            t_top_fold = save_results(d_fold_results, t_top_fold, score, o_model, curr_label, d_onehots, vectorizer, p_model, p_onehot, p_vectorizer)
            df_fold_results = pd.DataFrame.from_dict(d_fold_results, orient='index').T
            df_results = pd.concat([df_results, df_fold_results], ignore_index=True, sort=False)

        df_results = set_average(df_results, curr_label)
        df_results.to_csv(path_or_buf=p_result, mode='a', index=False, na_rep='', header=False, encoding='utf-8-sig')
        # init()
    # plot(df_results)


def get_labels(curr_df, col_label):
    srs_class = curr_df[col_label].copy()
    one_hot_vector = pd.get_dummies(srs_class, dummy_na=False)
    l_labels = list()
    for label in one_hot_vector.columns.tolist():
        l_labels.append(label)
    return l_labels


def get_vectorizer():
    # _max_features = None
    _max_features = 10000
    # _stopwords = None
    _stopwords = list(stopwords.words('english'))
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=_stopwords,
                                 max_features=_max_features, analyzer='word',
                                 encoding='utf-8', decode_error='strict',
                                 lowercase=True, norm='l2', smooth_idf=True,
                                 sublinear_tf=False)
    return vectorizer


def generate_corpus(p_train, p_train_corpus, p_test, p_test_corpus):
    if not os.path.exists(p_train_corpus):
        df_train = pd.read_csv(p_train)
        df_train_corpus = pd.DataFrame(df_train[col_text], columns=[col_text])
        df_test = pd.read_csv(p_test)
        df_test_corpus = pd.DataFrame(df_test[col_text], columns=[col_text])
        df_train_corpus = normalize_text_series(df_train_corpus, col_text)
        df_test_corpus = normalize_text_series(df_test_corpus, col_text)
        df_train_corpus.to_csv(path_or_buf=p_train_corpus, mode='w', index=False, na_rep='', header=True,
                               encoding='utf-8-sig')
        df_test_corpus.to_csv(path_or_buf=p_test_corpus, mode='w', index=False, na_rep='', header=True,
                              encoding='utf-8-sig')


def test_model(df_train, l_labels):
    d1 = {'a261b6e14276': [0.017484, 0.208719, 0.773796],  # id, Ineffective, Adequate, Effective
          '5a88900e7dc1': [0.012145, 0.708830, 0.279024],
          '9790d835736b': [0.035899, 0.405702, 0.558399],
          '75ce6d68b67b': [0.032348, 0.313499, 0.654153],
          '93578d946723': [0.023663, 0.309438, 0.666899]}

    d2 = {'a261b6e14276': [0.269297, 0.571185, 0.318058],
          '5a88900e7dc1': [0.269297, 0.614405, 0.266067],
          '9790d835736b': [0.269297, 0.571185, 0.230284],
          '75ce6d68b67b': [0.269297, 0.571185, 0.230284],
          '93578d946723': [0.269297, 0.571185, 0.230284]}

    l_params = [{'n_estimators': 450, 'max_depth': 4, 'eta': 0.1},
                {'n_estimators': 380, 'max_depth': 4, 'eta': 0.275},
                {'n_estimators': 450, 'max_depth': 4, 'eta': 0.1}]

    df_x = df_train.copy()
    df_x.drop(col_label, axis=1, inplace=True)

    df_y = df_train[col_label].copy()
    df_y = pd.DataFrame(df_y, columns=[col_label])
    df_y = onehot_encoder(df_y)

    tqdm.pandas()
    for i in tqdm(range(len(l_labels))):
        curr_label = l_labels[i]

        d_params = l_params[i]
        _n_estimators = d_params['n_estimators']
        _max_depth = d_params['max_depth']
        _eta = d_params['eta']

        o_model = XGBClassifier(max_depth=_max_depth, eta=_eta,
                                n_estimators=_n_estimators, verbosity=0,
                                random_state=3, use_label_encoder=False)

        x = df_x.copy()
        y_curr = df_y[curr_label].copy()
        y_curr = pd.DataFrame(y_curr, columns=[curr_label])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
        tqdm.pandas()
        for i_fold, (i_train, i_val) in enumerate(skf.split(x, y_curr)):
            x_train, x_test = x.iloc[i_train], x.iloc[i_val]
            y_train, y_test = y_curr.iloc[i_train], y_curr.iloc[i_val]

            o_model.fit(x_train, y_train)

            y_preds = o_model.predict(x_test).reshape(-1, 1)

            y_probs = o_model.predict_proba(x_test)[:, 1]

            acc_test = round(accuracy_score(y_test, y_preds), 3)
            precision = round(precision_score(y_test, y_preds), 3)
            recall = round(recall_score(y_test, y_preds), 3)
            f1 = round(f1_score(y_test, y_preds), 3)
            fpr, tpr, threshold = roc_curve(y_test, y_preds)
            auc_score = round(roc_auc_score(y_test, y_probs), 3)
            arr_precision, arr_recall, threshold_pr_auc = precision_recall_curve(y_test, y_probs)
            pr_auc_score = round(auc(arr_recall, arr_precision), 3)
            print(f'Label: {curr_label}, AUC: {auc_score}, precision: {precision}, recall: {recall}, f1: {f1}, PRAUC: {pr_auc_score}')


def show_values(axs, results, space=.01):
    def _single(ax):
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height() + (p.get_height() * 0.01)
            value = '{:.1f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center")
    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)


def plot(df_results):
    sns.set_theme(style="whitegrid")
    s_title = 'TTA AUC CV Results'
    sns_plot = sns.barplot(data=df_results)
    show_values(sns_plot, df_results)
    plt.title(s_title)
    plt.xticks(rotation=45)
    plt.show()


if __name__ == "__main__":
    p_project = '/home/edoli/PycharmProjects/Feedback-Prediction'

    p_resource = set_dir_get_path(p_project, 'resource')
    p_model = set_dir_get_path(p_project, 'model')
    p_onehot = set_dir_get_path(p_project, 'onehot')
    p_vectorizer = set_dir_get_path(p_project, 'vectorizer')
    p_opt = set_dir_get_path(p_project, 'opt')
    p_aug = set_dir_get_path(p_project, 'aug')

    p_train = p_resource + '/train' + '.csv'
    p_test = p_resource + '/test' + '.csv'

    p_general_train = p_project + '/df_general_train' + '.csv'
    p_general_test = p_project + '/df_general_test' + '.csv'
    p_y_train = p_project + '/df_y_train' + '.csv'
    p_text_train = p_project + '/df_text_train' + '.csv'
    p_text_test = p_project + '/df_text_test' + '.csv'

    p_train_final = p_project + '/df_train' + '.csv'
    p_test_final = p_project + '/df_test' + '.csv'

    p_w2v_train = p_aug + '/df_w2v_train' + '.csv'
    p_syn_train = p_aug + '/df_syn_train' + '.csv'

    p_train_corpus = p_project + '/corpus_train' + '.csv'
    p_test_corpus = p_project + '/corpus_test' + '.csv'

    col_text = 'discourse_text'
    col_id = 'discourse_id'
    col_label = 'discourse_effectiveness'

    df_train = pd.read_csv(p_train)
    df_test = pd.read_csv(p_test)

    get_missing_data(df_train)
    get_ratio(df_train)
    generate_corpus(p_train, p_train_corpus, p_test, p_test_corpus)

    i_run_start = time.time()

    l_labels = get_labels(df_train, col_label)

    train_model(df_train, l_labels, col_label, p_model, p_onehot, p_vectorizer, p_opt)

    infer_model(df_test, l_labels, p_project, p_model, p_onehot, p_vectorizer, p_test)

    i_run_end = time.time()
    run_time = i_run_end - i_run_start
    print('Finished in: %.2f hours (%.2f minutes).' % (run_time / 60 / 60, run_time / 60))

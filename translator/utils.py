import re
import os
import json
import unicodedata

import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import text


# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn")


def clean_seq(w):
    w = unicode_to_ascii(w.lower().strip())
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,।])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ঁ -৯, ".", "?", "!", ",", "।")
    w = re.sub(r"[^a-zA-Z ঁ -৯।?.!,]+", " ", w)

    w = w.strip()

    return w


def add_start_and_end_token_to_seq(sentence):
    # adding a start and an end token to the sentence
    # so that the model knows when to start and stop predicting.
    return "<start> " + sentence + " <end>"


def texts_to_sequences(texts, tokenizer):
    tensor = tokenizer.texts_to_sequences(texts)
    tensor = sequence.pad_sequences(tensor, padding="post")

    return tensor


def get_lang_tokenizer(texts):
    lang_tokenizer = text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(texts)

    return lang_tokenizer


def save_tokenizer(tokenizer, save_at, file_name):
    path_to_file = os.path.join(save_at, file_name)
    if not os.path.exists(save_at):
        os.makedirs(save_at)
    with open(path_to_file, "w", encoding="utf-8") as fp:
        tokenizer_json = tokenizer.to_json()
        fp.write(json.dumps(tokenizer_json, indent=4, ensure_ascii=False))
    print("Tokenizer write at: ", path_to_file)


def load_tokenizer(path_to_tokenizer_file):
    print("Loading: ", path_to_tokenizer_file)
    with open(path_to_tokenizer_file, "r", encoding="utf-8") as fp:
        tokenizer_json = json.load(fp)
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
    
    return tokenizer


def show_index_to_word_mapping(inp_lang_tok, tensor):
    print('-' * 45)
    for t in tensor:
        if t != 0:
            print("{0} ----> {1}".format(
                t, inp_lang_tok.index_word[t]))


w = clean_seq("শেষে সামনের দরজার সামনে দাঁড়িয়ে লাথি মেরে দরজাটা ভেঙে ফেলল।")
print(w)
# শেষে সামনের দরজার সামনে দাডিযে লাথি মেরে দরজাটা ভেঙে ফেলল ।
# শেষে সামনের দরজার সামনে দাডিযে লাথি মেরে দরজাটা ভেঙে ফেলল ।
# শেষে সামনের দরজার সামনে দাডিযে লাথি মেরে দরজাটা ভেঙে ফেলল।
# শেষে সামনের দরজার সামনে দাঁড়িয়ে লাথি মেরে দরজাটা ভেঙে ফেলল ।
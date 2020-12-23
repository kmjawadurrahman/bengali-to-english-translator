import io
import os

import utils


class SUParaDataset():
    def __init__(self, path_to_eng_file, path_to_beng_file, num_data_to_load):
        self.path_to_eng_file = path_to_eng_file
        self.path_to_beng_file = path_to_beng_file
        self.num_data_to_load = num_data_to_load

    def read_data(self):
        eng_lines = io.open(
            self.path_to_eng_file, encoding="UTF-8").read().strip().split("\n")
        beng_lines = io.open(
            self.path_to_beng_file, encoding="UTF-8").read().strip().split("\n")

        return eng_lines, beng_lines

    def make_sequence_pair(self, eng_lines, beng_lines):
        seq_pairs = []
        for eng_line, beng_line in zip(eng_lines[:self.num_data_to_load], beng_lines[:self.num_data_to_load]):
            pair = []
            for seq in [eng_line, beng_line]:
                seq = utils.clean_seq(seq)
                seq = utils.add_start_and_end_token_to_seq(seq)
                pair.append(seq)
            seq_pairs.append(pair)

        return seq_pairs

    def create_dataset(self):
        eng_lines, beng_lines = self.read_data()
        word_pairs = self.make_sequence_pair(eng_lines, beng_lines)

        return zip(*word_pairs)

    def load_data(self):
        # creating cleaned input, output pairs
        targ_lang_text, inp_lang_text = self.create_dataset()

        targ_lang_tokenizer = utils.get_lang_tokenizer(targ_lang_text)
        inp_lang_tokenizer = utils.get_lang_tokenizer(inp_lang_text)

        target_tensor = utils.texts_to_sequences(targ_lang_text, targ_lang_tokenizer)
        input_tensor = utils.texts_to_sequences(inp_lang_text, inp_lang_tokenizer)

        tensor_pair = (input_tensor, target_tensor)
        tokenizer_pair = (inp_lang_tokenizer, targ_lang_tokenizer)

        return tensor_pair, tokenizer_pair

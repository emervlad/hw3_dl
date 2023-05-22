from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


class BPETokenizer:
    def __init__(self, sentence_list, pad_flag, vocab_size):
        """
        sentence_list - список предложений для обучения
        """
        # TODO: Реализуйте конструктор c помощью https://huggingface.co/docs/transformers/fast_tokenizers, обучите токенизатор, подготовьте нужные аттрибуты(word2index, index2word)
        self.pad_flag = pad_flag
        self.word2index = {}
        self.word2count = {}
        #self.index2word = {0: "SOS", 1: "EOS", 2: "UNK", 3: "PAD", 4: " "}
        #UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
        #'<unk>', '<pad>', '<bos>', '<eos>'
        self.word2index = {'<bos>': 2, '<eos>': 3, '<unk>': 0, '<pad>': 1}
        self.n_words = len(self.word2index)
        self.max_sent_len = 25
        self.special_tokens_set = self.word2index.keys()

        tokenizer = Tokenizer(BPE(unk_token='<unk>'))
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=list(self.word2index.keys()))

        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.train_from_iterator(sentence_list, trainer)

        self.tokenizer = tokenizer

        self.word2index = tokenizer.get_vocab()
        self.index2word = {self.word2index[x]: x for x in self.word2index}
        self.sos = [tokenizer.token_to_id("<bos>")]
        self.eos = [tokenizer.token_to_id("<eos>")]

        print(f'Space tokenizer fitted - {len(self.word2index)} tokens')

    def pad_sent(self, token_ids_list):
        if len(token_ids_list) < self.max_sent_len:
            padded_token_ids_list = token_ids_list + [self.word2index['<pad>']] * (self.max_sent_len - len(token_ids_list))
        else:
            padded_token_ids_list = token_ids_list[:self.max_sent_len - 1] + [self.word2index['<eos>']]
        return padded_token_ids_list

    def __call__(self, sentence):
        """
        sentence - входное предложение
        """
        # TODO: Реализуйте метод токенизации с помощью обученного токенизатора
        tokenized_data = self.tokenize(sentence)
        if self.pad_flag:
            tokenized_data = self.pad_sent(tokenized_data)
        return tokenized_data
    
    def tokenize(self, sentence):
        return self.sos + self.tokenizer.encode(sentence).ids + self.eos



    def decode(self, token_list):
        """
        token_list - предсказанные ID вашего токенизатора
        """
        # TODO: Реализуйте метод декодирования предсказанных токенов
        predicted_tokens = self.tokenizer.decode(token_list).split()

        #for token_id in token_list:
        #    predicted_token = self.index2word[token_id]
        #    predicted_tokens.append(predicted_token)
        filtered_tokens = list(filter(lambda x: x not in self.special_tokens_set, predicted_tokens))

        return filtered_tokens

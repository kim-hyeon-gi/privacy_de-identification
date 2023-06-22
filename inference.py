from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from gluonnlp.data import SentencepieceTokenizer

from data_utils.pad_sequence import keras_pad_fn
from data_utils.utils import Config
from data_utils.vocab_tokenizer import Tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
from model.net import (
    KobertBiGRUCRF,
    KobertBiLSTMCRF,
    KobertCRF,
    KobertSequenceFeatureExtractor,
)


def main(parser):

    args = parser.parse_args()
    model_dir = Path('./experiments/base_model_with_crf_val')
    model_config = Config(json_path=model_dir / 'config.json')

    # load vocab & tokenizer
    tok_path = "./ptr_lm_model/tokenizer_78b3253a26.model"
    ptr_tokenizer = SentencepieceTokenizer(tok_path)

    with open(model_dir / "vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    tokenizer = Tokenizer(vocab=vocab, split_fn=ptr_tokenizer, pad_fn=keras_pad_fn, maxlen=model_config.maxlen)

    # load ner_to_index.json
    with open(model_dir / "ner_to_index.json", 'rb') as f:
        ner_to_index = json.load(f)
        index_to_ner = {v: k for k, v in ner_to_index.items()}

    # model
    model = KobertCRF(config=model_config, num_classes=len(ner_to_index), vocab=vocab)

    # load
    model_dict = model.state_dict()
    checkpoint = torch.load("/home/hyeongikim/Desktop/kobert_eagle/pytorch-bert-crf-ner/experiments/base_model_with_crf_val/best-epoch-5-step-8000-acc-0.998.bin", map_location=torch.device('cuda'))
    # checkpoint = torch.load("./experiments/base_model_with_crf_val/best-epoch-12-step-1000-acc-0.960.bin", map_location=torch.device('cpu'))
    convert_keys = {}
    for k, v in checkpoint['model_state_dict'].items():
        new_key_name = k.replace("module.", '')
        if new_key_name not in model_dict:
            print("{} is not int model_dict".format(new_key_name))
            continue
        convert_keys[new_key_name] = v

    model.load_state_dict(convert_keys)
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    decoder_from_res = DecoderFromNamedEntitySequence(tokenizer=tokenizer, index_to_ner=index_to_ner)

    input_texts = "예 안녕하세요 저 한양대 에리카 소속 김기연 연구원이라고 합니다\n 네네  \n네 어제 모니터 두 개랑 복합기 관련해서 언제 배송되는지 문의 드렸었는데요\n \
    저 1시 괜찮을거 같고요 위치는 제가 주차 가능한 스타벅스가 있거든요 그래가지고 거기로 하는게 어떠실까 제안을 드리는데 괜찮을까요 "
    
    input_texts = "아 예 대표님 안녕하세요 김기연입니다 \n\
                    제가 오전에 미팅이 있어가지고 문자를 이제 확인했습니다 그래서\n\
                    아아 네네 오늘 스타벅스가 오늘 오전에 영업을 안하나 보네요\n\
                    제가 이런거 알고있었어야했는데 저도 몰라가지고 아우 그래서 2시까지 그럼 관문 베이커리라는 곳으로 가면 될까요? \n\
                    바로 위에 보니까 카페 되게 큰 게 있네요"
    input_texts = "받았습니다.\n\
 정보운영팀 송규복입니다.\n\
 네 안녕하세요.\n\
 혹시 여기 버즈니 개발팀 이태율입니다.\n\
 버니요.\n\
 버즈니 개발팀 이태훈인데요.\n\
 하정우 주임 님 전화번호라고 해서 전화드렸거든요.\n\
 지금 자리 비어서 제가 당겨 받았거든요.\n\
 네네 혹시 가지고 네네 네 혹시 번호 좀 넘겨주실 수 있을까요.\n\
 잠시만요 네 버즈 개발팀 이태윤이고요 네 dbju 개발 쪽 지원 요청하셔가지고요 네 010 네 3천7에요.\n\
 37에 칠 둘 하나 칠 번호로 전화 요청 부탁드립니다.\n\
 하나 7이요 네 맞습니다.\n\
 010 3007 7217 맞으시죠 네 맞습니다.\n\
 네 메모 남겨놓을게요 네 감사합니다.\n\
 네 회의 들어가서 몇 시에 올지는 잘 모르겠어요.\n\
 네네 알겠습니다.\n\
 네 네"
    
    # input_texts = "이게 그 또 출장 잘 다녀오고 이번주에 그 제가 진행하던 케이글로벌 스타트업 공모전에 파이널리스트가 결승에 진출해서 어제 발표까지 마쳤거든요\n\
    # 오전 10시나 11시 정도에 저는 괜찮아서 이번에도 과천도 괜찮으신가요?"
    
    for s in input_texts.split("\n"):
        list_of_input_ids = tokenizer.list_of_string_to_list_of_cls_sep_token_ids([s])
        x_input = torch.tensor(list_of_input_ids).long().to(device)
        list_of_pred_ids = model(x_input)
        # list_of_pred_ids = torch.transpose(list_of_pred_ids,1,0)
        
        list_of_ner_word, decoding_ner_sentence = decoder_from_res(list_of_input_ids=list_of_input_ids, list_of_pred_ids=list_of_pred_ids)
        
        print( decoding_ner_sentence)
    
class DecoderFromNamedEntitySequence():
    def __init__(self, tokenizer, index_to_ner):
        self.tokenizer = tokenizer
        self.index_to_ner = index_to_ner

    def __call__(self, list_of_input_ids, list_of_pred_ids):
        input_token = self.tokenizer.decode_token_ids(list_of_input_ids)[0]
        
        pred_ner_tag = [self.index_to_ner[pred_id] for pred_id in list_of_pred_ids[0]]

    

        # ----------------------------- parsing list_of_ner_word ----------------------------- #
        list_of_ner_word = []
        entity_word, entity_tag, prev_entity_tag = "", "", ""
        for i, pred_ner_tag_str in enumerate(pred_ner_tag):
            if "B-" in pred_ner_tag_str:
                entity_tag = pred_ner_tag_str[-3:]

                if prev_entity_tag != entity_tag and prev_entity_tag != "":
                    list_of_ner_word.append({"word": entity_word.replace("▁", " "), "tag": prev_entity_tag, "prob": None})

                entity_word = input_token[i]
                prev_entity_tag = entity_tag
            elif "I-"+entity_tag in pred_ner_tag_str:
                entity_word += input_token[i]
            else:
                if entity_word != "" and entity_tag != "":
                    list_of_ner_word.append({"word":entity_word.replace("▁", " "), "tag":entity_tag, "prob":None})
                entity_word, entity_tag, prev_entity_tag = "", "", ""


        # ----------------------------- parsing decoding_ner_sentence ----------------------------- #
        decoding_ner_sentence = ""
        is_prev_entity = False
        prev_entity_tag = ""
        is_there_B_before_I = False

        for token_str, pred_ner_tag_str in zip(input_token, pred_ner_tag):
            token_str = token_str.replace('▁', ' ')  # '▁' 토큰을 띄어쓰기로 교체

            if 'B-' in pred_ner_tag_str:
                if is_prev_entity is True:
                    decoding_ner_sentence += ':' + prev_entity_tag+ '>'

                if token_str[0] == ' ':
                    token_str = list(token_str)
                    token_str[0] = ' <'
                    token_str = ''.join(token_str)
                    decoding_ner_sentence += token_str
                else:
                    decoding_ner_sentence += '<' + token_str
                is_prev_entity = True
                prev_entity_tag = pred_ner_tag_str[-3:] # 첫번째 예측을 기준으로 하겠음
                is_there_B_before_I = True

            elif 'I-' in pred_ner_tag_str:
                decoding_ner_sentence += token_str

                if is_there_B_before_I is True: # I가 나오기전에 B가 있어야하도록 체크
                    is_prev_entity = True
            else:
                if is_prev_entity is True:
                    decoding_ner_sentence += ':' + prev_entity_tag+ '>' + token_str
                    is_prev_entity = False
                    is_there_B_before_I = False
                else:
                    decoding_ner_sentence += token_str

        return list_of_ner_word, decoding_ner_sentence


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data_in', help="Directory containing config.json of data")
    # parser.add_argument('--model_dir', default='./experiments/base_model', help="Directory containing config.json of model")
    parser.add_argument('--model_dir', default='./experiments/base_model_with_crf_val', help="Directory containing config.json of model")
    # parser.add_argument('--model_dir', default='./experiments/base_model_with_crf', help="Directory containing config.json of model")
    # parser.add_argument('--model_dir', default='./experiments/base_model_with_bilstm_crf', help="Directory containing config.json of model")
    # parser.add_argument('--model_dir', default='./experiments/base_model_with_bigru_crf', help="Directory containing config.json of model")

    main(parser)
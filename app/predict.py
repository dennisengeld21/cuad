import argparse
import glob
import logging
import os
import random
import timeit
import json
from http.client import HTTPException

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from tqdm import tqdm, trange

import transformers
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)

from train import to_list, get_balanced_dataset
from utils import (
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from transformers.trainer_utils import is_main_process

from fastapi import FastAPI

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

##### Functions
###### APP

app = FastAPI()

@app.get("/")
async def root():
    return {"ready steady"}


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

@app.get("/{model_name}/{file_id}")
def predict(model_name :str, file_id: str):

    #### set args
    file_path = "./in/" + file_id + ".txt"
    model_name_or_path = ""
    if model_name == "roberta":
        model_name_or_path = "./roberta-base"
    if model_name == "robertaLarge":
        model_name_or_path = "./roberta-large"
    if model_name == "deberta":
        model_name_or_path = "./deberta-v2-xlarge"

    if model_name_or_path == "":
        raise HTTPException(status_code=404, detail="model not found")

    config_name = ""
    cache_dir = ""
    tokenizer_name = ""
    do_lower_case = False
    predict_file = './data/predict_file.json'

    ### create squad file for prediction
    with open("./data/skelett.json", "r") as f:
        skelett_dict = json.load(f)
    with open(file_path) as f:
        context = f.readlines()
    skelett_dict['data'][0]['title'] = str(file_id)
    skelett_dict['data'][0]['paragraphs'][0]['context'] = str(context)
    with open(predict_file, 'w') as fp:
        json.dump(skelett_dict, fp)

    #### load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name if tokenizer_name else model_name_or_path,
        do_lower_case=do_lower_case,
        cache_dir=cache_dir if cache_dir else None,
        use_fast=False,  # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
    )

    ### preprocess pred file

    processor = SquadV1Processor()
    examples = processor.get_dev_examples("./data", filename="predict_file.json")
    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=384,
        doc_stride=128,
        max_query_length=64,
        is_training=not True,
        return_dataset="pt",
        threads=1,
    )
    pred_sampler = SequentialSampler(dataset)
    pred_loader = DataLoader(dataset, sampler=pred_sampler, batch_size=8)

    ### load model
    config = AutoConfig.from_pretrained(
        config_name if config_name else model_name_or_path,
        cache_dir=cache_dir if cache_dir else None,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        cache_dir=cache_dir if cache_dir else None,
    )

    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    logging.warning(device)

    all_results = []

    for batch in tqdm(pred_loader, desc="Predicting"):
        with torch.no_grad():

            # pull batched items from loader
            #input_ids = batch['input_ids'].to('cpu')
            #attention_mask = batch['attention_mask'].to('cpu')
            # make predictions
            inputs = {
                "input_ids": batch[0].to(device),
                "attention_mask": batch[1].to(device),
                "token_type_ids": batch[2].to(device),
            }

            feature_indices = batch[3].to(device)

            outputs = model(**inputs)

            for i, feature_index in enumerate(feature_indices):
                eval_feature = features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs.to_tuple()]

                # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
                # models only use two.
                if len(output) >= 5:
                    start_logits = output[0]
                    start_top_index = output[1]
                    end_logits = output[2]
                    end_top_index = output[3]
                    cls_logits = output[4]

                    result = SquadResult(
                        unique_id,
                        start_logits,
                        end_logits,
                        start_top_index=start_top_index,
                        end_top_index=end_top_index,
                        cls_logits=cls_logits,
                    )

                else:
                    start_logits, end_logits = output
                    result = SquadResult(unique_id, start_logits, end_logits)

                all_results.append(result)

    with open(predict_file, "r") as f:
        json_test_dict = json.load(f)

    ### simulate args
    n_best_size = 20
    max_answer_length = 512
    output_prediction_file = "./out/" + file_id + ".json"
    output_nbest_file = "./out/nbest_" + file_id +".json"
    output_null_log_odds_file = "./out/null_odds_" + file_id + ".json"
    verbose_logging = False
    version_2_with_negative = True
    null_score_diff_threshold = 0.0

    compute_predictions_logits(
        json_test_dict,
        examples,
        features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        verbose_logging,
        version_2_with_negative,
        null_score_diff_threshold,
        tokenizer,
    )

    return {"writing prediction to cuad/out/"}

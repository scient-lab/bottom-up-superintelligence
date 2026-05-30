'''
Copyright (c) 2025 The Trustees of Princeton University
Authors: Bhishma Dedhia, Yuval Kansal, Niraj K. Jha

Licensed for academic and research use only.
See LICENSE file for full terms.
'''

import json
import datasets
from transformers import AutoTokenizer
from argparse import ArgumentParser
from functools import partial
import random
import os

def ApplyTemplate(question_item, tokenizer):
    question = question_item['question_and_explanation'].split('<Question>')[1].split('</Question>')[0]
    options = question_item['question_and_explanation'].split('<Options>')[1].split('</Options>')[0]
    #randomly select one option letter for the answer instruction from A. B. C. and D.
    answer_option = random.choice(['A', 'B', 'C', 'D'])
    answer_instruction = 'Please only output the choice letter in the answer field e.g. Final Answer: ' + answer_option
    prompt = question + '\nOptions:' + options + '\n' + answer_instruction
    cot = question_item['question_and_explanation'].split('<Explanation>')[1].split('</Explanation>')[0]
    answer = question_item['question_and_explanation'].split('<Answer>:')[1].split('</Answer>')[0]

    text = tokenizer.apply_chat_template([
        {"role": "user", "content": prompt.strip()},
        {
            "role": "assistant", 
            "content": "<think>\n" + cot.strip() + "\n</think>\nFinal Answer: " + answer.strip()
        }
    ], tokenize=False)

    return dict(text=text)

def main(args):
    # load tokenizer from cache
    tokenizer = AutoTokenizer.from_pretrained(os.environ.get('BUS_MODEL_NAME', 'Qwen/Qwen3-4B'), cache_dir=os.environ.get('HF_HOME', None))
    with open(args.dataset_train_path, "r") as f:
        questions_train = json.load(f)

    # kg-pipeline fork-patch: upstream `generate_curriculum.py` doesn't emit
    # a 'category' field (kg-pipeline's csv_to_kg adapter writes a single dummy
    # 'smoke_unsorted' category for all concepts). `.get(..., default)` keeps
    # the schema present for downstream code that expects it, without crashing
    # on KeyError. Real paper-scale runs likely have a separate categorization
    # step that backfills this field.
    questions_dict = {
            'category': [question.get('category', 'smoke_unsorted') for question in questions_train],
            'source_concept': [question['source_concept'] for question in questions_train],
            'target_concept': [question['target_concept'] for question in questions_train],
            'paths': [question['paths'] for question in questions_train],
            'question_and_explanation': [question['question_and_explanation'] for question in questions_train],
    }
    dataset = datasets.Dataset.from_dict(questions_dict)
    dataset = dataset.map(partial(ApplyTemplate, tokenizer=tokenizer))
    dataset_w_split = datasets.DatasetDict({'train': dataset})
    # display the first 5 examples
    for i in range(5):
        print(dataset_w_split['train'][i]['text'])
        print('*'*100)
        
    op_dir = os.path.dirname(args.dataset_train_path)
    dir_name = 'tokenized_' + os.path.basename(args.dataset_train_path).replace('.json', '')
    dataset_w_split.save_to_disk(f"{op_dir}/{dir_name}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_train_path", type=str, default="/curriculum_training_data/curriculum_dataset_hop_3_decontaminated.json")
    args = parser.parse_args()
    main(args)

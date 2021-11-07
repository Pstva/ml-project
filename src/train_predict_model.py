#!/usr/bin/python
# -*- coding: UTF-8 -*-


import pandas as pd
import torch
from transformers import XLMRobertaTokenizer
from transformers import XLMRobertaForSequenceClassification
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup
from seqeval.metrics import accuracy_score
from tqdm import tqdm
from collections import defaultdict


# Загрузка данных

en = pd.read_csv("data/eng_train_data.csv")
fr = pd.read_csv("data/fr_text.csv")

# Классы - 0/1/2
target_names = list(set(en["class"]))

# Загрузка токенизатора и модели

MODEL = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = XLMRobertaForSequenceClassification.from_pretrained(
    MODEL, num_labels=len(target_names)
)
model.to(device)


# Функции для подготовки и загрузки данных в модель

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_SEQ_LENGTH = 200


class BertInputItem(object):
    """An item with all the necessary attributes for finetuning BERT."""

    def __init__(self, text, input_ids, input_mask, segment_ids, label_id):
        self.text = text
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_examples_to_inputs(
    example_texts, example_labels, max_seq_length, tokenizer
):
    """Loads a data file into a list of `InputBatch`s."""

    input_items = []
    examples = zip(example_texts, example_labels)
    for (ex_index, (text, label)) in enumerate(examples):

        # Create a list of token ids
        input_ids = tokenizer.encode(f"[CLS] {text} [SEP]")
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]

        # All our tokens are in the first input segment (id 0).
        segment_ids = [0] * len(input_ids)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        input_items.append(
            BertInputItem(
                text=text,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label,
            )
        )

    return input_items


def get_data_loader(features, max_seq_length, batch_size=32, shuffle=True):

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
    return dataloader


# Подготовка данных для модели
train_texts, test_texts, train_labels, test_labels = train_test_split(
    en["review_body"], en["class"], test_size=0.1, random_state=1, shuffle=True
)

train_features = convert_examples_to_inputs(
    train_texts, train_labels, MAX_SEQ_LENGTH, tokenizer
)

test_features = convert_examples_to_inputs(
    test_texts, test_labels, MAX_SEQ_LENGTH, tokenizer
)

fr["class"] = -1

pred_features = convert_examples_to_inputs(
    fr["review_body"], fr["class"], MAX_SEQ_LENGTH, tokenizer
)


train_dataloader = get_data_loader(train_features, MAX_SEQ_LENGTH, shuffle=True)
test_dataloader = get_data_loader(test_features, MAX_SEQ_LENGTH, shuffle=False)
pred_dataloader = get_data_loader(pred_features, MAX_SEQ_LENGTH, shuffle=False)


# Функция для evalution на тестовом датасете


def evaluate(model, dataloader, device="cuda"):
    model.eval()

    predicted_labels, correct_labels = [], []

    model.to(device)
    for step, batch in enumerate(tqdm(dataloader, desc="Evaluation iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            logits = model(
                input_ids, attention_mask=input_mask, token_type_ids=segment_ids
            )
        logits = logits[0]
        outputs = np.argmax(logits.to("cpu").detach().numpy(), axis=1)
        label_ids = label_ids.to("cpu").numpy()

        predicted_labels += list(outputs)
        correct_labels += list(label_ids)

    return accuracy_score(correct_labels, predicted_labels)


# Функция для обучения модели

EPOCHS = 3
warmup_proportion = 0.1
batch_size = 32
learning_rate = 5e-5
gradient_accumulation_steps = 1

num_train_steps = int(
    len(en["review_body"]) / batch_size / gradient_accumulation_steps * EPOCHS
)
num_warmup_steps = int(warmup_proportion * num_train_steps)

param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [
            p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.01,
    },
    {
        "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, correct_bias=False)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
)


def train(model, train_dataloader, patience=2, max_grad_norm=5):
    model.to(device)
    model.train()
    y_true = []
    y_pred = []
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Training iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        outputs = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
            labels=label_ids,
        )
        loss = outputs[0]
        logits = outputs[1]
        logits = np.argmax(logits.to("cpu").detach().numpy(), axis=1)
        label_ids = label_ids.to("cpu").numpy()
        y_true += list(label_ids)
        y_pred += list(logits)

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()

        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1

        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
    return accuracy_score(y_true, y_pred), tr_loss / nb_tr_steps


# Обучение модели

EPOCHS = 3

history = defaultdict(list)
for epoch in range(EPOCHS):

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("-" * 10)

    train_acc, train_loss = train(model, train_dataloader)

    print(f"Train loss {train_loss} accuracy {train_acc} ")

    val_acc = evaluate(model, test_dataloader)

    print(f"Val   accuracy {val_acc}")
    print()

    history["train_acc"].append(train_acc)
    history["train_loss"].append(train_loss)
    history["val_acc"].append(val_acc)

# Предсказание для данных на французском


def predict_classes(model, dataloader):
    model.to(device)
    model.eval()
    predicted_labels = []
    for step, batch in enumerate(tqdm(dataloader)):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            output = model(
                input_ids, attention_mask=input_mask, token_type_ids=segment_ids
            )
            logits = output[0]
        outputs = np.argmax(logits.to("cpu"), axis=1)
        label_ids = label_ids.to("cpu").numpy()

        predicted_labels += list(outputs)
    return predicted_labels


preds = predict_classes(model, pred_dataloader)


def create_submission(preds):
    result = []
    for i in range(len(preds)):
        result.append([i, preds[i].item()])
    return result


sent_labels = create_submission(preds)
subm = pd.DataFrame(sent_labels, columns=["id", "class"])
subm.to_csv("preditcions/submission_roberta_3epoch.csv", index=False)

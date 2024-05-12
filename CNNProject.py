import pandas as pd
import numpy as np

from transformers import (AutoModelForSeq2SeqLM,
                          AutoTokenizer,
                          DataCollatorForSeq2Seq,
                          TrainingArguments,
                          Trainer,
                          IntervalStrategy,
                          EarlyStoppingCallback,
                         )
from datasets import Dataset, DatasetDict, load_metric
import torch
import nltk
nltk.download("punkt", quiet=True)

metric = load_metric("rouge", trust_remote_code=True)

train = pd.read_csv("test.csv")
train_dataset = Dataset.from_pandas(train)
cheakPoint = "facebook/bart-large-cnn"
model = AutoModelForSeq2SeqLM.from_pretrained(cheakPoint).to(device)
tokenizer = AutoTokenizer.from_pretrained(cheakPoint)
# Working on a sample
train_dataset = train_dataset.shuffle(seed=42)
train, val = train_dataset.select(range(400)), train_dataset.select(range(400, 490))
dataset_dict = DatasetDict({"train": train, "validation": val})
dataset_dict.remove_columns("id")
def batch_tokenize_preprocess(batch, tokenizer, encoder_max_length, decoder_max_length):
    
    source, target = batch["article"], batch["highlights"]
    source_tokenized = tokenizer(source, padding="max_length", truncation=True, max_length=encoder_max_length )
    target_tokenized = tokenizer(target, padding="max_length", truncation=True, max_length=decoder_max_length)

    # Ignore padding in the loss
    target_labels = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]

    # Create a dictionary for the batch
    batch_dict = {
        "input_ids": source_tokenized["input_ids"],
        "attention_mask": source_tokenized["attention_mask"],
        "labels": target_labels,
    }

    return batch_dict

train_data = train.map(
    lambda batch: batch_tokenize_preprocess(batch, tokenizer, encoder_max_length, decoder_max_length),
    batched=True,
    remove_columns=train.column_names,
)


validation_data = val.map(
    lambda batch: batch_tokenize_preprocess(batch, tokenizer, encoder_max_length, decoder_max_length),
    batched=True,
    remove_columns=val.column_names,
)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
#     print(f"preds: {preds}")
#     print(f"labels: {labels}")
    decoded_preds = [tokenizer.batch_decode(np.argmax(pred, axis=1), skip_special_tokens=True) for pred in preds]
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = [tokenizer.batch_decode(label, skip_special_tokens=True) for label in labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {
        k: round(v, 4) for k, v in result.items()
    }
    return result

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = TrainingArguments(
    output_dir='bart_CNN_NLP',
    num_train_epochs=4,  
    per_device_train_batch_size=batch_size, 
    per_device_eval_batch_size=batch_size,
    warmup_steps=500,
    weight_decay=0.1,
    label_smoothing_factor=0.1,
    logging_dir="bart_logs",
    logging_steps=20,
    load_best_model_at_end=True,
    evaluation_strategy = "steps",
    eval_steps = 40,
    save_steps=1e6,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=validation_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
)

#Train the Model
trainer.train()

def generate_summary(test_samples, model, max_length):
    inputs = tokenizer(
        test_samples,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    outputs = model.generate(input_ids, attention_mask=attention_mask)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return output_str



sample = "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."
res = generate_summary(sample, trainer.model, max_length=1028)
print(res)

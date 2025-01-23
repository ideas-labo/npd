import os
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import KFold
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def preprocess_data(dataset, tokenizer):
    def tokenize_function(dataset):
        return tokenizer(dataset["text"], padding=True, truncation=True, return_tensors="pt")

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets

repo_name = "./roberta-trainer-pp"
modelpath = "./model_pp"
datasetpath = "./data_pp/cppdatasets"
# checkpoint_path = ""
model_save_path = "./model_pp"

# Load the dataset
# raw_datasets = DatasetDict.load_from_disk(datasetpath)
# texts = raw_datasets['train']['text']
# labels = raw_datasets['train']['label']

# # Initialize tokenizer and collator
tokenizer = AutoTokenizer.from_pretrained(modelpath)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# kf = KFold(n_splits=10, shuffle=True, random_state=42)  # Set up 10-Fold Cross Validation

# all_preds = []
# all_labels = []

# # Ensure using the first GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# for fold, (train_index, val_index) in enumerate(kf.split(texts)):
#     if fold != 3:  # Only use fold 4
#         continue

#     print(f"Fold {fold + 1}")

#     # Split data
#     train_texts = np.array(texts)[train_index]
#     train_labels = np.array(labels)[train_index]
#     val_texts = np.array(texts)[val_index]
#     val_labels = np.array(labels)[val_index]

#     # Create DatasetDict for training and validation
#     train_dataset_dict = Dataset.from_dict({'text': train_texts.tolist(), 'label': train_labels.tolist()})
#     val_dataset_dict = Dataset.from_dict({'text': val_texts.tolist(), 'label': val_labels.tolist()})

#     # Data preprocessing
#     train_dataset = preprocess_data(train_dataset_dict, tokenizer)
#     val_dataset = preprocess_data(val_dataset_dict, tokenizer)

#     # Initialize model
#     model = RobertaForSequenceClassification.from_pretrained(modelpath, num_labels=5)

#     # Set up training arguments
#     training_args = TrainingArguments(
#         output_dir=f"{repo_name}_fold_{fold + 1}",
#         learning_rate=1e-5,
#         per_device_train_batch_size=4,
#         per_device_eval_batch_size=4,
#         num_train_epochs=10,
#         weight_decay=0.01,
#         save_strategy="epoch",
#         evaluation_strategy="epoch",
#         load_best_model_at_end=True,
#         metric_for_best_model="f1",
#         fp16=True,
#     )

#     # Initialize Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#         data_collator=data_collator,
#         tokenizer=tokenizer,
#         compute_metrics=compute_metrics,
#     )

#     # Train model
#     if os.path.exists(checkpoint_path):
#         trainer.train(resume_from_checkpoint=checkpoint_path)
#     else:
#         trainer.train()

#     # Save the trained model
#     trainer.save_model(model_save_path)

#     # Evaluate model
#     eval_results = trainer.evaluate()
    
#     # Collect predictions and labels for later aggregation
#     predictions, true_labels = [], []
#     for batch in trainer.get_eval_dataloader():
#         with torch.no_grad():
#             inputs = {k: v.to(trainer.args.device) for k, v in batch.items()}
#             outputs = model(**inputs)
#             preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
#             labels1 = inputs['labels'].cpu().numpy()
#             predictions.extend(preds)
#             true_labels.extend(labels1)
    
#     all_preds.extend(predictions)
#     all_labels.extend(true_labels)

# # Calculate and print aggregated results
# overall_accuracy = accuracy_score(all_labels, all_preds)
# overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)

# print(f"Aggregated Results across all folds:")
# print(f"Overall Accuracy: {overall_accuracy}")
# print(f"Overall Precision: {overall_precision}")
# print(f"Overall Recall: {overall_recall}")
# print(f"Overall F1 Score: {overall_f1}")

# Generate classification report
type_list = ['NONE', 'Optimization', 'Function', 'Resource', 'Tradeoff']
# report = classification_report(all_labels, all_preds, target_names=type_list)
# print("Classification Report:\n", report)

# Inference on the test set
texts = []
true_labels = []
predictions = []

with open("./data_pp/cppp.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()
    for line in lines:       
        texts.append(line)
            

# Initialize the model for inference
model = RobertaForSequenceClassification.from_pretrained(model_save_path, num_labels=5)
model.eval()

with open("./data_pp/predictions.txt", "w", encoding="utf-8") as out_file:
    for text in texts:
        with torch.no_grad():
            inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            first_part = text.split(' ')[0]
            out_file.write(f"{first_part}\t{type_list[prediction]}\n")

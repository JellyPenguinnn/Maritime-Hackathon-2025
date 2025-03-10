#!/usr/bin/env python
import os
import sys
import re
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report
)

# Transformers
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import optuna
from torch.utils.data import Dataset

# -----------------------------------------
# 1) Device Setup
# -----------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# -----------------------------------------
# 2) Configuration
# -----------------------------------------
CONFIG = {
    "model_name": "microsoft/deberta-v3-base",  # Switch to base model for faster training
    "max_length": 128,  # Smaller sequence length for speed
    "num_labels": 4,    # [NOT_A_DEFICIENCY, LOW, MEDIUM, HIGH]
    "test_size": 0.2,   # 80% train, 20% val
    "seed": 42,
    "n_trials": 5,      # Fewer Optuna trials for quicker search
    "use_label_smoothing": True,
    "label_smoothing_factor": 0.05,
    "use_class_weights": False,  # We'll rely on label smoothing
    "num_train_epochs": 5,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type != "cpu":
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])

# -----------------------------------------
# 3) Structured Text Parsing
# -----------------------------------------
SECTION_HEADERS = [
    "Immediate Causes",
    "Root Cause Analysis",
    "Corrective Action",
    "Preventive Action"
]

def parse_structured_text(def_text):
    """Extract domain‐specific sections from the deficiency text."""
    sections = {header: "" for header in SECTION_HEADERS}
    if not isinstance(def_text, str) or not def_text.strip():
        return sections
    for i, header in enumerate(SECTION_HEADERS):
        if i < len(SECTION_HEADERS) - 1:
            next_header = SECTION_HEADERS[i+1]
            pattern = rf"{header}:(.*?)(?={next_header}:|$)"
        else:
            pattern = rf"{header}:(.*)$"
        match = re.search(pattern, def_text, flags=re.IGNORECASE|re.DOTALL)
        if match:
            sections[header] = match.group(1).strip()
    return sections

def build_structured_text(sections):
    """Rebuild a structured text block for the model."""
    lines = []
    for header in SECTION_HEADERS:
        content = sections[header].strip()
        if content:
            lines.append(f"{header.upper()}: {content}")
        else:
            lines.append(f"{header.upper()}: [None]")
    return "\n".join(lines)

# -----------------------------------------
# 4) Domain Heuristics
# -----------------------------------------
def domain_features_from_sections(sections):
    """Example: Add numeric flags for domain signals in each section."""
    feats = {
        "immediate_critical": 0,
        "rootcause_urgent": 0
    }
    if "critical" in sections["Immediate Causes"].lower():
        feats["immediate_critical"] = 1
    if "urgent" in sections["Root Cause Analysis"].lower():
        feats["rootcause_urgent"] = 1
    return feats

# -----------------------------------------
# 5) Keyword Patterns
# -----------------------------------------
SEVERITY_KEYWORDS = {
    "high": [
        r"\b(detainable|critical|urgent|major|serious|nonconformity|violation|severe|hazardous|emergency|fire safety|pollution prevention|life-threatening|structural failure|non-operational)\b"
    ],
    "medium": [
        r"\b(moderate|concern|warning|defect|improper|inadequate|maintenance|corrosion|expired|documentation|training|protective equipment)\b"
    ],
    "low": [
        r"\b(minor|cosmetic|observational|cleanliness|administrative|record keeping|non-essential|notification|recommendation)\b"
    ]
}

def count_keywords(text, patterns):
    text = str(text).lower()
    return sum(len(re.findall(p, text)) for p in patterns)

def unify_severity(x):
    x = str(x).strip().upper()
    if "NOT A DEFICIENCY" in x:
        return "NOT_A_DEFICIENCY"
    elif "LOW" in x:
        return "LOW"
    elif "MEDIUM" in x:
        return "MEDIUM"
    elif "HIGH" in x:
        return "HIGH"
    else:
        return "NOT_A_DEFICIENCY"

def encode_categorical(df, column_name):
    """Label-encode a categorical column."""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df[column_name] = df[column_name].fillna("Unknown").astype(str)
    df[column_name] = le.fit_transform(df[column_name])
    return df, le

# -----------------------------------------
# 6) Preprocessing with Structured Parsing
# -----------------------------------------
def preprocess_data(
    df,
    scaler=None,
    is_train=True,
    require_label=True,
    extra_categorical=["VesselGroup"]  # minimal
):
    if require_label:
        required_cols = ["def_text", "consensus_severity", "age"]
    else:
        required_cols = ["def_text", "age"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"[ERROR] Missing required column '{col}' in data.")

    df = df.copy()
    df = df.dropna(subset=required_cols)
    df["def_text"] = df["def_text"].fillna("No deficiency text provided").astype(str)
    df["age"] = df["age"].fillna(df["age"].median())

    if require_label:
        label_map = {"NOT_A_DEFICIENCY": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
        df["consensus_severity"] = df["consensus_severity"].apply(unify_severity)
        df["labels"] = df["consensus_severity"].map(label_map)

    # Parse sections
    df["parsed_sections"] = df["def_text"].apply(parse_structured_text)
    df["domain_feats"] = df["parsed_sections"].apply(domain_features_from_sections)
    df["structured_text"] = df["parsed_sections"].apply(build_structured_text)

    # Combine structured text + original text
    df["combined_text"] = df.apply(
        lambda row: f"Age: {row['age']}.\n\n{row['structured_text']}\n\nOriginal: {row['def_text']}",
        axis=1
    )

    # Basic numeric features
    df["text_length"] = df["def_text"].apply(lambda x: len(str(x)))
    df["keyword_high"] = df["def_text"].apply(lambda x: count_keywords(x, SEVERITY_KEYWORDS["high"]))
    df["keyword_medium"] = df["def_text"].apply(lambda x: count_keywords(x, SEVERITY_KEYWORDS["medium"]))
    df["keyword_low"] = df["def_text"].apply(lambda x: count_keywords(x, SEVERITY_KEYWORDS["low"]))

    # Domain feats
    df["immediate_critical"] = df["domain_feats"].apply(lambda x: x["immediate_critical"])
    df["rootcause_urgent"] = df["domain_feats"].apply(lambda x: x["rootcause_urgent"])

    # Encode minimal categorical
    for catcol in extra_categorical:
        if catcol not in df.columns:
            df[catcol] = "Unknown"
        df, _ = encode_categorical(df, catcol)

    # Numeric columns
    numeric_cols = [
        "age", "text_length", "keyword_high", "keyword_medium", "keyword_low",
        "immediate_critical", "rootcause_urgent"
    ]
    if "VesselGroup" in df.columns:
        numeric_cols.append("VesselGroup")

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    # Scale
    if is_train:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided for non-training data.")
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df, scaler, numeric_cols

# -----------------------------------------
# 7) Custom Dataset
# -----------------------------------------
class DeficiencyDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, numeric_cols):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.numeric_cols = numeric_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoding = self.tokenizer(
            row["combined_text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        numeric_feats = torch.tensor(
            np.array(row[self.numeric_cols].tolist(), dtype=np.float32),
            dtype=torch.float
        )

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "numeric_feats": numeric_feats
        }
        if "labels" in row:
            item["labels"] = torch.tensor(row["labels"], dtype=torch.long)
        return item

# -----------------------------------------
# 8) Hybrid Model
# -----------------------------------------
class DeficiencyClassifier(nn.Module):
    def __init__(self, model_name, num_labels, feature_size, label_smoothing=0.0):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.text_model = AutoModel.from_pretrained(model_name, config=self.config)

        self.numeric_processor = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(64)
        )
        combined_dim = self.config.hidden_size + 64
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )
        if label_smoothing > 0:
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, numeric_feats, labels=None):
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        numeric_output = self.numeric_processor(numeric_feats)
        combined = torch.cat([cls_output, numeric_output], dim=1)
        logits = self.classifier(combined)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}

# -----------------------------------------
# 9) Metrics
# -----------------------------------------
def compute_metrics(eval_pred):
    logits, label_ids = eval_pred
    if label_ids is None:
        return {}
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(label_ids, preds, average="weighted", zero_division=0)
    acc = accuracy_score(label_ids, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# -----------------------------------------
# 10) Hyperparameter Search
# -----------------------------------------
def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 10),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [2, 4, 8])
    }

# -----------------------------------------
# Main
# -----------------------------------------
def main():
    train_file = "final_consensus_severity.csv"
    test_file = "psc_severity_test.csv"

    if not os.path.exists(train_file):
        print(f"[ERROR] Training file '{train_file}' not found.")
        sys.exit(1)

    # Load & Preprocess
    df = pd.read_csv(train_file)
    df, scaler, numeric_cols = preprocess_data(
        df, scaler=None, is_train=True, require_label=True, extra_categorical=["VesselGroup"]
    )

    # Train/Val Split
    train_df, val_df = train_test_split(
        df, test_size=CONFIG["test_size"], stratify=df["labels"], random_state=CONFIG["seed"]
    )
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    val_df, _, _ = preprocess_data(
        val_df, scaler=scaler, is_train=False, require_label=True, extra_categorical=["VesselGroup"]
    )

    # Create Datasets
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    train_dataset = DeficiencyDataset(train_df, tokenizer, CONFIG["max_length"], numeric_cols)
    val_dataset = DeficiencyDataset(val_df, tokenizer, CONFIG["max_length"], numeric_cols)

    # Label smoothing
    smoothing_factor = CONFIG["label_smoothing_factor"] if CONFIG["use_label_smoothing"] else 0.0

    def model_init():
        return DeficiencyClassifier(
            CONFIG["model_name"],
            CONFIG["num_labels"],
            feature_size=len(numeric_cols),
            label_smoothing=smoothing_factor
        )

    # Training Arguments (use eval_strategy to avoid the warning)
    training_args = TrainingArguments(
        output_dir="./best_model",
        eval_strategy="epoch",           # replaced 'evaluation_strategy'
        save_strategy="epoch",
        logging_steps=100,
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=CONFIG["per_device_eval_batch_size"],
        num_train_epochs=CONFIG["num_train_epochs"],
        fp16=False,  # For MPS, keep it False
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=CONFIG["seed"]
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        tokenizer=tokenizer  # Warning about deprecation, but still works fine
    )

    # Hyperparameter Search
    print("[INFO] Starting hyperparameter search...")
    best_run = trainer.hyperparameter_search(
        hp_space=hp_space,
        direction="maximize",
        n_trials=CONFIG["n_trials"],
        backend="optuna"
    )
    print("[INFO] Hyperparameter Search complete.")
    print("[INFO] Best Hyperparameters:", best_run.hyperparameters)

    for param, value in best_run.hyperparameters.items():
        setattr(trainer.args, param, value)

    # Final Training
    print("[INFO] Starting final training with best hyperparameters...\n")
    trainer.train()
    print("\n[INFO] Training complete.\n")

    # Evaluate on Training & Validation
    label_names = ["NOT_A_DEFICIENCY", "LOW", "MEDIUM", "HIGH"]

    train_preds_output = trainer.predict(train_dataset)
    train_preds = np.argmax(train_preds_output.predictions, axis=-1)
    train_labels = train_df["labels"].values
    print("\n[TRAIN SET Classification Report]")
    print(classification_report(train_labels, train_preds, target_names=label_names, zero_division=0))

    val_preds_output = trainer.predict(val_dataset)
    val_preds = np.argmax(val_preds_output.predictions, axis=-1)
    val_labels = val_df["labels"].values
    print("\n[VALIDATION SET Classification Report]")
    print(classification_report(val_labels, val_preds, target_names=label_names, zero_division=0))

    eval_results = trainer.evaluate()
    print("\n[INFO] Validation Metrics:", eval_results)

    # Save Model
    trainer.save_model("./final_model")
    tokenizer.save_pretrained("./final_model")
    print("[INFO] Final model and tokenizer saved to './final_model'.")

    # Predict on Test if available
    if os.path.exists(test_file):
        test_df = pd.read_csv(test_file)
        if "def_text" in test_df.columns and "age" in test_df.columns:
            print(f"[INFO] Generating predictions on test file '{test_file}'...")
            test_df, _, _ = preprocess_data(
                test_df, scaler=scaler, is_train=False, require_label=False, extra_categorical=["VesselGroup"]
            )
            if len(test_df) == 0:
                print("[WARNING] Test file is empty after preprocessing. Skipping predictions.")
                return
            test_dataset = DeficiencyDataset(test_df, tokenizer, CONFIG["max_length"], numeric_cols)
            test_preds_output = trainer.predict(test_dataset)
            test_preds = np.argmax(test_preds_output.predictions, axis=-1)
            id2label = {0: "NOT_A_DEFICIENCY", 1: "LOW", 2: "MEDIUM", 3: "HIGH"}
            test_df["predicted_severity"] = [id2label[p] for p in test_preds]
            out_file = "psc_severity_test_with_predictions.csv"
            test_df.to_csv(out_file, index=False)
            print(f"[INFO] Test predictions saved to '{out_file}'.")
        else:
            print("[WARNING] Test file missing 'def_text' or 'age'. Skipping prediction.")
    else:
        print("[INFO] No test file found. Skipping test prediction.")

if __name__ == "__main__":
    main()

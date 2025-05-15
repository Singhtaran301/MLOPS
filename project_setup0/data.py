import torch
import datasets
import pytorch_lightning as pl

from datasets import load_dataset
from transformers import AutoTokenizer


class DataModule(pl.LightningDataModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.model_name = model_name
        self.tokenizer = None

    def prepare_data(self):
        # Download tokenizer and dataset (no state assignment here)
        AutoTokenizer.from_pretrained(self.model_name)
        load_dataset("glue", "cola")

    def tokenize_data(self, example):
        return self.tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=512
        )

    def setup(self, stage=None):
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load and prepare datasets
        cola_dataset = load_dataset("glue", "cola")
        
        # we set up only relevant datasets when stage is specified
        if stage == "fit" or stage is None:
            self.train_data = cola_dataset["train"].map(
                self.tokenize_data, 
                batched=True
            )
            self.train_data.set_format(
                type="torch", 
                columns=["input_ids", "attention_mask", "label"]
            )

            self.val_data = cola_dataset["validation"].map(
                self.tokenize_data, 
                batched=True
            )
            self.val_data.set_format(
                type="torch", 
                columns=["input_ids", "attention_mask", "label"]
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, 
            batch_size=self.batch_size, 
            shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, 
            batch_size=self.batch_size, 
            shuffle=False
        )


if __name__ == "__main__":
    data_model = DataModule()
    data_model.prepare_data()
    data_model.setup()
    print(next(iter(data_model.train_dataloader()))["input_ids"].shape)
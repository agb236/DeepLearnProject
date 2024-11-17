import os
import librosa
import pandas as pd
import numpy as np
import json
import torch
import torchaudio
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from IPython.display import Audio
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForSequenceClassification
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
from torch.optim import AdamW

class AudioMNISTDataset(Dataset):
    def __init__(self, root_dir, processor, split="train", test_size=0.2, val_size=0.1, target_sample_rate=16000, random_seed=42):
        """
        Args:
            root_dir (str): Root directory for AudioMNIST (e.g., '/AudioMNIST/data/').
            processor (Wav2Vec2Processor): Processor for Wav2Vec2 model.
            split (str): One of "train", "val", or "test".
            test_size (float): Proportion of the data to use for testing.
            val_size (float): Proportion of the training data to use for validation.
            target_sample_rate (int): Sample rate required by Wav2Vec2.
            random_seed (int): Seed for reproducibility of splits.
        """
        self.root_dir = root_dir
        self.processor = processor
        self.target_sample_rate = target_sample_rate
        self.split = split
        self.audio_paths = []
        self.labels = []

        # Load all file paths and labels
        speaker_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        all_files = []
        all_labels = []

        for speaker_id in speaker_dirs:
            speaker_dir = os.path.join(root_dir, speaker_id)
            for filename in os.listdir(speaker_dir):
                if filename.endswith(".wav"):
                    # Extract digit from filename format '{digit}_{speaker_id}_{sample_no}.wav'
                    digit = int(filename.split("_")[0])
                    filepath = os.path.join(speaker_dir, filename)
                    all_files.append(filepath)
                    all_labels.append(digit)

        # Split into train, val, test based on speakers for non-overlapping splits
        train_files, test_files, train_labels, test_labels = train_test_split(
            all_files, all_labels, test_size=test_size, random_state=random_seed, stratify=all_labels
        )
        train_files, val_files, train_labels, val_labels = train_test_split(
            train_files, train_labels, test_size=val_size, random_state=random_seed, stratify=train_labels
        )

        
        # Assign split-specific files and labels
        if split == "train":
            self.audio_paths, self.labels = train_files, train_labels
        elif split == "val":
            self.audio_paths, self.labels = val_files, val_labels
        elif split == "test":
            self.audio_paths, self.labels = test_files, test_labels
        else:
            raise ValueError("split should be one of 'train', 'val', or 'test'.")

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        # Load the audio file
        audio_path = self.audio_paths[idx]
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if necessary
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        # Process waveform with Wav2Vec2 processor
        inputs = self.processor(waveform.squeeze().numpy(), sampling_rate=self.target_sample_rate, return_tensors="pt")
        
        # Retrieve the input values and add label
        input_values = inputs.input_values.squeeze()
        label = torch.tensor(self.labels[idx])
        
        return input_values, label
        
def create_dataloaders(root_dir, processor, batch_size=16, test_size=0.2, val_size=0.1, random_seed=42):
    """
    Create DataLoaders for the train, validation, and test splits of the AudioMNIST dataset.
    
    Args:
        root_dir (str): Root directory containing the AudioMNIST data.
        processor (Wav2Vec2Processor): Wav2Vec2 processor.
        batch_size (int): Batch size for DataLoaders.
        test_size (float): Proportion of data to use for testing.
        val_size (float): Proportion of training data to use for validation.
        random_seed (int): Random seed for reproducibility.
    
    Returns:
        dict: Dictionary containing DataLoaders for 'train', 'val', and 'test' splits.
    """
    train_dataset = AudioMNISTDataset(root_dir, processor, split="train", test_size=test_size, val_size=val_size, random_seed=random_seed)
    val_dataset = AudioMNISTDataset(root_dir, processor, split="val", test_size=test_size, val_size=val_size, random_seed=random_seed)
    test_dataset = AudioMNISTDataset(root_dir, processor, split="test", test_size=test_size, val_size=val_size, random_seed=random_seed)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return {"train": train_loader, "val": val_loader, "test": test_loader}

def collate_fn(batch):
    """
    Custom collate function to pad input_values and stack labels for batch processing.
    """
    input_values = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])

    # Pad sequences to the max length in the batch
    input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)

    return input_values, labels




def train_and_validate(model, train_loader, val_loader, optimizer, loss_fn, epochs=3):
    # Training loop
    for epoch in range(epochs):
        model.train()  # Ensure the model is in training mode
        total_loss = 0  # Track total loss for the epoch
        correct_predictions = 0
        total_predictions = 0

        # Iterate through the training data
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            input_values, labels = batch
            input_values = input_values.to(device)
            labels = labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_values, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

        # Print average loss and accuracy for training
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct_predictions / total_predictions * 100
        print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Validation step after each epoch
        model.eval()  # Set the model to evaluation mode
        val_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():  # No need to compute gradients during validation
            for batch in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}"):
                input_values, labels = batch
                input_values = input_values.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(input_values, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(logits, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

                # Print predictions for each file in the batch
                #for i in range(len(labels)):
                #    print(f"Predicted: {predicted[i].item()}, True Label: {labels[i].item()}")

        # Print average validation loss and accuracy
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_predictions / total_predictions * 100
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%\n")



def test_model(model, test_loader):
    # Validation step after each epoch
        model.eval()  # Set the model to evaluation mode
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():  # No need to compute gradients during validation
            for batch in tqdm(test_loader, desc=f"Test"):
                input_values, labels = batch
                input_values = input_values.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(input_values, labels=labels)
                logits = outputs.logits

                # Calculate accuracy
                _, predicted = torch.max(logits, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

        # Print average validation loss and accuracy
        test_accuracy = correct_predictions / total_predictions * 100
        print(f"Test Accuracy: {test_accuracy:.2f}%\n")
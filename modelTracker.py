import numpy as np
import os
import torch
import torch.nn as nn
from collections.abc import Mapping
from typing import Callable, List, Dict, Optional, Any

class ModelTracker:
    def __init__(self, model: nn.Module, layer_names: List[str], custom_select_fns: Optional[List[Callable]] = None):
        """
        Tracks activations and computes metrics for a model.
        
        Args:
            model (nn.Module): The model to track.
            layer_names (List[str]): Names of the layers to hook.
            custom_select_fns (Optional[List[Callable]]): Custom selection functions for each layer.
        """
        self.model = model
        self.layer_names = layer_names
        self.custom_select_fns = custom_select_fns or [lambda x: x for _ in layer_names]
        assert len(self.layer_names) == len(self.custom_select_fns), \
            "Each layer must have a corresponding selection function."

        self.hooks = []
        self.activations = {}
        self.metrics = {}

        self.activate_hooks()


    def _hook_fn(self, layer_name: str, custom_select_fn: Callable):
        """
        Hook function to capture activations and concatenate across batches.
        """
        def hook(module, input, output):
            if isinstance(output, tuple):
                if isinstance(output[0], torch.Tensor):
                    output = output[0]
                elif isinstance(output[1], torch.Tensor):
                    output = output[1]
                elif isinstance(output[2], torch.Tensor):
                    output = output[2]
                else:
                    raise ValueError("Bad format in tuple output.")
            
            # Apply custom selection and mean pooling
            processed_output = custom_select_fn(output.mean(dim=1).detach())
            
            # Concatenate activations across batches
            if layer_name not in self.activations:
                self.activations[layer_name] = processed_output
            else:
                self.activations[layer_name] = torch.cat(
                    (self.activations[layer_name], processed_output), dim=0
                )
        return hook


    def activate_hooks(self):
        """
        Attach hooks to specified layers.
        """
        for layer_name, sel_fn in zip(self.layer_names, self.custom_select_fns):
            layer = dict(self.model.named_modules()).get(layer_name)
            if not layer:
                raise ValueError(f"Layer {layer_name} not found in the model.")
            hook = layer.register_forward_hook(self._hook_fn(layer_name, sel_fn))
            self.hooks.append(hook)

    def deactivate_hooks(self):
        """
        Remove all registered hooks.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def add_metric(self, metric_name: str, metric_fn: Callable):
        """
        Add a metric function to compute on activations.

        Args:
            metric_name (str): Name of the metric.
            metric_fn (Callable): Metric function that takes activations and labels.
        """
        self.metrics[metric_name] = metric_fn

    def compute_metrics(self, label_dict: Mapping[str, Any]):
        """
        Compute all registered metrics on the activations.

        Args:
            label_dict (Mapping): A dictionary mapping layer names to labels.

        Returns:
            Dict: Computed metrics for all layers.
        """
        results = {}
        for metric_name, metric_fn in self.metrics.items():
            results[metric_name] = metric_fn(self.activations, label_dict)
        return results

    def forward_pass(self, dataloader: torch.utils.data.DataLoader, metadata: Optional[Dict[str, Any]] = None):
        """
        Perform a forward pass and collect metadata and labels.

        Args:
            dataloader (torch.utils.data.DataLoader): The data loader.
            metadata (Optional[Dict[str, Any]]): Optional metadata dictionary.

        Returns:
            List[Dict]: List of dictionaries containing label, file path, and optional metadata.
        """
        result = []
        with torch.no_grad():
            for batch in dataloader:
                input_values, labels = batch[:2]  # File paths included in DataLoader
                input_values = input_values.to(next(self.model.parameters()).device)
                labels = labels.to(next(self.model.parameters()).device)

                # Forward pass
                _ = self.model(input_values)

                if metadata:
                    file_paths=batch[2]
                    for label, file_path in zip(labels.cpu().numpy(), file_paths):
                        entry = {
                            "label": int(label),
                            "file_path": file_path,
                        }
                        if metadata:
                            speaker_id = os.path.basename(os.path.dirname(file_path))
                            entry["metadata"] = metadata.get(speaker_id, {})
                            entry["speaker_id"] = speaker_id
                        result.append(entry)
                else:
                    result=None
        return result

''' Examples:

from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForSequenceClassification
from functions import AudioMNISTDataset, create_dataloaders, collate_fn
import json

# Load the metadata
with open("audioMNIST_meta.json", "r") as f:
    metadata = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

dataloaders = create_dataloaders("AudioMNIST/data/", processor, batch_size=16)

train_loader = dataloaders["train"]
val_loader = dataloaders["val"]
test_loader = dataloaders["test"]

# Load saved model
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=10)
model.load_state_dict(torch.load("seq-class-head.pth", weights_only=True))
model.to(device)
model.eval()


# Initialize the tracker for wav2vec2
layer_names = [
    "wav2vec2.feature_projection",
    "wav2vec2.encoder.layers.0.attention",
    "wav2vec2.encoder.layers.1.attention",
    "wav2vec2.encoder.layers.2.attention",
    "wav2vec2.encoder.layers.3.attention",
    "wav2vec2.encoder.layers.4.attention",
    "wav2vec2.encoder.layers.5.attention",
    "wav2vec2.encoder.layers.6.attention",
    "wav2vec2.encoder.layers.7.attention",
    "wav2vec2.encoder.layers.8.attention",
    "wav2vec2.encoder.layers.9.attention",
    "wav2vec2.encoder.layers.10.attention",
    "wav2vec2.encoder.layers.11.attention",
]

# Initialize tracker
tracker = ModelTracker(model, layer_names)

# Forward pass through the model with the test data
data_dict = tracker.forward_pass(test_loader, metadata)

# Deactivate hooks after forward pass
tracker.deactivate_hooks()

# Stack collected features for each layer
activations = {}
for layer_name in layer_names:
    activations[layer_name] = tracker.activations[layer_name].cpu()

# Extract labels and metadata
labels = [entry['label'] for entry in data_dict]
labels_np = np.array(labels)
metadata_dict = [entry['metadata'] for entry in data_dict]
speaker_ids = [entry['speaker_id'] for entry in data_dict]
sid_np = np.array(speaker_ids)

# Convert activations to NumPy arrays
numpy_activations = {layer: act.numpy() for layer, act in activations.items()}

# Example: Access activations for specific layers
hook0 = numpy_activations["wav2vec2.feature_projection"]
print(f"Hook0 shape: {hook0.shape}")

# Example with ResNet
from torchvision.models import resnet18
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Example with resnet
model = resnet18(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define layers to track
layer_names = [
    "conv1",
    "layer1.0.conv1",
    "layer1.0.conv2",
    "layer2.0.conv1",
    "layer2.0.conv2",
    "layer3.0.conv1",
    "layer3.0.conv2",
    "layer4.0.conv1",
    "layer4.0.conv2",
    "fc"  # Fully connected layer
]

# Initialize the tracker
tracker = ModelTracker(model, layer_names)

# Define a dataset and DataLoader (e.g., ImageNet-like dataset)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = datasets.FakeData(transform=transform, size=100)  # Fake dataset for demonstration
dataloader = DataLoader(dataset, batch_size=16)

# Perform a forward pass and collect activations (no metadata)
data_dict = tracker.forward_pass(dataloader, None)

# Deactivate hooks after the forward pass
tracker.deactivate_hooks()

# Stack collected features for each layer
activations = {}
for layer_name in layer_names:
    activations[layer_name] = tracker.activations[layer_name].cpu()

# Convert activations to NumPy arrays
numpy_activations = {layer: act.numpy() for layer, act in activations.items()}

# Example: Access activations for specific layers
conv1_activations = numpy_activations["conv1"]  # Shape: [num_samples, num_channels, height, width]
fc_activations = numpy_activations["fc"]  # Shape: [num_samples, num_classes]

# Print shapes for verification
print(f"conv1 activations shape: {conv1_activations.shape}")
print(f"fc activations shape: {fc_activations.shape}")
'''
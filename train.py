from dataset import CustomDataset
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import AdamW
import datetime
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from enum import Enum


class ModelName(Enum):
    SWIN_BASE = "microsoft/swin-base-patch4-window7-224"
    RESNET50 = "microsoft/resnet-50"
    VIT_BASE = "google/vit-base-patch16-224"
    EFFICIENTNET_B0 = "efficientnet-b0"




def train(model, train_loader, valid_loader, calib_config, date, device=None, ):
    """
    Train and validate a model.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for the training dataset.
        valid_loader: DataLoader for the validation dataset.
        epochs: Number of training epochs.
        lr: Learning rate for the optimizer.
        device: The device to run the model on (e.g., 'cuda' or 'cpu').
    """
    os.makedirs("temp", exist_ok=True)
    epochs = calib_config["epochs"]
    lr = calib_config["lr"]


    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=calib_config['lr'],
                                                    epochs=calib_config['epochs'],
                                                    steps_per_epoch=len(train_loader),
                                                    pct_start=calib_config['pct_start'],
                                                    # final_div_factor=calib_config['final_div_factor']
                                                    )
    print(f"calib_config:{calib_config}")
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_accuracy = np.inf
    # Training loop
    for epoch in range(epochs):
        # Train phase
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        print(f"\nEpoch {epoch+1}/{epochs}")
        for batch in tqdm(train_loader, desc="Training"):
            # Move data to the device
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(pixel_values=inputs)  # Use the appropriate Swin model input
            logits = outputs.logits

            # Compute the loss
            loss = loss_fn(logits, labels)

            # Backpropagation
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Track loss and accuracy
            running_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

        # Calculate and display training metrics
        avg_loss = running_loss / len(train_loader)
        accuracy = correct_preds / total_preds
        print(f"Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct_preds = 0
        val_total_preds = 0

        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validating"):
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(pixel_values=inputs)
                logits = outputs.logits

                # Compute the loss
                loss = loss_fn(logits, labels)
                val_loss += loss.item()

                # Track accuracy
                _, preds = torch.max(logits, dim=1)
                val_correct_preds += (preds == labels).sum().item()
                val_total_preds += labels.size(0)

        # Calculate and display validation metrics
        avg_val_loss = val_loss / len(valid_loader)
        val_accuracy = val_correct_preds / val_total_preds
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if val_accuracy < best_val_accuracy:
                torch.save(model.state_dict(), f"temp/{model.__class__.__name__}_{date}_best_val_accuracy.pt")


def eval_model(model, test_loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    # Test phase
    model.eval()
    test_loss = 0.0
    test_correct_preds = 0
    test_total_preds = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
          inputs, labels = batch
          inputs = inputs.to(device)
          labels = labels.to(device)

          # Forward pass
          outputs = model(pixel_values=inputs)
          logits = outputs.logits

          # Compute the loss
          loss = loss_fn(logits, labels)
          test_loss += loss.item()

          # Track accuracy
          _, preds = torch.max(logits, dim=1)
          test_correct_preds += (preds == labels).sum().item()
          test_total_preds += labels.size(0)

          # Collect predictions and labels for confusion matrix
          all_preds.extend(preds.cpu().numpy())
          all_labels.extend(labels.cpu().numpy())

    # Calculate and display test metrics
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = test_correct_preds / test_total_preds
    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical", values_format="d")
    plt.title("Confusion Matrix")
    plt.savefig(f"temp/{model.__class__.__name__}_{date}_test_confusion_matrix.png")


if __name__ == '__main__':

    date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print(date)
    calib_config = [{'model_name': ModelName.SWIN_BASE.value, 'batch_size': 32, 'lr' : 1e-3, 'epochs' : 15, 'pct_start': 0.2}]
    print(f"calib_config: {calib_config}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(calib_config["model_name"] )
    model =  AutoModelForImageClassification.from_pretrained(calib_config["model_name"] )
    print("Successfully loaded model and feature extractor")

    # https://huggingface.co/datasets/marmal88/skin_cancer
    ds = load_dataset("marmal88/skin_cancer")
    # Extract data from the DatasetDict
    train_ds = ds['train']
    val_ds = ds['validation']
    test_ds = ds['test']

    label_encoder = LabelEncoder()
    # Fit the label encoder on the unique classes
    dx_classes = train_ds['dx']
    encoded_labels = label_encoder.fit_transform(dx_classes)

    # Map label indices to class names
    class_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    print("Class mapping:")
    for i, label in class_mapping.items():
        print(f"{i}: {label}")

    # Prepare train, validation, and test datasets using the custom dataset class
    train_dataset = CustomDataset(train_ds, class_mapping)
    valid_dataset = CustomDataset(val_ds, class_mapping)
    test_dataset = CustomDataset(test_ds, class_mapping)

    batch_size = calib_config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print(f"Training  {calib_config['model_name']}")

    train(model, train_loader, valid_loader, calib_config, date, device=None)

    print("Evaluating model on test set")

    eval_model(model, test_loader)


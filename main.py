import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from CNN import CNN
from XODataset import XODataset


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(dataloader), correct / len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(dataloader), correct / len(dataloader.dataset)


def visualize_predictions(model_path, test_dataset, device):
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode

    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i, (image, label) in enumerate(test_dataset):
        if i >= 5:
            break
        output = model(image.unsqueeze(0).to(device))
        pred = torch.argmax(output, 1).item()
        axes[i].imshow(image.squeeze(), cmap="gray")
        axes[i].set_title(f"Pred: {'X' if pred == 1 else 'O'}")
        axes[i].axis("off")
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformations (convert to tensor and normalize)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load dataset
    data_dir = "./data/x_o_images"
    labels_file = os.path.join(data_dir, "labels.txt")
    dataset = XODataset(data_dir, labels_file, transform=transform)

    # Split dataset into train (80%), validation (10%), and test (10%)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Model, Loss, Optimizer
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    epochs = 20
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch + 1}: Train Loss {train_loss:.4f}, Acc {train_acc:.4f}, Val Loss {val_loss:.4f}, Acc {val_acc:.4f}")

    # Test the model
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Save the trained model later will be published to hugging face
    model_save_path = os.path.join("./data/model", "cnn_xo_classifier.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")

    # Load saved model and visualize predictions
    visualize_predictions(model_save_path, test_dataset, device)


if __name__ == "__main__":
    main()

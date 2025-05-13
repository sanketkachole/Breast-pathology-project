import torch
import torch.nn as nn
import torch.optim as optim
from model import BreastCancerCalssifer, train_one_epoch, evaluate
from data_loader import get_breakhis_loader

def run_training(root_dir, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # 1. Setup
    device = torch.device(device)
    model = BreastCancerCalssifer().to(device)

    train_loader = get_breakhis_loader(
        root_dir=root_dir,
        magnification='40X',
        batch_size=32,
        shuffle=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Optional: learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_val_acc = 0.0
    patience = 3
    patience_counter = 0
    num_epochs = 20

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, train_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print("Model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        scheduler.step()

if __name__ == '__main__':
    run_training(root_dir='data/BreaKHis_v1/BreaKHis_v1/histology_slides/breast')

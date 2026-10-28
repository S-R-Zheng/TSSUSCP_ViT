import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from model_TSSUSCP_ViT import TSSUSCP_ViT_pretrain



def train_the_model(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(train_loader)
    print(f'epochs:{epoch} avg_loss: {avg_loss}')


if __name__ == "__main__":
    learning_rate = 1e-4
    learning_rate_min = 1e-5

    epochs = 100
    batchsize = 128

    img_size = 128

    num_ops = 2
    magnitude = 9

    in_channel = 3
    num_classes = 1000

    patch_size = (32, 32)
    squeeze_rate = (8, 8)
    embed_channel = 36
    num_heads = 12
    num_encoder_layers=6
    ffn_channel = 144

    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),

        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),

        transforms.RandAugment(num_ops=num_ops, magnitude=magnitude),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_dataset = datasets.ImageFolder(
        root="./ImageNet/train",
        transform=transform_train
    )

 
    train_loader = DataLoader(
        train_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    model = TSSUSCP_ViT_pretrain(
        img_size=img_size,
        patch_size=patch_size,
        squeeze_rate=squeeze_rate,
        in_channel=in_channel,
        embed_channel=embed_channel,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        ffn_channel=ffn_channel,
        num_classes=num_classes,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    scheduler = CosineAnnealingLR(optimizer,T_max=epochs - 1,eta_min=learning_rate_min)

    device = torch.device("cuda")

    model.to(device)
    model.compile(mode="max-autotune",dynamic=False)

    for epoch in range(1, epochs + 1):
        train_the_model(model,train_loader, optimizer,epoch)
        scheduler.step()

    torch.save(model, f"TSSUSCP_ViT_pretrained_{epochs}_ImageNet1K.pth")

    print("Training complete and model saved.")
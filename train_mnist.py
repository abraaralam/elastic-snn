import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn
import torchvision
import torchvision.transforms as transforms

# =============================================================
# Quantized LeNet-style network w/ 8bit weights and
# activations. Using Kernel size 3 with no padding, 
# input(28×28×1)
#  - conv1 (K=3, no pad)   -  [26×26×6]: outer ring not in convolution output - 26x26, 1 channel -> 6
#  - avgpool (2×2)         -  [13×13×6]: pooling halves data to 13x13
#  - conv2 (K=3, no pad)   -  [11×11×16]: outer ring not in convolution output - 11x11, 6 channels -> 16 
#  - avgpool (2×2)         -  [5×5×16]: pooling halves then floors data to 5x5 
#  - flatten               -  [400]: 16×5×5=400 reshaped into one vector
#  - fc1 (400→120)         -  [120]: reduced classification
#  - fc2 (120→10)          -  [10]: final classification between 10 digits
# Produces quant_lenet_mnist_8bit.pth for weight extraction
# Run on FINN docker container for working brevitas + torch environment
# =============================================================

class QuantLeNet(nn.Module):
    def __init__(self, bit_width=8):
        super().__init__()
        self.conv1 = qnn.QuantConv2d(1, 6, kernel_size=3, weight_bit_width=bit_width, bias=False)
        self.act1  = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=False)
        self.pool1 = nn.AvgPool2d(2)

        self.conv2 = qnn.QuantConv2d(6, 16, kernel_size=3, weight_bit_width=bit_width, bias=False)
        self.act2  = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=False)
        self.pool2 = nn.AvgPool2d(2)

        self.fc1 = qnn.QuantLinear(16 * 5 * 5, 120, weight_bit_width=bit_width, bias=False)
        self.act3 = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=False)
        self.fc2 = qnn.QuantLinear(120, 10, weight_bit_width=bit_width, bias=False)

    def forward(self, x):
        x = self.pool1(F.relu(self.act1(self.conv1(x))))
        x = self.pool2(F.relu(self.act2(self.conv2(x))))
        x = x.flatten(1)
        x = F.relu(self.act3(self.fc1(x)))
        x = self.fc2(x)
        return x

# Get activation quantization scales by running a forward pass (weight scales are pulled separately, below)
def extract_scales(model, real_image_batch):
    model.eval()
    with torch.no_grad():
        _ = model(real_image_batch)

    scales = {}
    for name, module in model.named_modules():
        if isinstance(module, qnn.QuantIdentity):
            scale = module.act_quant.scale().item()
            level = 2 ** module.act_quant.bit_width().item()
            scales[name] = {"scale": scale, "level": level}
            print(f"{name}: scale={scale}, level={level}")
    return scales


def train(model, device, epochs=5):
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set  = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_set, batch_size=256)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("training 8-bit QuantLeNet on MNIST START!")
    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = F.cross_entropy(out, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                pred = model(imgs).argmax(dim=1)
                correct += (pred == labels).sum().item()
        acc = 100.0 * correct / len(test_set)
        print(f"Epoch {epoch+1}: test accuracy = {acc:.2f}%")

    torch.save(model.state_dict(), "quant_lenet_mnist_8bit.pth")
    print("Saved weights to quant_lenet_mnist_8bit.pth")

    return test_loader


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuantLeNet(bit_width=8).to(device)

    # Only retrain if asked
    if "--retrain" in sys.argv:
        test_loader = train(model, device)
    else:
        model.load_state_dict(torch.load("quant_lenet_mnist_8bit.pth", map_location=device))
        transform = transforms.Compose([transforms.ToTensor()])
        test_set = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=256)

    model.eval()

    sample_imgs, _ = next(iter(test_loader))
    scales = extract_scales(model, sample_imgs.to(device))

    w1_scale = model.conv1.quant_weight().scale.item()
    w2_scale = model.conv2.quant_weight().scale.item()   
    w3_scale = model.fc1.quant_weight().scale.item()     
    print("conv1 weight scale:", w1_scale)
    print("conv2 weight scale:", w2_scale)
    print("fc1 weight scale:", w3_scale)

    v_thr_conv1 = model.act1.act_quant.scale().item() / w1_scale
    v_thr_conv2 = model.act2.act_quant.scale().item() / w2_scale
    v_thr_fc1   = model.act3.act_quant.scale().item() / w3_scale
    print("V_THR for conv1's PE:", v_thr_conv1)
    print("V_THR for conv2's PE:", v_thr_conv2)
    print("V_THR for fc1's PE:", v_thr_fc1)

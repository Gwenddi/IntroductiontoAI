from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 第一个卷积层，输入通道数为1，输出通道数为32，卷积核大小为3*3，步长为1
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # 第一个卷积层，输入通道数为32，输出通道数为64，卷积核大小为3*3，步长为1
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        # 两层dropout层，丢弃率分别为为0.25和0.5，用于防止过拟合
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # 两层全连接层，分别输入大小为9216，输出大小为128和输入大小为128，输出大小为10
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        #填补模型
        #填补的模型
        # 前向传播过程

        # 第一个卷积层
        x = self.conv1(x)
        x = F.relu(x)

        # 第二个卷积层
        x = self.conv2(x)
        x = F.relu(x)

        # 最大池化层，池化窗口大小为2*2
        x = F.max_pool2d(x, 2)

        # 第一个dropout层
        x = self.dropout1(x)
        # 将张量展开成一维
        x = torch.flatten(x, 1)

        # 第一个全连接层
        x = self.fc1(x)
        x = F.relu(x)

        # 第二个dropout层
        x = self.dropout2(x)
        # 第二个全连接层
        x = self.fc2(x)

        # 使用log_softmax进行分类预测
        output = F.log_softmax(x, dim=1)
        return output


# 模型训练，设置模型为训练模式，加载训练数据，计算损失函数，执行梯度下降
def train(args, model, device, train_loader, optimizer, epoch):
    # 将模型设置为训练模式
    model.train()
    # 遍历训练数据集的每个批次
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据和目标标签移动到指定的设备（如CPU、GPU）
        data, target = data.to(device), target.to(device)
        # 梯度清零进行新的前向传播和反向传播
        optimizer.zero_grad()
        output = model(data) # 前向传播，获得输出

        # 使用负对数似然损失函数计算loss值
        loss = F.nll_loss(output, target)
        # 反向传播，计算梯度
        loss.backward()

        # 更新模型参数，执行优化步骤
        optimizer.step()

        # 每隔一定批次输出训练状态
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                print(1)
                break


# 模型验证，设置模型为验证模式，加载验证数据，计算损失函数和准确率
def test(model, device, test_loader):
    # 设置模型为评估（验证）模式
    model.eval()

    # 初始化测试损失和正确分类的样本数量
    test_loss = 0
    correct = 0

    # 在验证过程中不需要计算梯度，因此使用torch.no_grad()上下文管理器
    with torch.no_grad():
        # 遍历验证数据加载器的每个批次
        for data, target in test_loader:
            # 将输入数据和目标标签移动到指定的设备（如CPU或GPU）
            data, target = data.to(device), target.to(device)

            # 进行前向传播，获取模型的输出
            output = model(data)

            # 计算损失，使用负对数似然损失函数（Negative Log Likelihood Loss）
            test_loss += F.nll_loss(output, target, reduction='sum').item()

            # 获取每个样本的预测值
            pred = output.argmax(dim=1, keepdim=True)

            # 统计正确分类的样本数量
            correct += pred.eq(target.view_as(pred)).sum().item()

    # 计算平均测试损失
    test_loss /= len(test_loader.dataset)

    # 打印测试结果，包括平均损失和准确率
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # 设置训练参数
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)

    # 使用CPU进行训练
    device = torch.device("cpu")

    # 设置训练和测试数据加载器参数
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载MNIST训练和测试数据集
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)

    # 创建训练和测试数据加载器
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # 创建神经网络模型、优化器和学习率调度器
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # 循环训练和测试，然后进行学习率调度
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    # 如果指定了保存模型，则保存当前模型参数
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == '__main__':
    main()

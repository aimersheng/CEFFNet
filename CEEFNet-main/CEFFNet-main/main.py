import os
import math
import sys
import json
import time
import shutil
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2

from utils import RandomErase, AverageMeter, accuracy


class Solver():
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args.device_ids}' if torch.cuda.is_available() else "cpu")

        # 保存配置文件
        config_path = f'{self.args.save_dir}/config.json'
        with open(config_path, 'w') as f:
            json.dump(vars(args), f)

        # 模型初始化
        print(f"=> Creating model: {args.arch}")
        if args.arch == 'mobilenetv2':
            self.model = mobilenet_v2(pretrained=args.pretrained)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = torch.nn.Linear(in_features, args.num_classes)

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.SGD(params, lr=args.lr,
                                   momentum=args.momentum,
                                   weight_decay=args.weight_decay)

        # 加载预训练/恢复模型
        if args.resume:
            if os.path.isfile(args.resume):
                print(f"=> Loading checkpoint '{args.resume}'")
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                self.best_val_acc = checkpoint['best_val_acc']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print(f"=> Loaded checkpoint (epoch {checkpoint['epoch']})")
            else:
                print(f"=> No checkpoint found at '{args.resume}'")

        # 冻结层处理
        if args.freeze_layers:
            for name, param in self.model.named_parameters():
                if "classifier" not in name and "fc" not in name:
                    param.requires_grad_(False)
                else:
                    print(f"Training layer: {name}")

        self.model = self.model.to(self.device)
        self.best_val_acc = 0
        self._prepare_data()

    def _prepare_data(self):
        self.transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        }

    def train(self):
        train_dataset = datasets.ImageFolder(self.args.train_dir, self.transforms['train'])
        val_dataset = datasets.ImageFolder(self.args.val_dir, self.transforms['val'])

        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size,
                                  shuffle=True, num_workers=self.args.nw, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size,
                                shuffle=False, num_workers=self.args.nw, pin_memory=True)

        train_acc_log = []
        val_acc_log = []
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(self.args.epochs):
            self.model.train()
            losses = AverageMeter()
            top1 = AverageMeter()

            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                lr = adjust_learning_rate(self.optimizer, epoch, self.args)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                acc1, _ = accuracy(outputs, targets, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1[0], inputs.size(0))

            val_loss, val_acc = self._validate(val_loader, criterion)

            is_best = val_acc > self.best_val_acc
            self.best_val_acc = max(val_acc, self.best_val_acc)
            self._save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_val_acc': self.best_val_acc,
                'optimizer': self.optimizer.state_dict(),
            }, is_best)

            train_acc_log.append(top1.avg.item())
            val_acc_log.append(val_acc.item())

            print(f'Epoch {epoch + 1}/{self.args.epochs} | '
                  f'Train Loss: {losses.avg:.4f} Acc: {top1.avg:.2f}% | '
                  f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | '
                  f'Best Acc: {self.best_val_acc:.2f}%')

        if self.args.plot_curve:
            self._plot_curves(train_acc_log, val_acc_log)

    def _validate(self, val_loader, criterion):
        self.model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                acc1, _ = accuracy(outputs, targets, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1[0], inputs.size(0))

        return losses.avg, top1.avg

    def _save_checkpoint(self, state, is_best):
        filename = f'{self.args.save_dir}/checkpoint.pth'
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, f'{self.args.save_dir}/model_best.pth')

    def _plot_curves(self, train_log, val_log):
        train_log = [x.item() if isinstance(x, torch.Tensor) else x for x in train_log]
        val_log = [x.item() if isinstance(x, torch.Tensor) else x for x in val_log]

        plt.figure()
        plt.plot(train_log, label='Train Acc')
        plt.plot(val_log, label='Val Acc')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.savefig(f'{self.args.save_dir}/acc_curve.png')
        plt.close()

    def test(self):
        val_dataset = datasets.ImageFolder(self.args.val_dir, self.transforms['val'])
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

        self.model.eval()
        total_time = 0.0
        total_samples = 0

        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(self.device)
                start_time = time.time()
                _ = self.model(inputs)
                end_time = time.time()

                total_time += (end_time - start_time)
                total_samples += 1

        avg_time = total_time / total_samples
        print(f"=> Average Testing Time (ATT) per image: {avg_time:.6f} seconds")


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# 使用示例
class Args:
    arch = 'mobilenetv2'
    device_ids = 0
    num_classes = 30
    pretrained = True
    resume = None
    freeze_layers = False
    save_dir = 'CEFFNet-main/checkpoints'
    batch_size = 32
    nw = 4
    lr = 0.001
    momentum = 0.9
    weight_decay = 5e-4
    epochs = 200
    print_freq = 10
    plot_curve = True
    train_dir = ''
    val_dir = ''


if __name__ == '__main__':
    args = Args()
    solver = Solver(args)
    solver.train()
    solver.test()
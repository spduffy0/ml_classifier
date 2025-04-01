import DataSetImageRenderer as renderer
import torch 
import torchvision as tvision
import torchvision.transforms as ttransf

from NeuralNetwork import NeuralNetwork
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from Trainer import Trainer
from typing import Callable, Optional

class Classifier:
    train_set: CIFAR10
    train_loader: DataLoader
    test_set: CIFAR10
    test_loader: DataLoader

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck')

    def __init__(self):
        
        # Setup the normalized transform data
        transform = ttransf.Compose(
            [ttransf.ToTensor(),
            ttransf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        # Initialize the training data
        self.train_set, self.train_loader = self.initalize_data(
            transform
        )

        # Initialize the testing data
        self.test_set, self.test_loader = self.initalize_data(
            transform,
            is_train_data=False,
        )

    # Initializes the transform data set and loader
    def initalize_data(
        self,
        transform: Optional[Callable],
        batch_size: int=10,
        is_train_data: bool=True
        ):

        # Initializes the data set with CIFAR10
        data_set = tvision.datasets.CIFAR10(
            root='./data',
            train=is_train_data,
            download=True,
            transform=transform
        )

        # Initializes the dataloader with the data set
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )

        return data_set, data_loader

if __name__ == "__main__":
    classifier = Classifier()

    # TODO: Add Args for batch size
    # TODO: Add Args on if to show outputs or not
    renderer.show_example(classifier.train_loader, classifier.classes)
    
    # Train the network
    trainer = Trainer()
    trainer.train(classifier.train_loader)

    
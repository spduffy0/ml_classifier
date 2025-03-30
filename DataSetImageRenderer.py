import matplotlib.pyplot as plt
import numpy as np
import torchvision as tvision
from torch.utils.data import DataLoader

# Render Images
def display_images(img):
    # Unnormalize image data
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Show an example of the training Images
def show_example(trainloader: DataLoader, classes):
    # get some random training images
    data_iter = iter(trainloader)
    images, labels = next(data_iter)

    # show images
    display_images(tvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
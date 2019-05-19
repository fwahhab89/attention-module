import torch

from MODELS import model_resnet
import numpy as np

def load_model(model, model_path):
    model_state = torch.load(model_path)
    model.load_state_dict(model_state, strict=False)
    return model


model = model_resnet.ResidualNet(network_type='ImageNet', depth=50, num_classes=1001, att_type='cbam')

model = load_model(model=model, model_path='../checkpoints/resnet50_cbam.pth')

image = np.array(np.zeros(shape=(1, 3, 224, 224)))
print(image)
image = torch.Tensor(image)
print(image)
output = model(image)
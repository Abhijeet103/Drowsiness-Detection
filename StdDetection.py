import torch
from matplotlib import pyplot as plt
import numpy as np


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

img  = 'IMG_20231021_011031_790.jpg'

result =   model(img)
result.save('output.png')

print(result)
# Display the image
plt.imshow(np.squeeze(result.render()))

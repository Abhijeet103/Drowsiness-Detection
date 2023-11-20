import torch
from matplotlib import pyplot as plt
import numpy as np


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

img  = 'WhatsApp Image 2023-11-08 at 10.21.16_8baf9a4b.jpg'

result =   model(img)
result.save('output.png')

print(result)
# Display the image
plt.imshow(np.squeeze(result.render()))

import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import faiss
import os
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = models.resnet50(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path):
    image = Image.open(image_path)
    if image.mode != 'RGB':
            image = image.convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image)
    return features.squeeze().numpy()


image_features_list = []

lists=os.listdir("Images")
for i in lists:
    image_path=f"Images/{i}"
    image_features_list.append(extract_features(image_path))

features_matrix = np.array(image_features_list).astype('float32')

index = faiss.IndexFlatL2(features_matrix.shape[1])
index.add(features_matrix)

print(f"Number of images in index: {index.ntotal} and dimensions : {index.d}")

query_image_path = "Tiger2.jpg"
query_features = extract_features(query_image_path)

k = 5
D, I = index.search(np.array([query_features]).astype('float32'), k)

imglist=os.listdir("Images")
similar_image=[]

for idx, distance in zip(I[0], D[0]):
    print(f"Similar Image Index: {idx}, Similarity Score: {distance}")

    query_image = Image.open(query_image_path)
    similar_image_path = 'Images/'+imglist[idx]
    similar_image.append(Image.open(similar_image_path) )

fig, axs = plt.subplots(1, 6)
axs[0].imshow(query_image)
axs[0].set_title("Query Image")
axs[1].imshow(similar_image[0])
axs[1].set_title("Most Similar Image")
axs[2].imshow(similar_image[1])
axs[3].imshow(similar_image[2])
axs[4].imshow(similar_image[3])
axs[5].imshow(similar_image[4])
plt.show()

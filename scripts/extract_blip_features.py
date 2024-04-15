print('BLIP Feature Extraction')
print('importing python packages...')
import os
import torch
from PIL import Image
import numpy as np
from numpy import asarray

# https://medium.com/@enrico.randellini/image-and-text-features-extraction-with-blip-and-blip-2-how-to-build-a-multimodal-search-engine-a4ceabf51fbe
from lavis.models import load_model_and_preprocess

import pickle, gzip, json
from tqdm import tqdm


# Set filepaths
shapenet_images_path = './data/shapenet-images/screenshots'
ann_files = ["train.json", "val.json", "test.json"]
folds = './amt/folds_adversarial'

keys = os.listdir(shapenet_images_path)

# Load pre-trained BLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
# clip_model, preprocess = clip.load("ViT-B/32", device=device)
print('downloading models and preprocessors...')
blip_model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)


# Extract BLIP visual features
data = {}
print('extracting visual features...')
for key in tqdm(keys):
    pngs = os.listdir(os.path.join(shapenet_images_path, f"{key}"))
    pngs = [os.path.join(shapenet_images_path, f"{key}", p) for p in pngs if "png" in p]
    pngs.sort()

    for png in pngs:
        # im = Image.open(png)
        im = Image.open(png).convert("RGB")
        # image = preprocess(im).unsqueeze(0).to(device)
        image = vis_processors["eval"](im).unsqueeze(0).to(device)

        sample = {"image": image, "text_input": ['placeholder']}
        # image_features = clip_model.encode_image(image).squeeze(0).detach().cpu().numpy()
        image_features = blip_model.extract_features(sample, mode="image").image_embeds[0,0,:] # size (768)
        image_features = image_features.tolist()
        name = png.split('/')[-1].replace(".png", "")

        data[name] = image_features

save_path = './data/shapenet-blipViT32-frames.json.gz'
json.dump(data, gzip.open(save_path,'wt'))

print('visual features saved.')
print('extracting language features...')
# Extract BLIP language features
anns = []
for file in ann_files:
    fname_rel = os.path.join(folds, file)
    print(fname_rel)
    with open(fname_rel, 'r') as f:
        anns = anns + json.load(f)

lang_feat = {}
for d in tqdm(anns):
    ann = d['annotation']

    # text = clip.tokenize([ann]).to(device)
    # feat = clip_model.encode_text(text)
    text_input = txt_processors["eval"](ann)
    sample = {"image": None, "text_input": [text_input]}

    # feat = feat.squeeze(0).detach().cpu().numpy()
    feat = blip_model.extract_features(sample, mode="text").text_embeds[0,0,:] # size (768)
    feat = feat.tolist()
    lang_feat[ann] = feat

save_path = './data/langfeat-512-blipViT32.json.gz'
json.dump(lang_feat, gzip.open(save_path,'wt'))
print('langauge features saved.')
print('feature extraction complete.')
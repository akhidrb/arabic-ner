import time

import pytesseract
import torch
from PIL import Image
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline

from helpers import split_sentences

# extract text from image
image = Image.open('files/cr-1.png')
text = pytesseract.image_to_string(image, lang='ara')

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("hatmimoha/arabic-ner")
model = AutoModelForTokenClassification.from_pretrained("hatmimoha/arabic-ner")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)

# Tag the text
start_time = time.time()
sentences = split_sentences(text)

annotations = nlp(sentences)

entities = []
tags = []
for sentence in annotations:
  for item in sentence:
    if item["word"].startswith("##"):
      entities[-1] = entities[-1] + item["word"].replace("##", "")
    else:
      entities.append(item["word"])
      tags.append(item["entity"])

# get persons from item tags

item_labels = []
for item, label in zip(entities, tags):
  if "B-PERSON" in label:
    item_labels.append(item)
  elif "I-PERSON" in label:
    item_labels[-1] += f" {item}"

names = ''
for item in item_labels:
  names += f"{item}\n"

output_file_path = 'output.txt'
with open(output_file_path, 'w', encoding='utf-8') as file:
  file.write(names)

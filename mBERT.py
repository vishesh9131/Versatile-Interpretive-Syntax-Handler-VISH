from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
import torch
import torch.nn.functional as F

# Load the fine-tuned model and tokenizer from the local directory
tokenizer = MobileBertTokenizer.from_pretrained("./local_mobilebert")
model = MobileBertForSequenceClassification.from_pretrained("./local_mobilebert")

# Tokenize input
inputs = tokenizer("Hello, how are you?", return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Apply softmax to logits to get probabilities
probabilities = F.softmax(logits, dim=1)

# Get the predicted class
predicted_class = torch.argmax(probabilities, dim=1)

print("Logits:", logits)
print("Probabilities:", probabilities)
print("Predicted class:", predicted_class)
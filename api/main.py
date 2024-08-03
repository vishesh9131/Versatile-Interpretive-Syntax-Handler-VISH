from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch.nn as nn
import json
import torch
import numpy as np

# Define the TransformerEncoder and other necessary classes
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x):
        attn_output, _ = self.att(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_length):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, max_length)
        self.transformer_encoder = TransformerEncoder(embed_dim=embedding_dim, num_heads=8, ff_dim=512)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim) -> (seq_len, batch_size, embed_dim)
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0)  # (seq_len, batch_size, embed_dim) -> (batch_size, embed_dim, seq_len)
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.fc(x)
        return self.softmax(x)

class Tokenizer:
    def __init__(self):
        self.word_index = {}
        self.index_word = {}
        self.num_words = 0

    def fit_on_texts(self, texts):
        word_freq = {}
        for text in texts:
            words = text.split()
            for word in words:
                if word not in word_freq:
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        self.word_index = {word: idx + 1 for idx, (word, _) in enumerate(sorted_words)}
        self.index_word = {idx: word for word, idx in self.word_index.items()}
        self.num_words = len(self.word_index) + 1

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequences.append([self.word_index.get(word, 0) for word in text.split()])
        return sequences

def pad_sequences(sequences, maxlen, padding='post'):
    padded_sequences = np.zeros((len(sequences), maxlen), dtype=int)
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            padded_sequences[i] = seq[:maxlen]
        else:
            if padding == 'post':
                padded_sequences[i, :len(seq)] = seq
            elif padding == 'pre':
                padded_sequences[i, -len(seq):] = seq
    return padded_sequences

def generate_text(model, tokenizer, seed_text, max_length, num_words, device, temperature=1.0, top_p=0.9):
    model.eval()
    seed_sequence = tokenizer.texts_to_sequences([seed_text])[0]
    generated_text = seed_text

    for _ in range(num_words):
        padded_sequence = pad_sequences([seed_sequence], maxlen=max_length, padding='post')
        padded_sequence = torch.tensor(padded_sequence, dtype=torch.long).to(device)
        with torch.no_grad():
            predicted_probs = model(padded_sequence).cpu().numpy()[0]

        # Apply temperature
        predicted_probs = np.log(predicted_probs + 1e-9) / temperature
        predicted_probs = np.exp(predicted_probs) / np.sum(np.exp(predicted_probs))

        # Top-p (nucleus) sampling
        sorted_indices = np.argsort(predicted_probs)[::-1]
        cumulative_probs = np.cumsum(predicted_probs[sorted_indices])
        top_p_indices = sorted_indices[cumulative_probs <= top_p]
        if len(top_p_indices) == 0:
            top_p_indices = sorted_indices[:1]
        top_p_probs = predicted_probs[top_p_indices]
        top_p_probs = top_p_probs / np.sum(top_p_probs)
        predicted_word_index = np.random.choice(top_p_indices, p=top_p_probs)

        predicted_word = tokenizer.index_word.get(predicted_word_index, '')

        if predicted_word == '':
            break

        seed_sequence.append(predicted_word_index)
        seed_sequence = seed_sequence[1:]
        generated_text += ' ' + predicted_word

    return generated_text

# Load the tokenizer
with open('tokenizer.json', 'r') as f:
    word_index = json.load(f)
tokenizer = Tokenizer()
tokenizer.word_index = word_index
tokenizer.index_word = {v: k for k, v in word_index.items()}

# Model parameters
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 96
max_length = 100

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerModel(vocab_size, embedding_dim, max_length).to(device)
model.load_state_dict(torch.load('VISH.pth', map_location=device))
model.eval()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class TextGenerationRequest(BaseModel):
    seed_text: str
    num_words: int
    temperature: float
    top_p: float

@app.post("/generate_text/")
def generate_text_endpoint(request: TextGenerationRequest):
    try:
        generated_text = generate_text(model, tokenizer, request.seed_text, max_length, request.num_words, device, request.temperature, request.top_p)
        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
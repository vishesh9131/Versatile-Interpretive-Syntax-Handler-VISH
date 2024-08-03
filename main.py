import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import re
import os
import json

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

class TextDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def build_model(vocab_size, embedding_dim, max_length):
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

    return TransformerModel(vocab_size, embedding_dim, max_length)

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    text = text.split('\n')
    return text

def preprocess_data(text, max_length):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences, tokenizer

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

def main():
    filepath = 'data_1.txt'
    text = load_data(filepath)
    random.shuffle(text)
    split_idx = int(0.8 * len(text))
    train_text, test_text = text[:split_idx], text[split_idx:]

    max_length = 100

    # Load the tokenizer if it exists
    if os.path.exists('tokenizer.json'):
        with open('tokenizer.json', 'r') as f:
            word_index = json.load(f)
        tokenizer = Tokenizer()
        tokenizer.word_index = word_index
        tokenizer.index_word = {v: k for k, v in word_index.items()}
        X_train_pad = pad_sequences(tokenizer.texts_to_sequences(train_text), maxlen=max_length, padding='post')
        X_test_pad = pad_sequences(tokenizer.texts_to_sequences(test_text), maxlen=max_length, padding='post')
    else:
        X_train_pad, tokenizer = preprocess_data(train_text, max_length)
        X_test_pad, _ = preprocess_data(test_text, max_length)
        # Save the tokenizer
        with open('tokenizer.json', 'w') as f:
            json.dump(tokenizer.word_index, f)

    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 96

    y_train = np.random.randint(vocab_size, size=len(X_train_pad))
    y_test = np.random.randint(vocab_size, size=len(X_test_pad))

    train_dataset = TextDataset(X_train_pad, y_train)
    test_dataset = TextDataset(X_test_pad, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(vocab_size, embedding_dim, max_length).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    model_path = 'transformer_model.pth'
    if os.path.exists(model_path):
        # Load the model if it exists
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("Model loaded from disk.")
    else:
        # Train the model if it doesn't exist
        for epoch in range(num_epochs):
            model.train()
            for sequences, labels in train_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

        # Save the model
        torch.save(model.state_dict(), model_path)
        print("Model saved to disk.")

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            test_loss += criterion(outputs, labels).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

    seed_text = "Once upon a time"
    num_words = 100
    generated_text = generate_text(model, tokenizer, seed_text, max_length, num_words, device)
    print(generated_text)

if __name__ == "__main__":
    main()
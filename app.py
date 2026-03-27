import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F

# --- 1. ARHITEKTURA TVOG AI MODELA ---
class OgnjenAIModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Embedding sloj - pretvara slova u brojeve koje AI razume
        self.token_embedding_table = nn.Embedding(vocab_size, 32)
        # Linearni sloj - predviđa sledeće slovo
        self.lm_head = nn.Linear(32, vocab_size)

    def forward(self, idx):
        logits = self.token_embedding_table(idx)
        logits = self.lm_head(logits)
        return logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(idx)
            logits = logits[:, -1, :] # Gledamo samo poslednji karakter
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- 2. PODACI ZA TRENING (Ovde ga učiš šta da priča) ---
trening_tekst = "ognjen pravi najaci ai 2026. kodiranje je buducnost. roblox studio i python su zakon."
chars = sorted(list(set(trening_tekst)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# --- 3. STREAMLIT INTERFEJS ---
st.set_page_config(page_title="OGNJEN AI 0", page_icon="🧠")

st.markdown("""
    <style>
    .main { background-color: #000; color: #00ff00; }
    h1 { font-family: 'Courier New'; text-shadow: 0 0 10px #00ff00; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 OGNJEN-GPT-0 (Custom Architecture)")
st.write("Ovaj AI ne koristi Google ili OpenAI. On je istreniran od nule u ovom kodu.")

# Inicijalizacija modela
model = OgnjenAIModel(vocab_size)

# Simulacija kratkog treninga (da profesorka vidi proces)
if st.button("ISTRENIRAJ MOJ AI"):
    with st.status("🧠 Neuronska mreža uči (Backpropagation)..."):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        # Mali trening loop
        input_data = torch.tensor([encode(trening_tekst[:8])], dtype=torch.long)
        for i in range(100):
            logits = model(input_data)
            # Ovde se dešava matematika (Loss function)
            loss = F.cross_entropy(logits.view(-1, vocab_size), input_data.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    st.success(f"Trening završen! Loss (greška): {loss.item():.4f}")

# Generisanje teksta
st.markdown("---")
st.subheader("Generisanje rečenice:")
start_char = st.selectbox("Izaberi početno slovo:", chars)

if st.button("NEKA AI PROGOVORI"):
    context = torch.tensor([[stoi[start_char]]], dtype=torch.long)
    generisano = decode(model.generate(context, max_new_tokens=30)[0].tolist())
    st.code(generisano, language="text")

import streamlit as st
from transformers import pipeline
import torch

# Podešavanje stranice da izgleda profesionalno
st.set_page_config(page_title="AI Code Master", page_icon="⚡", layout="centered")

# CSS za malo neon stila (da impresioniraš informatičarku)
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    h1 { color: #00ffcc; text-shadow: 0 0 10px #00ffcc; }
    .stButton>button { background-color: #00ffcc; color: black; border-radius: 10px; font-weight: bold; }
    </style>
    """, unsafe_allow_name=True)

# Funkcija za učitavanje modela (keširamo je da ne koči aplikaciju)
@st.cache_resource
def load_ai():
    # Koristimo 'distilgpt2' jer je lagan i neće srušiti besplatni server
    return pipeline("text-generation", model="distilgpt2")

st.title("⚡ AI CODE MASTER PRO")
st.subheader("Cloud-Native Generator Koda")

# Input polje
prompt = st.text_area("Unesi početak koda (npr. # Python function to add numbers):", height=150)

if st.button("GENERISI KOD"):
    if prompt:
        with st.spinner('AI procesira na Cloud serveru...'):
            generator = load_ai()
            # Generisanje rezultata
            output = generator(prompt, max_length=100, num_return_sequences=1, truncation=True)
            
            st.markdown("### Rezultat:")
            st.code(output[0]['generated_text'], language='python')
    else:
        st.warning("Prvo ukucaj nešto!")

st.sidebar.markdown("---")
st.sidebar.write("👤 **Developer:** [ognjen]")
st.sidebar.write("🖥️ **Server:** Streamlit Cloud (CPU Basic)")
st.sidebar.write("🧠 **Model:** GPT-2 Distilled")

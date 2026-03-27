import streamlit as st
from transformers import pipeline

# Podešavanje stranice
st.set_page_config(page_title="AI Code Master", page_icon="⚡")

# POPRAVLJEN CSS DEO
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    h1 { color: #00ffcc; text-shadow: 0 0 10px #00ffcc; }
    </style>
    """, unsafe_allow_html=True)

# Funkcija za učitavanje modela
@st.cache_resource
def load_ai():
    return pipeline("text-generation", model="distilgpt2")

st.title("⚡ AI CODE MASTER PRO")
st.subheader("Cloud-Native Generator Koda")

prompt = st.text_area("Unesi početak koda:", "def pozdrav():", height=150)

if st.button("GENERISI KOD"):
    if prompt:
        with st.spinner('AI razmišlja...'):
            generator = load_ai()
            output = generator(prompt, max_length=100, num_return_sequences=1, truncation=True)
            st.code(output[0]['generated_text'], language='python')
    else:
        st.warning("Prvo ukucaj nešto!")

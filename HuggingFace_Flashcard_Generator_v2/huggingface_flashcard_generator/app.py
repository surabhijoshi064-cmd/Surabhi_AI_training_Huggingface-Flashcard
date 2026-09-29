import streamlit as st
from flashcard_generator import split_facts, load_model, make_card, DEFAULT_MODEL

st.set_page_config(page_title="AI Flashcard Generator", page_icon="🧠", layout="centered")
st.title("🧠 AI Flashcard Generator")
st.caption("Create study flashcards from your notes using a Hugging Face model.")

notes = st.text_area("Paste your notes", height=240, placeholder="Example: CPU is the central processing unit of a computer. It executes instructions...")
count = st.slider("Number of flashcards", 1, 15, 5)
model_name = st.selectbox("Model", [DEFAULT_MODEL, "google/flan-t5-base"])

if st.button("Generate Flashcards", type="primary"):
    if not notes.strip():
        st.warning("Please enter some notes first.")
    else:
        facts = split_facts(notes)
        with st.spinner("Loading model and generating flashcards..."):
            tokenizer, model, device = load_model(model_name)
            cards = []
            for fact in facts[:count]:
                card = make_card(tokenizer, model, device, fact)
                if card:
                    cards.append(card)
        st.success(f"Generated {len(cards)} flashcards on {device.upper()}.")
        for i, (question, answer) in enumerate(cards, 1):
            with st.expander(f"Card {i}: {question}"):
                st.write("**Answer:**", answer)

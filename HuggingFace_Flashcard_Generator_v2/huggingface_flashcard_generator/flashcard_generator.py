import argparse
import csv
import re
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_MODEL = "google/flan-t5-small"


def read_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Input file not found: {path}")
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit("Input file is empty.")
    return text


def split_facts(text: str):
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 15]


def load_model(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model, device


def generate_text(tokenizer, model, device, prompt, max_new_tokens=60):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=4, do_sample=False)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def make_card(tokenizer, model, device, fact):
    q_prompt = (
        "Create one simple study question from the fact below. "
        "Use only the fact. Return only the question.\n\nFACT:\n" + fact
    )
    question = generate_text(tokenizer, model, device, q_prompt, 50)
    question = re.sub(r"^(question|q)\s*:\s*", "", question, flags=re.I).strip()
    if not question:
        return None
    if not question.endswith("?"):
        question += "?"

    a_prompt = (
        "Answer the question using only the fact below. Keep the answer short. "
        "Return only the answer.\n\nFACT:\n" + fact + "\n\nQUESTION:\n" + question
    )
    answer = generate_text(tokenizer, model, device, a_prompt, 60)
    answer = re.sub(r"^(answer|a)\s*:\s*", "", answer, flags=re.I).strip()
    if not answer or len(answer) < 2:
        return None
    return question, answer


def generate_cards(facts, count, model_name):
    tokenizer, model, device = load_model(model_name)
    cards, seen = [], set()
    for fact in facts[:count]:
        card = make_card(tokenizer, model, device, fact)
        if not card:
            continue
        q, a = card
        key = q.lower()
        if key not in seen:
            seen.add(key)
            cards.append((q, a))
    return cards, device


def save_csv(cards, output):
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "Answer"])
        writer.writerows(cards)


def main():
    parser = argparse.ArgumentParser(description="Generate AI flashcards from notes using Hugging Face Transformers.")
    parser.add_argument("input", help="TXT notes file")
    parser.add_argument("-n", "--count", type=int, default=5, help="Number of cards")
    parser.add_argument("-o", "--output", default="flashcards.csv", help="Output CSV")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model name")
    args = parser.parse_args()

    if args.count < 1:
        raise SystemExit("Card count must be at least 1.")

    facts = split_facts(read_text(args.input))
    if not facts:
        raise SystemExit("No usable sentences were found in the notes.")

    print(f"Facts found: {len(facts)}")
    print(f"Model: {args.model}")
    print("Loading model and generating cards...")
    cards, device = generate_cards(facts, min(args.count, len(facts)), args.model)
    save_csv(cards, args.output)

    print(f"Device: {device}")
    print(f"Generated: {len(cards)} flashcards")
    print(f"Saved to: {args.output}")
    for i, (q, a) in enumerate(cards, 1):
        print(f"\nCard {i}\nQ: {q}\nA: {a}")


if __name__ == "__main__":
    main()

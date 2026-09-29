# Hugging Face AI Flashcard Generator — V2

A beginner-friendly BTech AI/ML project that converts study notes into question-answer flashcards using Hugging Face Transformers.

## Features
- Generate flashcards from a TXT file.
- Simple Streamlit web interface.
- CPU and GPU support.
- Choose between FLAN-T5 Small and Base.
- Export flashcards to CSV from the command line.
- Duplicate-question filtering.

## Project structure

```text
HuggingFace_Flashcard_Generator_v2/
└── huggingface_flashcard_generator/
    ├── app.py
    ├── flashcard_generator.py
    ├── notes.txt
    └── requirements.txt
```

## Windows setup

```powershell
cd huggingface_flashcard_generator
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the web app

```powershell
streamlit run app.py
```

## Run from command line

```powershell
python flashcard_generator.py notes.txt -n 5 -o flashcards.csv
```

## How it works

```text
Study Notes
    ↓
Sentence / Fact Extraction
    ↓
Hugging Face FLAN-T5
    ↓
Question Generation
    ↓
Answer Generation
    ↓
Flashcards
    ↓
CSV / Web Display
```

## Technologies
- Python
- Hugging Face Transformers
- FLAN-T5
- PyTorch
- Streamlit

## Notes
The first run downloads the selected model from Hugging Face. Model size affects download time, RAM usage, and generation speed.

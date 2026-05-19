# Career Copilot MVP

This is a Python MVP for two English-only workflows:

- Resume analysis
- Interview preparation and evaluation

## Features

- Upload a job description, resume, and prompt file
- Generate resume analysis reports as Word documents
- Generate likely interview questions as Word documents
- Score answers with editable STAR values
- Calculate average, strong, and weak metrics automatically
- Generate a final interview evaluation report as a Word document

## Run locally

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your API key:

```bash
export OPENROUTER_API_KEY="your_api_key"
```

4. Start the app:

```bash
streamlit run app.py
```

## Notes

- The UI is in English.
- If `OPENROUTER_API_KEY` is set, the app fetches models dynamically from OpenRouter and auto-filters models containing `:free`.
- The first available `:free` model is selected by default, and the UI shows input price, output price, and context length for the selected model.
- You can still use `OPENAI_API_KEY` as a fallback, but dynamic `:free` filtering is only enabled for OpenRouter.
- Prompt files can be uploaded as `DOCX` or `TXT`.
- Resume files can be uploaded as `DOCX`, `PDF`, or `TXT`.

# Agentic Healthcare Assistant for Medical Task Automation

**Applied Generative AI · Course-End Project 2**

An **Agentic AI** virtual medical assistant that autonomously coordinates healthcare
administration. It plans multi-step requests and uses tools to **book appointments**,
**manage patient records**, **retrieve & summarize medical histories**, and **search
trusted medical sources** (MedlinePlus / WHO). Built with a **LangChain tool-calling
agent**, a **FAISS** patient knowledge base (RAG), **conversation memory**, and an
free **Google Gemini** LLM.

## Files

| File | Purpose |
|------|---------|
| `AgenticHealthcareAssistant.ipynb` | Full notebook — all 8 steps, self-contained, runs top-to-bottom. **Primary deliverable.** |
| `healthcare_core.py` | The engine: data layer, 6 agent tools, FAISS RAG, memory, logging, agent, evaluation. Imported by the app. |
| `app.py` | Streamlit dashboard — chat, patient/doctor views, appointment tracking, medical info, evaluation, memory & logs. |
| `Data/doctors.json` | Doctors, specialties, and available slots (the Doctor Schedule). |
| `Data/patients.json` | Patient EHR — structured history, medications, alerts, and unstructured notes. |
| `Data/appointments.json` | Booked appointments (updated in real time by the agent). |
| `Data/tool_logs.json` | Tool-usage log for performance monitoring. |
| `requirements.txt`, `.env.example` | Setup. |

> All patients, doctors, and records are **fictional** sample data for demonstration.

## How the 8 capstone steps map to the code

1. **Agent planning & goal decomposition** — the planning `SYSTEM_PROMPT` + the tool-calling agent (`build_agent`) break a request into ordered sub-goals.
2. **Tool & memory setup** — six tools (appointment booking, EHR management, patient RAG, medical search), a **FAISS** vector store over patient summaries, and **ConversationBufferMemory** for long-term patient context.
3. **Prompt engineering & task chaining** — the system prompt plus each tool's docstring guide summarization, planning, and action-triggering; patient context enters via memory.
4. **Agent execution flow** — `run_agent` on the sample scenario (book a nephrologist for a CKD patient + summarize treatments), exposing the full plan/tool trace.
5. *(covered within 1–4 as the execution flow)*
6. **Model evaluation & monitoring** — `evaluate_summaries` uses **QAEvalChain**; `performance_metrics` reports booking success rate and per-tool success from the logs.
7. **Data visualization & UI** — the **Streamlit** dashboard: patient/doctor views, real-time appointment tracking, latest medical info, evaluation metrics.
8. **Memory & logs interface** — the dashboard's Memory & Logs tab shows memory traces, planning/tool traces, and tool success/failure; the scenario button tests different inputs.

## The six agent tools

`find_available_slots` · `book_appointment` · `get_patient_history` ·
`add_or_update_record` · `semantic_patient_search` (FAISS RAG) · `search_medical_info`
(MedlinePlus/WHO, live with offline fallback).

## Setup

Run from inside this folder, in a terminal.

```bash
python3 -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
cp .env.example .env        # then edit .env and set GOOGLE_API_KEY (free: https://aistudio.google.com/apikey)
```

## Run the notebook

Open `AgenticHealthcareAssistant.ipynb` and *Run All*. The agent cells require the API key.

## Run the Streamlit app

```bash
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). Try the **Run sample
scenario** button on the Assistant tab, then watch the Appointments and Memory & Logs tabs
update in real time.

## Medical information search

`search_medical_info` queries the live **MedlinePlus (NLM)** web service and parses the
results. If the network is unavailable (e.g. during grading), it falls back to an offline
cache of trusted-source summaries for common conditions, so the feature always works.

## Safety note

This is an administrative assistant that provides general, trusted-source information — not
personal medical advice or diagnosis. Patient data is fictional and for demonstration only.
# agentic-healthcare-assistant

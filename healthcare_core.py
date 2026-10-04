"""
Agentic Healthcare Assistant - Core Engine
==========================================

Shared logic for the Agentic Healthcare Assistant capstone (Course-End Project 2).
Imported by both the Streamlit app (``app.py``) and mirrored in the notebook.

It provides, mapped to the capstone steps:

  Part 1 - Agentic System Design
    1. Agent planning / goal decomposition  -> build_agent() system prompt + tool-calling agent
    2. Tool & memory setup                   -> 6 tools, FAISS patient vector store, ConversationBufferMemory
    3. Prompt engineering / task chaining     -> SYSTEM_PROMPT, per-tool docstrings
    4. Agent execution flow                   -> run_agent() on multi-step scenarios

  Part 2 - LLMOps
    6. Model evaluation                       -> evaluate_summaries() (QAEvalChain)
       Performance monitoring                 -> tool-usage logging + performance_metrics()
    7/8. Data/UI + memory & logs              -> data accessors used by app.py

LLM backend: Google Gemini (set GOOGLE_API_KEY; a .env file is supported).
The medical-information search uses the live MedlinePlus (NLM) web service with an
offline fallback so it still works without internet (e.g. during grading).
"""

from __future__ import annotations

import os
import re
import json
import html
import datetime as dt
from typing import Dict, List, Optional
from urllib.parse import quote_plus
from urllib.request import urlopen

# ---------------------------------------------------------------------------
# LangChain
# ---------------------------------------------------------------------------
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ===========================================================================
# DATA LAYER  (STEP 2 - EHR / Patient DB + Doctor Schedule)
# ===========================================================================
DATA_DIR = os.environ.get(
    "HEALTHCARE_DATA",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data"),
)


def _path(name: str) -> str:
    return os.path.join(DATA_DIR, name)


def configure(data_dir: str) -> None:
    """Point the engine at a data directory (folder with the JSON files)."""
    global DATA_DIR
    DATA_DIR = data_dir


def _load(name: str):
    with open(_path(name), "r") as f:
        return json.load(f)


def _save(name: str, obj) -> None:
    with open(_path(name), "w") as f:
        json.dump(obj, f, indent=2)


# convenience accessors used by the app / notebook
def get_doctors() -> List[Dict]:
    return _load("doctors.json")


def get_patients() -> List[Dict]:
    return _load("patients.json")


def get_appointments() -> List[Dict]:
    return _load("appointments.json")


def get_patient(patient_id: str) -> Optional[Dict]:
    return next((p for p in get_patients() if p["patient_id"] == patient_id), None)


def patient_summary_text(p: Dict) -> str:
    """Flatten a patient record into one natural-language summary block."""
    hist = "; ".join(f"{h['date']} [{h['type']}] {h['detail']}"
                     for h in p.get("structured_history", []))
    meds = ", ".join(p.get("medications", [])) or "none on file"
    alerts = ", ".join(p.get("alerts", [])) or "none"
    return (
        f"Patient {p['patient_id']} - {p['name']}, {p['age']}yo {p['gender']}. "
        f"Conditions: {', '.join(p.get('conditions', [])) or 'none recorded'}. "
        f"History: {hist}. Current medications: {meds}. "
        f"Alerts: {alerts}. Clinical notes: {p.get('notes', '')}"
    )


# ===========================================================================
# TOOL-USAGE LOGGING  (STEP 6 - performance monitoring)
# ===========================================================================
def log_tool_call(tool_name: str, tool_input: str, success: bool, detail: str = "") -> None:
    try:
        logs = _load("tool_logs.json")
    except Exception:
        logs = []
    logs.append({
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tool": tool_name,
        "input": str(tool_input)[:300],
        "success": bool(success),
        "detail": str(detail)[:300],
    })
    _save("tool_logs.json", logs)


def get_logs() -> List[Dict]:
    try:
        return _load("tool_logs.json")
    except Exception:
        return []


def clear_logs() -> None:
    _save("tool_logs.json", [])


def performance_metrics() -> Dict:
    """Aggregate tool-success and booking metrics from the logs (STEP 6)."""
    logs = get_logs()
    total = len(logs)
    ok = sum(1 for l in logs if l["success"])
    per_tool: Dict[str, Dict[str, int]] = {}
    for l in logs:
        d = per_tool.setdefault(l["tool"], {"calls": 0, "success": 0})
        d["calls"] += 1
        d["success"] += int(l["success"])
    bookings = [l for l in logs if l["tool"] == "book_appointment"]
    booking_ok = sum(1 for l in bookings if l["success"])
    return {
        "total_calls": total,
        "overall_success_rate": round(ok / total, 3) if total else None,
        "per_tool": per_tool,
        "booking_attempts": len(bookings),
        "booking_success": booking_ok,
        "booking_success_rate": round(booking_ok / len(bookings), 3) if bookings else None,
    }


# ===========================================================================
# MEDICAL INFORMATION SEARCH  (Medline / WHO)
# ===========================================================================
_MEDLINE_URL = ("https://wsearch.nlm.nih.gov/ws/query"
                "?db=healthTopics&term={term}&retmax={n}")

# Offline fallback - concise, trusted-source style summaries for common conditions.
_OFFLINE_KB = {
    "chronic kidney disease": (
        "Chronic Kidney Disease (CKD) is the gradual loss of kidney function over time. "
        "Management focuses on slowing progression: control of blood pressure (often with "
        "ACE inhibitors or ARBs), glycemic control in diabetes, and newer SGLT2 inhibitors "
        "which protect the kidneys. Dietary sodium, potassium and protein are often restricted, "
        "and nephrotoxic drugs (e.g. NSAIDs) are avoided. Advanced stages may require dialysis "
        "or transplantation. (Source: MedlinePlus / WHO)"
    ),
    "diabetes": (
        "Type 2 Diabetes is a chronic condition of high blood glucose. Treatment combines "
        "lifestyle change with medications such as metformin, SGLT2 inhibitors and GLP-1 "
        "receptor agonists. Regular monitoring of HbA1c, eyes, feet and kidney function is "
        "recommended to prevent complications. (Source: MedlinePlus / WHO)"
    ),
    "hypertension": (
        "Hypertension (high blood pressure) raises the risk of heart disease, stroke and kidney "
        "damage. First-line treatment includes lifestyle measures and medications such as ACE "
        "inhibitors, ARBs, calcium-channel blockers and thiazide diuretics. (Source: MedlinePlus / WHO)"
    ),
    "coronary artery disease": (
        "Coronary Artery Disease results from atherosclerotic narrowing of the coronary arteries. "
        "Management includes antiplatelets (aspirin), statins, beta-blockers, lifestyle change and, "
        "where needed, revascularization (PCI/stent or bypass). (Source: MedlinePlus / WHO)"
    ),
    "asthma": (
        "Asthma is a chronic inflammatory airway disease. Controller therapy uses inhaled "
        "corticosteroids, often combined with long-acting beta-agonists; short-acting relievers "
        "treat acute symptoms. Avoiding triggers and an action plan are key. (Source: MedlinePlus / WHO)"
    ),
    "hypothyroidism": (
        "Hypothyroidism is an underactive thyroid gland. It is treated with daily levothyroxine "
        "replacement, titrated to normalize TSH, with periodic blood monitoring. (Source: MedlinePlus / WHO)"
    ),
}


def _strip_html(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()


def search_medical_info_raw(query: str, max_results: int = 3) -> Dict:
    """Query the live MedlinePlus web service; fall back to an offline KB.

    Returns {"query", "source", "results": [{title, url, summary}], "live": bool}.
    """
    # --- try live MedlinePlus ---
    try:
        url = _MEDLINE_URL.format(term=quote_plus(query), n=max_results)
        with urlopen(url, timeout=12) as resp:
            xml = resp.read().decode("utf-8", errors="ignore")
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml)
        results = []
        for doc in root.findall(".//document")[:max_results]:
            item = {"title": "", "url": doc.get("url", ""), "summary": ""}
            for c in doc.findall("content"):
                name = c.get("name")
                val = _strip_html("".join(c.itertext()))
                if name == "title":
                    item["title"] = val
                elif name in ("FullSummary", "snippet") and not item["summary"]:
                    item["summary"] = val
            if item["title"]:
                results.append(item)
        if results:
            return {"query": query, "source": "MedlinePlus (NLM) - live",
                    "results": results, "live": True}
    except Exception:
        pass

    # --- offline fallback ---
    q = query.lower()
    for key, text in _OFFLINE_KB.items():
        if key in q or any(w in q for w in key.split()):
            return {"query": query, "source": "Offline trusted-source cache (MedlinePlus/WHO)",
                    "results": [{"title": key.title(), "url": "https://medlineplus.gov/",
                                 "summary": text}], "live": False}
    return {"query": query, "source": "Offline trusted-source cache (MedlinePlus/WHO)",
            "results": [{"title": query.title(), "url": "https://medlineplus.gov/",
                         "summary": ("No cached entry for this term. For authoritative, up-to-date "
                                     "information consult MedlinePlus (medlineplus.gov) or the WHO "
                                     "(who.int).")}], "live": False}


# ===========================================================================
# FAISS PATIENT VECTOR STORE  (STEP 2 - vector DB for patient summaries)
# ===========================================================================
_PATIENT_VS = None  # set by build_patient_vectorstore()


def build_patient_vectorstore(embeddings=None):
    """Embed each patient summary into a FAISS store for semantic retrieval."""
    global _PATIENT_VS
    from langchain_community.vectorstores import FAISS
    if embeddings is None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    docs = [Document(page_content=patient_summary_text(p),
                     metadata={"patient_id": p["patient_id"], "name": p["name"]})
            for p in get_patients()]
    _PATIENT_VS = FAISS.from_documents(docs, embeddings)
    return _PATIENT_VS


# ===========================================================================
# AGENT TOOLS  (STEP 2 - tool setup; STEP 3 - each docstring is a tool prompt)
# ===========================================================================
@tool
def find_available_slots(specialty: str) -> str:
    """Find doctors of a given specialty and their available appointment slots.
    Use this to discover open slots before booking. `specialty` examples:
    'Nephrologist', 'Cardiologist', 'Endocrinologist', 'General Physician'."""
    try:
        docs = [d for d in get_doctors()
                if specialty.lower() in d["specialty"].lower() and d["available_slots"]]
        if not docs:
            log_tool_call("find_available_slots", specialty, False, "no doctors/slots")
            return f"No available doctors found for specialty '{specialty}'."
        lines = [f"{d['name']} ({d['specialty']}, {d['location']}) - slots: "
                 f"{', '.join(d['available_slots'])}" for d in docs]
        log_tool_call("find_available_slots", specialty, True, f"{len(docs)} doctors")
        return "Available options:\n" + "\n".join(lines)
    except Exception as e:
        log_tool_call("find_available_slots", specialty, False, str(e))
        return f"Error finding slots: {e}"


@tool
def book_appointment(patient_id: str, doctor_name: str, slot: str) -> str:
    """Book a medical appointment. Provide the patient_id (e.g. 'P001'), the exact
    doctor_name (e.g. 'Dr. Anita Rao'), and an available slot in 'YYYY-MM-DD HH:MM'
    format. Returns a confirmation with an appointment id, or an error if the slot
    is unavailable."""
    try:
        doctors = get_doctors()
        doc = next((d for d in doctors if d["name"].lower() == doctor_name.lower()), None)
        patient = get_patient(patient_id)
        if not doc:
            log_tool_call("book_appointment", f"{patient_id}/{doctor_name}/{slot}", False, "doctor not found")
            return f"Doctor '{doctor_name}' not found."
        if not patient:
            log_tool_call("book_appointment", f"{patient_id}/{doctor_name}/{slot}", False, "patient not found")
            return f"Patient '{patient_id}' not found."
        if slot not in doc["available_slots"]:
            log_tool_call("book_appointment", f"{patient_id}/{doctor_name}/{slot}", False, "slot unavailable")
            return (f"Slot '{slot}' is not available for {doctor_name}. "
                    f"Available: {', '.join(doc['available_slots'])}")
        appts = get_appointments()
        appt_id = f"A{len(appts) + 1:03d}"
        appt = {
            "appointment_id": appt_id, "patient_id": patient_id,
            "patient_name": patient["name"], "doctor_id": doc["doctor_id"],
            "doctor_name": doc["name"], "specialty": doc["specialty"],
            "slot": slot, "status": "confirmed",
            "booked_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        appts.append(appt)
        _save("appointments.json", appts)
        # remove the slot from the doctor's availability
        doc["available_slots"].remove(slot)
        _save("doctors.json", doctors)
        log_tool_call("book_appointment", f"{patient_id}/{doctor_name}/{slot}", True, appt_id)
        return (f"Appointment {appt_id} confirmed: {patient['name']} with {doc['name']} "
                f"({doc['specialty']}) on {slot} at {doc['location']}.")
    except Exception as e:
        log_tool_call("book_appointment", f"{patient_id}/{doctor_name}/{slot}", False, str(e))
        return f"Error booking appointment: {e}"


@tool
def get_patient_history(patient_id: str) -> str:
    """Retrieve a patient's full medical history (diagnoses, treatments, labs,
    medications and alerts) by patient_id (e.g. 'P001'). Use this before
    summarizing a patient's history or making a clinical-admin decision."""
    try:
        p = get_patient(patient_id)
        if not p:
            log_tool_call("get_patient_history", patient_id, False, "not found")
            return f"No patient found with id '{patient_id}'."
        log_tool_call("get_patient_history", patient_id, True, p["name"])
        return patient_summary_text(p)
    except Exception as e:
        log_tool_call("get_patient_history", patient_id, False, str(e))
        return f"Error retrieving history: {e}"


@tool
def add_or_update_record(patient_id: str, detail: str, record_type: str = "note") -> str:
    """Add or update a patient's medical record. `record_type` is one of
    'diagnosis', 'treatment', 'lab', 'medication', 'alert', or 'note'. `detail`
    is the free-text content to store. Use this to maintain patient history."""
    try:
        patients = get_patients()
        p = next((x for x in patients if x["patient_id"] == patient_id), None)
        if not p:
            log_tool_call("add_or_update_record", f"{patient_id}/{record_type}", False, "not found")
            return f"No patient found with id '{patient_id}'."
        today = dt.datetime.now().strftime("%Y-%m-%d")
        if record_type == "medication":
            p.setdefault("medications", []).append(detail)
        elif record_type == "alert":
            p.setdefault("alerts", []).append(detail)
        elif record_type == "note":
            p["notes"] = (p.get("notes", "") + f" [{today}] {detail}").strip()
        else:  # diagnosis / treatment / lab -> structured history
            p.setdefault("structured_history", []).append(
                {"date": today, "type": record_type, "detail": detail})
        _save("patients.json", patients)
        log_tool_call("add_or_update_record", f"{patient_id}/{record_type}", True, detail[:60])
        return f"Record updated for {p['name']} ({patient_id}): [{record_type}] {detail}"
    except Exception as e:
        log_tool_call("add_or_update_record", f"{patient_id}/{record_type}", False, str(e))
        return f"Error updating record: {e}"


@tool
def semantic_patient_search(query: str) -> str:
    """Semantically search the patient knowledge base (FAISS vector store) for
    patients or history matching a description, e.g. 'patients with kidney disease'
    or 'diabetic patients on metformin'. Returns the most relevant patient summaries."""
    try:
        if _PATIENT_VS is None:
            log_tool_call("semantic_patient_search", query, False, "vector store not built")
            return "Patient vector store is not initialized."
        hits = _PATIENT_VS.similarity_search(query, k=3)
        log_tool_call("semantic_patient_search", query, True, f"{len(hits)} hits")
        return "\n\n".join(f"[{h.metadata['patient_id']}] {h.page_content}" for h in hits)
    except Exception as e:
        log_tool_call("semantic_patient_search", query, False, str(e))
        return f"Error in semantic search: {e}"


@tool
def search_medical_info(query: str) -> str:
    """Search trusted external medical sources (MedlinePlus / WHO) for up-to-date
    disease information, e.g. 'latest treatment for chronic kidney disease'.
    Returns titles, summaries and source links. This is general information, not a
    diagnosis."""
    try:
        res = search_medical_info_raw(query)
        parts = [f"Source: {res['source']}"]
        for r in res["results"]:
            parts.append(f"- {r['title']}: {r['summary'][:600]} ({r['url']})")
        log_tool_call("search_medical_info", query, True,
                      f"live={res['live']}, {len(res['results'])} results")
        return "\n".join(parts)
    except Exception as e:
        log_tool_call("search_medical_info", query, False, str(e))
        return f"Error searching medical info: {e}"


ALL_TOOLS = [
    find_available_slots, book_appointment, get_patient_history,
    add_or_update_record, semantic_patient_search, search_medical_info,
]


# ===========================================================================
# AGENT  (STEP 1 - planning/decomposition; STEP 3 - prompt engineering)
# ===========================================================================
SYSTEM_PROMPT = """You are InsightMed, an autonomous Healthcare Administration Assistant for a clinic's front desk / attendants.

Your job is to coordinate multi-step administrative tasks on behalf of patients and attendants by PLANNING and then using tools.

CAPABILITIES (each backed by a tool):
- Book appointments: discover slots with `find_available_slots`, then confirm with `book_appointment`.
- Manage records: add or update patient history with `add_or_update_record`.
- Retrieve & summarize histories: load a record with `get_patient_history` (and `semantic_patient_search` to find patients), then summarize diagnoses, treatments and alerts in clear language.
- Medical information search: use `search_medical_info` for up-to-date disease info from trusted sources (MedlinePlus/WHO).

HOW TO WORK (planning & goal decomposition):
1. Break the request into ordered sub-goals.
2. Identify the patient and load context FIRST when the task concerns a specific patient.
3. Call the right tool for each sub-goal; never invent patient data, slots, doctors, or appointment ids - get them from tools.
4. For booking, always discover available slots before booking, and use an exact returned slot.
5. After the tools return, synthesize a concise, friendly answer that reports what you did (e.g. the appointment id) and summarizes any medical information clearly.

SAFETY: You provide administrative help and general, trusted-source medical information only - not personal medical advice or diagnosis. When you share disease information, note it is general information and advise consulting the treating physician. Be mindful that patient data is sensitive.
"""


def build_agent(model: str = "gemini-2.5-flash", temperature: float = 0.0,
                with_vectorstore: bool = True, verbose: bool = False):
    """Build the tool-calling agent with memory. Requires GOOGLE_API_KEY.

    Returns (agent_executor, memory)."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain.memory import ConversationBufferMemory

    if with_vectorstore and _PATIENT_VS is None:
        build_patient_vectorstore()

    llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    executor = AgentExecutor(
        agent=agent, tools=ALL_TOOLS, memory=memory,
        verbose=verbose, return_intermediate_steps=True,
        max_iterations=8, handle_parsing_errors=True,
    )
    return executor, memory


def run_agent(executor, user_input: str) -> Dict:
    """Run one turn and return {'output', 'steps': [(tool, input, observation)]}."""
    result = executor.invoke({"input": user_input})
    steps = []
    for action, observation in result.get("intermediate_steps", []):
        steps.append({
            "tool": getattr(action, "tool", "?"),
            "tool_input": getattr(action, "tool_input", ""),
            "observation": str(observation)[:500],
        })
    return {"output": result["output"], "steps": steps}


# ===========================================================================
# EVALUATION  (STEP 6 - QAEvalChain for summaries / search accuracy)
# ===========================================================================
def default_eval_examples() -> List[Dict[str, str]]:
    """Ground-truth QA set to evaluate summary & search quality."""
    return [
        {"query": "What are the main conditions of patient P001?",
         "answer": "Chronic Kidney Disease (stage 3) and hypertension."},
        {"query": "Which medication is patient P002 taking for diabetes?",
         "answer": "Metformin (1000mg twice daily)."},
        {"query": "What is a key treatment to slow chronic kidney disease progression?",
         "answer": "Blood-pressure control with ACE inhibitors/ARBs and SGLT2 inhibitors; avoid NSAIDs."},
        {"query": "What drug class is first-line for coronary artery disease secondary prevention?",
         "answer": "Antiplatelets such as aspirin, plus statins and beta-blockers."},
        {"query": "What is the controller therapy for asthma?",
         "answer": "Inhaled corticosteroids, often combined with a long-acting beta-agonist."},
    ]


def answer_with_tools(query: str) -> str:
    """Deterministic answer path for evaluation: patient questions use the EHR,
    medical questions use the search tool. Avoids spinning up the full agent per item."""
    q = query.lower()
    m = re.search(r"\bp0*\d{1,3}\b", q)
    if m:
        pid = m.group(0).upper()
        return get_patient_history.invoke({"patient_id": pid})
    # medical knowledge question
    res = search_medical_info_raw(query)
    return res["results"][0]["summary"] if res["results"] else ""


def evaluate_summaries(llm=None, examples=None):
    """Grade answers with QAEvalChain. Returns (graded, predictions)."""
    from langchain.evaluation.qa import QAEvalChain
    if llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    if examples is None:
        examples = default_eval_examples()
    predictions = [{"result": answer_with_tools(ex["query"])} for ex in examples]
    eval_chain = QAEvalChain.from_llm(llm)
    graded = eval_chain.evaluate(
        examples, predictions,
        question_key="query", answer_key="answer", prediction_key="result",
    )
    return graded, predictions

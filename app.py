"""
Agentic Healthcare Assistant - Streamlit Dashboard
==================================================

Run with:  streamlit run app.py
Requires GOOGLE_API_KEY in the environment or a .env file.

Implements the capstone's Part 2 UI requirements:
  - Patient and doctor views
  - Real-time appointment tracking
  - Chat with the agent (planning / tool traces = goal decomposition made visible)
  - Summary of latest retrieved medical information
  - Evaluation metrics for model responses and tool success
  - Memory & logs interface (memory traces, tool usage success/failure, scenario tester)
"""

import os
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv

import healthcare_core as hc

load_dotenv()
hc.configure(os.environ.get("HEALTHCARE_DATA",
             os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")))

st.set_page_config(page_title="Agentic Healthcare Assistant", page_icon="🏥", layout="wide")
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]


# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Starting the agent (building FAISS + tools)...")
def get_agent(model: str):
    hc.build_patient_vectorstore()
    executor, memory = hc.build_agent(model=model)
    return executor, memory


# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏥 Healthcare Assistant")
    st.caption("Agentic AI for medical task automation")
    model = st.selectbox("Agent LLM (Groq)",
                         ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"], index=0)

    # Agent runs on Groq (tool-calling); embeddings run on Google Gemini.
    groq_ok = bool(os.environ.get("GROQ_API_KEY"))
    google_ok = bool(os.environ.get("GOOGLE_API_KEY"))
    if groq_ok:
        st.success("GROQ_API_KEY detected (agent).")
    else:
        st.warning("Set GROQ_API_KEY — free at console.groq.com/keys")
        gk = st.text_input("…or paste your Groq API key", type="password")
        if gk:
            os.environ["GROQ_API_KEY"] = gk
            groq_ok = True
    if google_ok:
        st.success("GOOGLE_API_KEY detected (embeddings).")
    else:
        st.warning("Set GOOGLE_API_KEY — free at aistudio.google.com/apikey")
        gok = st.text_input("…or paste your Google API key", type="password")
        if gok:
            os.environ["GOOGLE_API_KEY"] = gok
            google_ok = True
    key_present = groq_ok and google_ok

    st.divider()
    if st.button("🗑️ Reset conversation & logs", use_container_width=True):
        st.session_state.pop("chat", None)
        hc.clear_logs()
        get_agent.clear()
        st.rerun()
    st.caption("Tip: data lives in Data/*.json and updates in real time as the agent acts.")


st.title("Agentic Healthcare Assistant")

doctors = hc.get_doctors()
patients = hc.get_patients()
appts = hc.get_appointments()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Patients", len(patients))
k2.metric("Doctors", len(doctors))
k3.metric("Appointments", len(appts))
open_slots = sum(len(d["available_slots"]) for d in doctors)
k4.metric("Open slots", open_slots)

tabs = st.tabs([
    "💬 Assistant", "👥 Patients & Doctors", "📅 Appointments",
    "🔎 Medical Info", "📊 Evaluation", "🧠 Memory & Logs",
])

# ===========================================================================
# TAB 1 - ASSISTANT (chat + planning/tool trace)
# ===========================================================================
with tabs[0]:
    st.subheader("Chat with the agent")
    st.caption("The agent plans multi-step requests and uses tools. Each reply shows the "
               "tool/planning trace so you can see the goal decomposition.")

    sample = ("My 70-year-old father is patient P001. He has chronic kidney disease. "
              "I want to book a nephrologist for him. Also, can you summarize the latest "
              "treatment methods?")
    c1, c2 = st.columns([3, 1])
    with c2:
        if st.button("▶️ Run sample scenario", use_container_width=True):
            st.session_state["pending"] = sample

    if not key_present:
        st.info("Add your Groq + Google API keys in the sidebar to use the assistant.")
    else:
        agent, memory = get_agent(model)
        if "chat" not in st.session_state:
            st.session_state.chat = []

        for turn in st.session_state.chat:
            with st.chat_message("user"):
                st.markdown(turn["user"])
            with st.chat_message("assistant"):
                st.markdown(turn["answer"])
                if turn["steps"]:
                    with st.expander(f"🧩 Planning & tool trace ({len(turn['steps'])} steps)"):
                        for i, s in enumerate(turn["steps"], 1):
                            st.markdown(f"**{i}. `{s['tool']}`** — input: `{s['tool_input']}`")
                            st.caption(s["observation"][:300])

        prompt = st.chat_input("Ask the assistant…")
        if "pending" in st.session_state:
            prompt = st.session_state.pop("pending")
        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Planning and acting…"):
                    try:
                        out = hc.run_agent(agent, prompt)
                    except Exception as e:
                        out = {"output": f"Error: {e}", "steps": []}
                st.markdown(out["output"])
                if out["steps"]:
                    with st.expander(f"🧩 Planning & tool trace ({len(out['steps'])} steps)"):
                        for i, s in enumerate(out["steps"], 1):
                            st.markdown(f"**{i}. `{s['tool']}`** — input: `{s['tool_input']}`")
                            st.caption(s["observation"][:300])
            st.session_state.chat.append(
                {"user": prompt, "answer": out["output"], "steps": out["steps"]})

# ===========================================================================
# TAB 2 - PATIENTS & DOCTORS
# ===========================================================================
with tabs[1]:
    left, right = st.columns(2)
    with left:
        st.subheader("👤 Patient view")
        pid = st.selectbox("Select patient",
                           [f"{p['patient_id']} — {p['name']}" for p in patients])
        p = hc.get_patient(pid.split(" — ")[0])
        if p:
            st.markdown(f"**{p['name']}** · {p['age']}yo {p['gender']}")
            st.markdown("**Conditions:** " + (", ".join(p.get("conditions", [])) or "—"))
            st.markdown("**Medications:** " + (", ".join(p.get("medications", [])) or "—"))
            alerts = p.get("alerts", [])
            if alerts:
                st.warning("⚠️ " + " · ".join(alerts))
            st.markdown("**History**")
            hist = pd.DataFrame(p.get("structured_history", []))
            if not hist.empty:
                st.dataframe(hist, use_container_width=True, hide_index=True)
            st.markdown("**Clinical notes**")
            st.info(p.get("notes", "—"))
    with right:
        st.subheader("🩺 Doctor view")
        spec = st.selectbox("Filter by specialty",
                            ["All"] + sorted({d["specialty"] for d in doctors}))
        rows = [{"Doctor": d["name"], "Specialty": d["specialty"],
                 "Location": d["location"], "Open slots": len(d["available_slots"])}
                for d in doctors if spec == "All" or d["specialty"] == spec]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown("**Availability**")
        for d in doctors:
            if spec == "All" or d["specialty"] == spec:
                with st.expander(f"{d['name']} — {d['specialty']}"):
                    st.write(", ".join(d["available_slots"]) or "No open slots")

# ===========================================================================
# TAB 3 - APPOINTMENTS (real-time tracking)
# ===========================================================================
with tabs[2]:
    st.subheader("📅 Real-time appointment tracking")
    adf = pd.DataFrame(hc.get_appointments())
    if adf.empty:
        st.info("No appointments yet.")
    else:
        st.dataframe(adf[["appointment_id", "patient_name", "doctor_name",
                          "specialty", "slot", "status", "booked_at"]],
                     use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        adf["specialty"].value_counts().plot(kind="bar", ax=ax, color=PALETTE)
        ax.set_title("Appointments by specialty", fontweight="bold")
        ax.set_ylabel("Count")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

# ===========================================================================
# TAB 4 - MEDICAL INFO (latest retrieved)
# ===========================================================================
with tabs[3]:
    st.subheader("🔎 Medical information search (MedlinePlus / WHO)")
    q = st.text_input("Search a condition or treatment",
                      "latest treatment for chronic kidney disease")
    if st.button("Search", type="primary"):
        with st.spinner("Searching trusted sources…"):
            res = hc.search_medical_info_raw(q)
        st.session_state["last_med"] = res
    res = st.session_state.get("last_med")
    if res:
        tag = "🟢 live" if res["live"] else "🟡 offline cache"
        st.caption(f"Source: {res['source']}  ({tag})")
        for r in res["results"]:
            st.markdown(f"**{r['title']}**")
            st.write(r["summary"])
            if r.get("url"):
                st.caption(r["url"])
        st.info("General information from trusted sources — not a diagnosis. "
                "Consult the treating physician.")

# ===========================================================================
# TAB 5 - EVALUATION
# ===========================================================================
with tabs[4]:
    st.subheader("📊 Model evaluation (QAEvalChain) & module performance")
    ceval, cperf = st.columns(2)
    with ceval:
        st.markdown("**Summary / search accuracy**")
        if not key_present:
            st.info("Add your Groq + Google API keys to run evaluation.")
        elif st.button("Run QAEvalChain evaluation"):
            with st.spinner("Grading answers…"):
                graded, preds = hc.evaluate_summaries()
                examples = hc.default_eval_examples()
            correct = 0
            for ex, pred, g in zip(examples, preds, graded):
                verdict = str(g.get("results", g.get("text", ""))).strip()
                ok = verdict.upper().startswith("CORRECT")
                correct += ok
                with st.expander(f"{'✅' if ok else '❌'} {ex['query']}"):
                    st.write(pred["result"][:400])
                    st.caption(f"Grade: {verdict}")
            st.metric("Accuracy", f"{correct}/{len(examples)} ({correct/len(examples):.0%})")
    with cperf:
        st.markdown("**Tool / module performance**")
        m = hc.performance_metrics()
        st.metric("Total tool calls", m["total_calls"])
        if m["overall_success_rate"] is not None:
            st.metric("Overall success rate", f"{m['overall_success_rate']:.0%}")
        if m["booking_success_rate"] is not None:
            st.metric("Booking success rate",
                      f"{m['booking_success_rate']:.0%} ({m['booking_success']}/{m['booking_attempts']})")
        if m["per_tool"]:
            pt = pd.DataFrame([
                {"tool": k, "calls": v["calls"], "success": v["success"]}
                for k, v in m["per_tool"].items()])
            st.dataframe(pt, use_container_width=True, hide_index=True)

# ===========================================================================
# TAB 6 - MEMORY & LOGS
# ===========================================================================
with tabs[5]:
    st.subheader("🧠 Memory traces & tool logs")
    cmem, clog = st.columns(2)
    with cmem:
        st.markdown("**Conversation memory (long-term patient context)**")
        if key_present and "chat" in st.session_state and st.session_state.chat:
            for turn in st.session_state.chat:
                st.markdown(f"🧑 **User:** {turn['user']}")
                st.markdown(f"🤖 **Assistant:** {turn['answer'][:300]}")
                st.divider()
        else:
            st.caption("No conversation yet — chat in the Assistant tab to populate memory.")
    with clog:
        st.markdown("**Tool usage log (success / failure across tasks)**")
        logs = pd.DataFrame(hc.get_logs())
        if logs.empty:
            st.caption("No tool calls logged yet.")
        else:
            st.dataframe(logs[["timestamp", "tool", "success", "detail"]].iloc[::-1],
                         use_container_width=True, hide_index=True, height=260)
            fig, ax = plt.subplots(figsize=(6, 3))
            g = logs.groupby("tool")["success"].agg(["sum", "count"])
            g["fail"] = g["count"] - g["sum"]
            g[["sum", "fail"]].plot(kind="bar", stacked=True, ax=ax,
                                    color=["#54A24B", "#E45756"])
            ax.set_title("Tool success vs failure", fontweight="bold")
            ax.set_ylabel("Calls"); ax.legend(["success", "failure"])
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

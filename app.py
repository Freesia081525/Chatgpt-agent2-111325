# app.py

import os
import io
import json
import time
import yaml
import base64
import pdfplumber
import streamlit as st
from typing import Optional, Dict, Any, List

# --- Provider Imports ---
import google.generativeai as genai
from openai import OpenAI
from xai_sdk import Client as XAIClient
from xai_sdk.chat import user as xai_user, system as xai_system, image as xai_image

# --- Chain/Templating Imports ---
from jinja2 import Environment, BaseLoader, StrictUndefined

# --- Visualization Imports ---
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import altair as alt


# ==============================================================================
# providers.py Content
# ==============================================================================
class ProviderRouter:
    def __init__(self,
                 google_api_key: Optional[str],
                 openai_api_key: Optional[str],
                 xai_api_key: Optional[str]):
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.xai_api_key = xai_api_key or os.getenv("XAI_API_KEY")

        # Lazy init clients
        self._genai_client = None
        self._openai_client = None
        self._xai_client = None

    def _get_gemini(self):
        if self._genai_client is None:
            if not self.google_api_key:
                raise RuntimeError("Missing GOOGLE_API_KEY/GEMINI_API_KEY.")
            genai.configure(api_key=self.google_api_key)
            self._genai_client = genai
        return self._genai_client

    def _get_openai(self):
        if self._openai_client is None:
            if not self.openai_api_key:
                raise RuntimeError("Missing OPENAI_API_KEY.")
            self._openai_client = OpenAI(api_key=self.openai_api_key)
        return self._openai_client

    def _get_xai(self):
        if self._xai_client is None:
            if not self.xai_api_key:
                raise RuntimeError("Missing XAI_API_KEY.")
            self._xai_client = XAIClient(api_key=self.xai_api_key, timeout=3600)
        return self._xai_client

    def call(
        self,
        provider: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        images: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        start = time.time()
        provider = (provider or "").lower()
        text = ""
        usage = {}

        if provider == "gemini":
            genai_client = self._get_gemini()
            model_obj = genai_client.GenerativeModel(model)
            resp = model_obj.generate_content(
                contents=[{"role": "user", "parts": [{"text": user_prompt}]}],
                system_instruction=system_prompt,
                generation_config={"temperature": temperature, "max_output_tokens": max_tokens}
            )
            text = getattr(resp, "text", "") or ""
            usage = {"provider": "gemini", "model": model}

        elif provider == "openai":
            client = self._get_openai()
            resp = client.chat.completions.create(
                model=model, temperature=temperature, max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            text = resp.choices[0].message.content
            usage = {
                "provider": "openai", "model": model,
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                "total_tokens": getattr(resp.usage, "total_tokens", None)
            }

        elif provider == "grok":
            xai = self._get_xai()
            chat = xai.chat.create(model=model)
            chat.append(xai_system(system_prompt))
            if images:
                for idx, img in enumerate(images):
                    chat.append(xai_user(f"附帶圖片 {idx+1}", xai_image(img)))
            chat.append(xai_user(user_prompt))
            resp = chat.sample()
            text = resp.content
            usage = {"provider": "grok", "model": model}

        else:
            raise ValueError(f"Unsupported provider: {provider}")

        elapsed = time.time() - start
        return {"text": text, "metrics": {"elapsed_sec": elapsed, **usage}}


# ==============================================================================
# chain.py Content
# ==============================================================================
def render_template(template_str: str, context: Dict[str, Any]) -> str:
    env = Environment(loader=BaseLoader(), undefined=StrictUndefined, autoescape=False, trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(template_str or "")
    return template.render(**context)


def run_chain(agents: List[Dict[str, Any]], router, base_context: Dict[str, Any], on_step=None) -> Dict[str, Any]:
    context = {"now_ts": time.time(), "outputs": {}, **base_context}
    trace = []
    for i, agent in enumerate(agents, start=1):
        step = {
            "index": i, "id": agent.get("id"), "name": agent.get("name"),
            "provider": agent.get("provider"), "model": agent.get("model"),
            "status": "running", "error": None, "metrics": {}
        }
        try:
            system_prompt = agent.get("system_prompt", "")
            prompt_template = agent.get("prompt", "")
            temperature = float(agent.get("temperature", 0.2))
            max_tokens = int(agent.get("max_tokens", 2048))

            user_prompt = render_template(prompt_template, context)
            sys_prompt = render_template(system_prompt, context)

            resp = router.call(
                provider=agent.get("provider", "gemini"), model=agent.get("model", "gemini-1.5-flash"),
                system_prompt=sys_prompt, user_prompt=user_prompt,
                temperature=temperature, max_tokens=max_tokens
            )
            text = resp["text"]
            out_key = agent.get("output_key", f"agent_{i}_output")
            context["outputs"][out_key] = text
            step["status"] = "success"
            step["metrics"] = resp.get("metrics", {})
            step["output_key"] = out_key
        except Exception as e:
            step["status"] = "error"
            step["error"] = str(e)
        finally:
            trace.append(step)
            if on_step:
                on_step(step)
        if step["status"] == "error":
            break
    context["trace"] = trace
    return context


# ==============================================================================
# viz.py Content
# ==============================================================================
def render_keyword_bar(keywords: list):
    df = pd.DataFrame(keywords, columns=["keyword"])
    df = df.groupby("keyword").size().reset_index(name="count").sort_values("count", ascending=False)
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('count:Q', title='Frequency'),
        y=alt.Y('keyword:N', sort='-x', title='Keyword'),
        tooltip=['keyword', 'count']
    ).properties(height=400)
    st.altair_chart(chart, use_container_width=True)


def render_similarity_heatmap(matrix, labels):
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    chart = alt.Chart(df.reset_index().melt('index')).mark_rect().encode(
        x=alt.X('index:N', title='Item A'),
        y=alt.Y('variable:N', title='Item B'),
        color=alt.Color('value:Q', scale=alt.Scale(scheme='blues')),
        tooltip=['index', 'variable', 'value']
    ).properties(height=400)
    st.altair_chart(chart, use_container_width=True)


def render_keyword_graph(nodes: list, links: list, height=600):
    g = Network(height=f"{height}px", width="100%", bgcolor="#FFFFFF", font_color="black", directed=False)
    ids = set()
    for n in nodes:
        nid = n.get("id") or n.get("name")
        if nid not in ids:
            ids.add(nid)
            g.add_node(nid, label=n.get("label", nid), title=f"Score: {n.get('score', '')}", value=float(n.get("score", 1) or 1))
    for l in links:
        s, t, w = l.get("source"), l.get("target"), float(l.get("weight", 1) or 1)
        if s in ids and t in ids:
            g.add_edge(s, t, value=w, title=f"Weight: {w}")
    g.force_atlas_2based()
    return g.generate_html()


def wow_status_badge(status: str):
    color = {"success": "#16a34a", "running": "#f59e0b", "error": "#ef4444"}.get(status, "#64748b")
    st.markdown(f'<span style="background:{color};color:white;padding:4px 8px;border-radius:12px;font-weight:600;">{status.upper()}</span>', unsafe_allow_html=True)


# ==============================================================================
# Streamlit App UI Functions
# ==============================================================================

DEFAULT_YAML_PATH = "agents.yaml"

def initialize_session_state():
    """Sets up the default values for the session state."""
    defaults = {
        "ctx": {},
        "agents": [],
        "yaml_text": "",
        "keys": {
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "XAI_API_KEY": os.getenv("XAI_API_KEY"),
        },
        "doc1_text": "", "doc2_text": "",
        "trace": [], "outputs": {}, "results_json": {}, "run_metrics": []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def render_sidebar():
    """Renders the sidebar for API key input."""
    with st.sidebar:
        st.title("Settings")
        st.caption("Agentic AI Document Comparison (Streamlit)")
        st.subheader("API Keys")
        keys = st.session_state.keys
        
        if not keys["GOOGLE_API_KEY"]:
            keys["GOOGLE_API_KEY"] = st.text_input("GOOGLE_API_KEY (Gemini)", type="password")
        else:
            st.success("Gemini: Using environment key")

        if not keys["OPENAI_API_KEY"]:
            keys["OPENAI_API_KEY"] = st.text_input("OPENAI_API_KEY (OpenAI-compatible)", type="password")
        else:
            st.success("OpenAI: Using environment key")

        if not keys["XAI_API_KEY"]:
            keys["XAI_API_KEY"] = st.text_input("XAI_API_KEY (Grok)", type="password")
        else:
            st.success("Grok: Using environment key")
        st.markdown("---")
        st.caption("Models will be selected per-agent in the Pipeline tab.")

def render_documents_tab(tab):
    """Renders the UI for the Documents tab."""
    with tab:
        st.header("Documents")
        col1, col2 = st.columns(2)

        def read_pdf(file) -> str:
            try:
                text_parts = [page.extract_text() or "" for page in pdfplumber.open(file).pages]
                return "\n".join(text_parts)
            except Exception:
                return ""

        with col1:
            st.subheader("Document A")
            uploaded1 = st.file_uploader("Upload PDF/TXT for Document A", type=["pdf", "txt"], key="doc1_upl")
            if uploaded1:
                if uploaded1.name.lower().endswith(".pdf"):
                    st.session_state.doc1_text = read_pdf(uploaded1)
                else:
                    st.session_state.doc1_text = uploaded1.read().decode("utf-8", errors="ignore")
            st.session_state.doc1_text = st.text_area("Or paste text for Document A", st.session_state.doc1_text, height=280)

        with col2:
            st.subheader("Document B")
            uploaded2 = st.file_uploader("Upload PDF/TXT for Document B", type=["pdf", "txt"], key="doc2_upl")
            if uploaded2:
                if uploaded2.name.lower().endswith(".pdf"):
                    st.session_state.doc2_text = read_pdf(uploaded2)
                else:
                    st.session_state.doc2_text = uploaded2.read().decode("utf-8", errors="ignore")
            st.session_state.doc2_text = st.text_area("Or paste text for Document B", st.session_state.doc2_text, height=280)

        ready = bool(st.session_state.doc1_text and st.session_state.doc2_text)
        st.info("Ready for processing." if ready else "Please provide both documents.")

def render_pipeline_tab(tab):
    """Renders the UI for the Pipeline tab."""
    with tab:
        st.header("Agent Pipeline")
        colA, colB = st.columns([2, 1])
        with colA:
            upload_yaml = st.file_uploader("Upload agents.yaml", type=["yaml", "yml"])
            if upload_yaml:
                st.session_state.yaml_text = upload_yaml.read().decode("utf-8")
            elif not st.session_state.agents and os.path.exists(DEFAULT_YAML_PATH):
                with open(DEFAULT_YAML_PATH, "r", encoding="utf-8") as f:
                    st.session_state.yaml_text = f.read()
            
            if st.session_state.yaml_text and not st.session_state.agents:
                 data = yaml.safe_load(st.session_state.yaml_text)
                 st.session_state.agents = data.get("pipeline", [])


            st.write("Active agents:", len(st.session_state.agents))
            for idx, ag in enumerate(st.session_state.agents):
                with st.expander(f"[{idx+1}] {ag.get('name', ag.get('id'))}", expanded=(idx < 3)):
                    prov_options = ["gemini", "openai", "grok"]
                    ag["provider"] = st.selectbox("Provider", prov_options, index=prov_options.index(ag.get("provider", "gemini")), key=f"prov_{idx}")
                    
                    model_options = {
                        "gemini": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"],
                        "openai": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
                        "grok": ["grok-1.5", "grok-1"]
                    }.get(ag["provider"], [])
                    
                    ag["model"] = st.selectbox("Model", model_options, index=0, key=f"model_{idx}")
                    ag["temperature"] = st.slider("Temperature", 0.0, 1.0, float(ag.get("temperature", 0.2)), 0.05, key=f"temp_{idx}")
                    ag["max_tokens"] = st.number_input("Max tokens", 64, 32768, int(ag.get("max_tokens", 2048)), 64, key=f"tok_{idx}")
                    ag["output_key"] = st.text_input("Output key", ag.get("output_key", f"agent_{idx+1}_output"), key=f"outk_{idx}")
                    ag["system_prompt"] = st.text_area("System prompt", ag.get("system_prompt", ""), key=f"sys_{idx}", height=120)
                    ag["prompt"] = st.text_area("User prompt template (Jinja)", ag.get("prompt", ""), key=f"usr_{idx}", height=150)

        with colB:
            st.subheader("YAML")
            st.session_state.yaml_text = yaml.safe_dump({"version": 1, "pipeline": st.session_state.agents}, allow_unicode=True, sort_keys=False)
            st.code(st.session_state.yaml_text, language="yaml")
            st.download_button("Download agents.yaml", data=st.session_state.yaml_text.encode("utf-8"), file_name="agents.yaml", mime="text/yaml")
            if st.button("Save agents.yaml to workspace"):
                with open(DEFAULT_YAML_PATH, "w", encoding="utf-8") as f:
                    f.write(st.session_state.yaml_text)
                st.success("Saved agents.yaml")

def render_run_tab(tab):
    """Renders the UI for the Run tab and executes the pipeline."""
    with tab:
        st.header("Execute")
        keys = st.session_state.keys
        agents = st.session_state.agents
        missing_keys = [
            "GOOGLE_API_KEY" for ag in agents if ag.get("provider") == "gemini" and not keys["GOOGLE_API_KEY"]
        ] + [
            "OPENAI_API_KEY" for ag in agents if ag.get("provider") == "openai" and not keys["OPENAI_API_KEY"]
        ] + [
            "XAI_API_KEY" for ag in agents if ag.get("provider") == "grok" and not keys["XAI_API_KEY"]
        ]

        if set(missing_keys):
            st.error(f"Missing API keys for providers in pipeline: {', '.join(set(missing_keys))}")
        
        if st.button("Run Agent Pipeline", type="primary", disabled=not (st.session_state.doc1_text and st.session_state.doc2_text)):
            st.session_state.trace, st.session_state.outputs, st.session_state.results_json, st.session_state.run_metrics = [], {}, {}, []
            status = st.status("Running agents...", expanded=True)
            router = ProviderRouter(google_api_key=keys["GOOGLE_API_KEY"], openai_api_key=keys["OPENAI_API_KEY"], xai_api_key=keys["XAI_API_KEY"])

            def on_step(step):
                with status:
                    cols = st.columns([0.8, 0.2])
                    cols[0].write(f"[{step['index']}] {step['name']} ({step['provider']}/{step['model']})")
                    with cols[1]:
                        wow_status_badge(step["status"])
                    if step["status"] == "success":
                        st.caption(f"Elapsed: {step['metrics'].get('elapsed_sec', 0):.2f}s")
                    elif step["status"] == "error":
                        st.error(step["error"])
                st.session_state.trace.append(step)
                if step["status"] == "success":
                    st.session_state.run_metrics.append(step["metrics"])

            base_context = {"doc1": st.session_state.doc1_text, "doc2": st.session_state.doc2_text}
            result_ctx = run_chain(st.session_state.agents, router, base_context, on_step=on_step)
            st.session_state.outputs = result_ctx.get("outputs", {})
            status.update(label="Run finished", state="complete", expanded=False)

            final_json_str = st.session_state.outputs.get("final_json", "{}")
            try:
                clean_json_str = final_json_str.strip().removeprefix("```json").removesuffix("```").strip()
                st.session_state.results_json = json.loads(clean_json_str)
                st.success("Parsed final JSON output successfully.")
            except json.JSONDecodeError:
                st.warning("Final output is not valid JSON. You can inspect raw outputs in the YAML tab.")
            st.toast("Pipeline completed")

        st.subheader("Trace")
        for t in st.session_state.trace:
            color = {"success": "green", "error": "red"}.get(t["status"], "orange")
            st.markdown(f"- [{t['index']}] <span style='color:{color}'>{t['status']}</span> — {t['name']} ({t['provider']}/{t['model']})", unsafe_allow_html=True)

def render_dashboard_tab(tab):
    """Renders the UI for the interactive Dashboard tab."""
    with tab:
        st.header("Interactive Dashboard")
        res = st.session_state.results_json or {}
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Summary")
            summary_md = res.get("summary_markdown") or st.session_state.outputs.get("summary_markdown", "")
            st.markdown(summary_md if summary_md else "No summary available yet.")
        
        with col2:
            st.subheader("Key Metrics")
            total_time = sum(m.get("elapsed_sec", 0) for m in st.session_state.run_metrics)
            st.metric("Total elapsed (s)", f"{total_time:.2f}")
            st.metric("Agents executed", len(st.session_state.trace))
            providers = sorted(set(m.get("provider") for m in st.session_state.run_metrics if m.get("provider")))
            st.write("Providers:", ", ".join(providers) if providers else "-")

        st.markdown("---")
        colsB = st.columns(2)
        with colsB[0]:
            st.subheader("Keywords")
            keywords = res.get("keywords")
            if keywords and isinstance(keywords, list):
                render_keyword_bar([k['term'] if isinstance(k, dict) else k for k in keywords])
            else:
                st.info("No keywords available.")
        
        with colsB[1]:
            st.subheader("Similarity Heatmap")
            sim, labels = res.get("similarity_matrix"), res.get("labels")
            if sim and labels:
                render_similarity_heatmap(sim, labels)
            else:
                st.info("No similarity matrix available.")

        st.subheader("Keyword Graph")
        nodes, links = res.get("graph", {}).get("nodes", []), res.get("graph", {}).get("links", [])
        if nodes and links:
            html = render_keyword_graph(nodes, links, height=600)
            st.components.v1.html(html, height=620, scrolling=True)
        else:
            st.info("No graph data available.")

def render_yaml_tab(tab):
    """Renders the UI for the YAML/Raw Outputs tab."""
    with tab:
        st.header("Raw Outputs and YAML")
        st.subheader("Current outputs")
        st.json(st.session_state.outputs or {})
        st.subheader("Final JSON (if any)")
        st.json(st.session_state.results_json or {})
        st.subheader("agents.yaml (editable)")
        edited = st.text_area("Edit YAML", st.session_state.yaml_text, height=400, key="yaml_editor")
        if st.button("Apply YAML"):
            try:
                data = yaml.safe_load(edited)
                st.session_state.agents = data.get("pipeline", [])
                st.session_state.yaml_text = edited
                st.success("YAML applied to pipeline.")
            except Exception as e:
                st.error(f"Invalid YAML: {e}")

# ==============================================================================
# Main Application
# ==============================================================================
def main():
    """The main function that runs the Streamlit application."""
    st.set_page_config(page_title="Agentic AI Document Comparison", layout="wide")
    
    initialize_session_state()
    render_sidebar()

    st.title("Agentic AI Document Comparison System")
    
    doc_tab, pipe_tab, run_tab, dash_tab, yaml_tab = st.tabs(["Documents", "Pipeline", "Run", "Dashboard", "YAML"])
    
    render_documents_tab(doc_tab)
    render_pipeline_tab(pipe_tab)
    render_run_tab(run_tab)
    render_dashboard_tab(dash_tab)
    render_yaml_tab(yaml_tab)

if __name__ == "__main__":
    main()

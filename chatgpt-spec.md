Files
1) requirements.txt
- streamlit
- pyyaml
- jinja2
- google-generativeai
- openai
- xai-sdk
- pandas
- numpy
- altair
- networkx
- pyvis
- pdfplumber
- python-dotenv

2) providers.py
import os
import time
from typing import Optional, Dict, Any, List

# Gemini
import google.generativeai as genai

# OpenAI-compatible
from openai import OpenAI

# Grok (xAI) - sample per user’s request
from xai_sdk import Client as XAIClient
from xai_sdk.chat import user as xai_user, system as xai_system, image as xai_image


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
            genai = self._get_gemini()
            model_obj = genai.GenerativeModel(model)
            # Gemini expects messages as text blocks, system via system_instruction
            resp = model_obj.generate_content(
                contents=[
                    {"role": "user", "parts": [{"text": user_prompt}]}
                ],
                system_instruction=system_prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
            )
            text = getattr(resp, "text", "") or ""
            # Usage fields vary; keep generic
            usage = {"provider": "gemini", "model": model}

        elif provider == "openai":
            client = self._get_openai()
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
            )
            text = resp.choices[0].message.content
            usage = {
                "provider": "openai",
                "model": model,
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                "total_tokens": getattr(resp.usage, "total_tokens", None),
            }

        elif provider == "grok":
            # xAI SDK (sample per user’s spec)
            xai = self._get_xai()
            chat = xai.chat.create(model=model)
            chat.append(xai_system(system_prompt))
            if images:
                # image URLs or base64 URLs supported. Sample:
                # chat.append(xai_user("Describe image", xai_image(images[0])))
                # For this system we mainly pass text; image support if present:
                for idx, img in enumerate(images):
                    chat.append(xai_user(f"附帶圖片 {idx+1}", xai_image(img)))
            chat.append(xai_user(user_prompt))
            resp = chat.sample()
            text = resp.content
            usage = {"provider": "grok", "model": model}

        else:
            raise ValueError(f"Unsupported provider: {provider}")

        elapsed = time.time() - start
        return {
            "text": text,
            "metrics": {
                "elapsed_sec": elapsed,
                **usage
            }
        }

3) chain.py
import time
from typing import Dict, Any, List
import yaml
from jinja2 import Environment, BaseLoader, StrictUndefined


def render_template(template_str: str, context: Dict[str, Any]) -> str:
    env = Environment(loader=BaseLoader(), undefined=StrictUndefined, autoescape=False, trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(template_str or "")
    return template.render(**context)


def run_chain(agents: List[Dict[str, Any]],
              router,
              base_context: Dict[str, Any],
              on_step=None) -> Dict[str, Any]:
    """
    agents: list of agent dicts parsed from YAML.
    router: ProviderRouter instance
    base_context: initial context including doc1_text, doc2_text
    on_step: callback(step_info) for UI updates
    """
    context = {
        "now_ts": time.time(),
        "outputs": {},
        **base_context
    }
    trace = []

    for i, agent in enumerate(agents, start=1):
        step = {
            "index": i,
            "id": agent.get("id"),
            "name": agent.get("name"),
            "provider": agent.get("provider"),
            "model": agent.get("model"),
            "status": "running",
            "error": None,
            "metrics": {}
        }
        try:
            system_prompt = agent.get("system_prompt", "")
            prompt_template = agent.get("prompt", "")
            temperature = float(agent.get("temperature", 0.2))
            max_tokens = int(agent.get("max_tokens", 2048))

            user_prompt = render_template(prompt_template, context)
            sys_prompt = render_template(system_prompt, context)

            resp = router.call(
                provider=agent.get("provider", "gemini"),
                model=agent.get("model", "gemini-2.5-flash"),
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens
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

4) viz.py
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import altair as alt
import streamlit as st


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
    g = Network(height=f"{height}px}", width="100%", bgcolor="#FFFFFF", font_color="black", directed=False)
    ids = set()
    for n in nodes:
        nid = n.get("id") or n.get("name")
        if nid in ids:
            continue
        ids.add(nid)
        g.add_node(nid, label=n.get("label", nid), title=f"Score: {n.get('score', '')}", value=float(n.get("score", 1) or 1))

    for l in links:
        s = l.get("source"); t = l.get("target"); w = float(l.get("weight", 1) or 1)
        if s in ids and t in ids:
            g.add_edge(s, t, value=w, title=f"Weight: {w}")

    g.force_atlas_2based()
    return g.generate_html()


def wow_status_badge(status: str):
    color = {
        "success": "#16a34a",
        "running": "#f59e0b",
        "error": "#ef4444"
    }.get(status, "#64748b")
    st.markdown(f"""
    <span style="background:{color};color:white;padding:4px 8px;border-radius:12px;font-weight:600;">
      {status.upper()}
    </span>
    """, unsafe_allow_html=True)

5) app.py
import os
import io
import json
import time
import yaml
import base64
import pdfplumber
import streamlit as st
from typing import Optional, Dict, Any, List

from providers import ProviderRouter
from chain import run_chain
from viz import render_keyword_bar, render_similarity_heatmap, render_keyword_graph, wow_status_badge


DEFAULT_YAML_PATH = "agents.yaml"

st.set_page_config(page_title="Agentic AI Document Comparison", layout="wide")

# Session state init
if "ctx" not in st.session_state:
    st.session_state.ctx = {}
if "agents" not in st.session_state:
    st.session_state.agents = []
if "yaml_text" not in st.session_state:
    st.session_state.yaml_text = ""
if "keys" not in st.session_state:
    st.session_state.keys = {
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "XAI_API_KEY": os.getenv("XAI_API_KEY"),
    }
if "doc1_text" not in st.session_state:
    st.session_state.doc1_text = ""
if "doc2_text" not in st.session_state:
    st.session_state.doc2_text = ""
if "trace" not in st.session_state:
    st.session_state.trace = []
if "outputs" not in st.session_state:
    st.session_state.outputs = {}
if "results_json" not in st.session_state:
    st.session_state.results_json = {}
if "run_metrics" not in st.session_state:
    st.session_state.run_metrics = []

# Sidebar: Theme + API keys
with st.sidebar:
    st.title("Settings")
    st.caption("Agentic AI Document Comparison (Streamlit)")

    st.subheader("API Keys")
    google_env = bool(st.session_state.keys["GOOGLE_API_KEY"])
    openai_env = bool(st.session_state.keys["OPENAI_API_KEY"])
    xai_env = bool(st.session_state.keys["XAI_API_KEY"])

    if not google_env:
        st.session_state.keys["GOOGLE_API_KEY"] = st.text_input("GOOGLE_API_KEY (Gemini)", type="password")
    else:
        st.success("Gemini: Using environment key")

    if not openai_env:
        st.session_state.keys["OPENAI_API_KEY"] = st.text_input("OPENAI_API_KEY (OpenAI-compatible)", type="password")
    else:
        st.success("OpenAI: Using environment key")

    if not xai_env:
        st.session_state.keys["XAI_API_KEY"] = st.text_input("XAI_API_KEY (Grok)", type="password")
    else:
        st.success("Grok: Using environment key")

    st.markdown("---")
    st.caption("Models will be selected per-agent in the Pipeline tab.")

st.title("Agentic AI Document Comparison System")

tabs = st.tabs(["Documents", "Pipeline", "Run", "Dashboard", "YAML"])

# Documents tab
with tabs[0]:
    st.header("Documents")
    col1, col2 = st.columns(2)

    def read_pdf(file) -> str:
        try:
            text_parts = []
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text_parts.append(page.extract_text() or "")
            return "\n".join(text_parts)
        except Exception:
            return ""

    with col1:
        st.subheader("Document A")
        uploaded1 = st.file_uploader("Upload PDF/TXT for Document A", type=["pdf", "txt"], key="doc1_upl")
        if uploaded1:
            if uploaded1.name.lower().endswith(".pdf"):
                st.session_state.doc1_text = read_pdf(uploaded1)
            elif uploaded1.name.lower().endswith(".txt"):
                st.session_state.doc1_text = uploaded1.read().decode("utf-8", errors="ignore")
        st.session_state.doc1_text = st.text_area("Or paste text for Document A", st.session_state.doc1_text, height=280)

    with col2:
        st.subheader("Document B")
        uploaded2 = st.file_uploader("Upload PDF/TXT for Document B", type=["pdf", "txt"], key="doc2_upl")
        if uploaded2:
            if uploaded2.name.lower().endswith(".pdf"):
                st.session_state.doc2_text = read_pdf(uploaded2)
            elif uploaded2.name.lower().endswith(".txt"):
                st.session_state.doc2_text = uploaded2.read().decode("utf-8", errors="ignore")
        st.session_state.doc2_text = st.text_area("Or paste text for Document B", st.session_state.doc2_text, height=280)

    ready = bool(st.session_state.doc1_text and st.session_state.doc2_text)
    st.info("Ready for processing." if ready else "Please provide both documents.")

# Pipeline tab
with tabs[1]:
    st.header("Agent Pipeline")
    colA, colB = st.columns([2, 1])
    with colA:
        upload_yaml = st.file_uploader("Upload agents.yaml", type=["yaml", "yml"])
        if upload_yaml:
            st.session_state.yaml_text = upload_yaml.read().decode("utf-8")
            data = yaml.safe_load(st.session_state.yaml_text)
            st.session_state.agents = data.get("pipeline", [])
        elif not st.session_state.agents:
            # load default if exists
            if os.path.exists(DEFAULT_YAML_PATH):
                with open(DEFAULT_YAML_PATH, "r", encoding="utf-8") as f:
                    st.session_state.yaml_text = f.read()
                    data = yaml.safe_load(st.session_state.yaml_text)
                    st.session_state.agents = data.get("pipeline", [])

        st.write("Active agents:", len(st.session_state.agents))
        for idx, ag in enumerate(st.session_state.agents):
            with st.expander(f"[{idx+1}] {ag.get('name', ag.get('id'))}", expanded=(idx < 3)):
                ag["provider"] = st.selectbox("Provider", ["gemini", "openai", "grok"], index=["gemini","openai","grok"].index(ag.get("provider","gemini")), key=f"prov_{idx}")
                # Suggested models
                model_options = {
                    "gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
                    "openai": ["gpt-5-nano", "gpt-4o-mini", "gpt-4.1-mini"],
                    "grok": ["grok-4-fast-reaoning", "grok-3-mini"]
                }[ag["provider"]]
                ag["model"] = st.selectbox("Model", model_options, index=min(0, len(model_options)-1), key=f"model_{idx}")
                ag["temperature"] = st.slider("Temperature", 0.0, 1.0, float(ag.get("temperature", 0.2)), 0.05, key=f"temp_{idx}")
                ag["max_tokens"] = st.number_input("Max tokens", 64, 32768, int(ag.get("max_tokens", 2048)), 64, key=f"tok_{idx}")
                ag["output_key"] = st.text_input("Output key", ag.get("output_key", f"agent_{idx+1}_output"), key=f"outk_{idx}")
                st.text_area("System prompt", ag.get("system_prompt", ""), key=f"sys_{idx}", height=120)
                st.text_area("User prompt template (Jinja)", ag.get("prompt", ""), key=f"usr_{idx}", height=150)

                # sync back updated fields from widgets
                ag["system_prompt"] = st.session_state[f"sys_{idx}"]
                ag["prompt"] = st.session_state[f"usr_{idx}"]

    with colB:
        st.subheader("YAML")
        st.session_state.yaml_text = yaml.safe_dump({"version": 1, "pipeline": st.session_state.agents}, allow_unicode=True, sort_keys=False)
        st.code(st.session_state.yaml_text, language="yaml")

        st.download_button("Download agents.yaml", data=st.session_state.yaml_text.encode("utf-8"), file_name="agents.yaml", mime="text/yaml")
        if st.button("Save agents.yaml to workspace"):
            with open(DEFAULT_YAML_PATH, "w", encoding="utf-8") as f:
                f.write(st.session_state.yaml_text)
            st.success("Saved agents.yaml")

# Run tab
with tabs[2]:
    st.header("Execute")
    missing_keys = []
    if any(ag.get("provider") == "gemini" for ag in st.session_state.agents) and not st.session_state.keys["GOOGLE_API_KEY"]:
        missing_keys.append("GOOGLE_API_KEY")
    if any(ag.get("provider") == "openai" for ag in st.session_state.agents) and not st.session_state.keys["OPENAI_API_KEY"]:
        missing_keys.append("OPENAI_API_KEY")
    if any(ag.get("provider") == "grok" for ag in st.session_state.agents) and not st.session_state.keys["XAI_API_KEY"]:
        missing_keys.append("XAI_API_KEY")

    if missing_keys:
        st.error(f"Missing API keys for: {', '.join(missing_keys)}")
    else:
        if st.button("Run Agent Pipeline", type="primary", disabled=not (st.session_state.doc1_text and st.session_state.doc2_text)):
            status = st.status("Running agents...", expanded=True)
            st.session_state.trace = []
            st.session_state.outputs = {}
            st.session_state.results_json = {}
            st.session_state.run_metrics = []

            router = ProviderRouter(
                google_api_key=st.session_state.keys["GOOGLE_API_KEY"],
                openai_api_key=st.session_state.keys["OPENAI_API_KEY"],
                xai_api_key=st.session_state.keys["XAI_API_KEY"]
            )

            def on_step(step):
                with status:
                    cols = st.columns([0.8, 0.2])
                    with cols[0]:
                        st.write(f"[{step['index']}] {step['name']} ({step['provider']}/{step['model']})")
                    with cols[1]:
                        wow_status_badge(step["status"])
                    if step["status"] == "success":
                        st.caption(f"Elapsed: {step['metrics'].get('elapsed_sec', 0):.2f}s | Provider: {step['metrics'].get('provider')} | Model: {step['metrics'].get('model')}")
                    elif step["status"] == "error":
                        st.error(step["error"])
                st.session_state.trace.append(step)
                if step["status"] == "success":
                    st.session_state.run_metrics.append(step["metrics"])

            base_context = {
                "doc1": st.session_state.doc1_text,
                "doc2": st.session_state.doc2_text
            }
            result_ctx = run_chain(st.session_state.agents, router, base_context, on_step=on_step)
            st.session_state.outputs = result_ctx.get("outputs", {})
            status.update(label="Run finished", state="complete", expanded=False)

            # Attempt to read final JSON if present
            # Convention: final aggregator outputs 'final_json'
            final_json = st.session_state.outputs.get("final_json")
            if final_json:
                try:
                    st.session_state.results_json = json.loads(final_json)
                    st.success("Parsed final JSON output successfully.")
                except Exception:
                    st.warning("Final output is not valid JSON. You can still inspect raw outputs in YAML tab.")

            st.toast("Pipeline completed")

    st.subheader("Trace")
    for t in st.session_state.trace:
        color = "green" if t["status"] == "success" else ("red" if t["status"] == "error" else "orange")
        st.markdown(f"- [{t['index']}] <span style='color:{color}'>{t['status']}</span> — {t['name']} ({t['provider']}/{t['model']})", unsafe_allow_html=True)

# Dashboard tab
with tabs[3]:
    st.header("Interactive Dashboard")
    res = st.session_state.results_json or {}
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Summary")
        summary_md = res.get("summary_markdown") or st.session_state.outputs.get("summary_markdown") or st.session_state.outputs.get("agent_27_output", "")
        if summary_md:
            st.markdown(summary_md)
        else:
            st.info("No summary available yet.")

    with col2:
        st.subheader("Key Metrics")
        total_time = sum(m.get("elapsed_sec", 0) for m in st.session_state.run_metrics)
        st.metric("Total elapsed (s)", f"{total_time:.2f}")
        st.metric("Agents executed", len(st.session_state.trace))
        providers_used = sorted(set(m.get("provider") for m in st.session_state.run_metrics if m.get("provider")))
        st.write("Providers:", ", ".join(providers_used) if providers_used else "-")

    st.markdown("---")
    colsB = st.columns(2)
    with colsB[0]:
        st.subheader("Keywords")
        keywords = res.get("keywords") or []
        if keywords and isinstance(keywords, list):
            render_keyword_bar(keywords)
        else:
            st.info("No keywords available.")

    with colsB[1]:
        st.subheader("Similarity Heatmap")
        sim = res.get("similarity_matrix")
        labels = res.get("labels")
        if sim and labels:
            render_similarity_heatmap(sim, labels)
        else:
            st.info("No similarity matrix available.")

    st.subheader("Keyword Graph")
    nodes = res.get("graph", {}).get("nodes", [])
    links = res.get("graph", {}).get("links", [])
    if nodes and links:
        html = render_keyword_graph(nodes, links, height=600)
        st.components.v1.html(html, height=620, scrolling=True)
    else:
        st.info("No graph data available.")

# YAML tab
with tabs[4]:
    st.header("Raw Outputs and YAML")
    st.subheader("Current outputs")
    st.json(st.session_state.outputs or {})

    st.subheader("Final JSON (if any)")
    st.json(st.session_state.results_json or {})

    st.subheader("agents.yaml (editable)")
    edited = st.text_area("Edit YAML", st.session_state.yaml_text, height=400)
    apply_btn = st.button("Apply YAML")
    if apply_btn:
        try:
            data = yaml.safe_load(edited)
            st.session_state.agents = data.get("pipeline", [])
            st.session_state.yaml_text = edited
            st.success("YAML applied to pipeline.")
        except Exception as e:
            st.error(f"Invalid YAML: {e}")

6) agents.yaml (Traditional Chinese, 31 agents)
version: 1
pipeline:
  - id: lang_normalizer
    name: 語言偵測與正規化器
    provider: gemini
    model: gemini-2.5-flash
    temperature: 0.1
    max_tokens: 1024
    output_key: normalized_docs
    system_prompt: |
      你是專精於跨語言文本預處理的 NLP 助理。
      任務：偵測兩份文件的語言與編碼問題，修復常見 OCR 錯誤（如 l/1、0/O），
      移除不可見字元，統一標點與空白，保留原本文意。
    prompt: |
      文件A:
      {{ doc1 }}

      文件B:
      {{ doc2 }}

      請輸出：
      - detected_lang_A
      - detected_lang_B
      - normalized_A
      - normalized_B

      格式：
      A語言:<...>
      B語言:<...>
      A文本:<...>
      B文本:<...>

  - id: docA_clean
    name: 文件A清理器
    provider: openai
    model: gpt-4o-mini
    temperature: 0.1
    max_tokens: 2048
    output_key: clean_A
    system_prompt: |
      你是文本清理與結構化專家，移除雜訊（HTML、重複行、破碎句）。
    prompt: |
      根據以下標準清理文本A，使其可供分析：
      - 保留語意
      - 去除重複
      - 合理分段與標點
      - 盡量保留專有名詞
      文本A來源：
      {{ outputs.normalized_docs }}

      僅輸出清理後的文本。

  - id: docB_clean
    name: 文件B清理器
    provider: openai
    model: gpt-4o-mini
    temperature: 0.1
    max_tokens: 2048
    output_key: clean_B
    system_prompt: |
      你是文本清理與結構化專家。
    prompt: |
      請以同樣規則清理文本B：
      來源（含A/B）： 
      {{ outputs.normalized_docs }}

      僅輸出清理後的文本B。

  - id: structure_extractor
    name: 結構抽取器（段落/標題）
    provider: gemini
    model: gemini-2.5-flash
    temperature: 0.2
    max_tokens: 2048
    output_key: structures
    system_prompt: |
      從文本A與B識別標題、段落、清單，建立可比對的結構框架。
    prompt: |
      請將下列文本A與文本B解析為結構化JSON：
      - sections: [{id, title, paragraphs: [..]}]
      - indices可供後續對齊
      文本A：
      {{ outputs.clean_A }}

      文本B：
      {{ outputs.clean_B }}

  - id: ner
    name: 命名實體抽取器
    provider: openai
    model: gpt-5-nano
    temperature: 0.2
    max_tokens: 2048
    output_key: ner_json
    system_prompt: |
      從雙文本抽取人名、地名、機構、日期、數量、專有名詞，輸出JSON。
    prompt: |
      請輸出：
      {
        "A": {"PERSON":[], "ORG":[], "LOC":[], "DATE":[], "NUM":[], "MISC":[]},
        "B": {"PERSON":[], "ORG":[], "LOC":[], "DATE":[], "NUM":[], "MISC":[]}
      }
      文本A：
      {{ outputs.clean_A }}

      文本B：
      {{ outputs.clean_B }}

  - id: keyphrase
    name: 關鍵詞與關鍵片語擷取
    provider: gemini
    model: gemini-2.5-flash
    temperature: 0.25
    max_tokens: 2048
    output_key: keywords
    system_prompt: |
      請以多策略擷取關鍵詞（RAKE/TF-IDF/顧名思義），合併去重，評分[0,1]。
    prompt: |
      請輸出JSON:
      {"A":[{"term":"...","score":0.x}], "B":[{"term":"...","score":0.x}]}
      參考文本A：
      {{ outputs.clean_A }}
      參考文本B：
      {{ outputs.clean_B }}

  - id: topics
    name: 主題建模與標籤器
    provider: openai
    model: gpt-4.1-mini
    temperature: 0.3
    max_tokens: 2048
    output_key: topics_json
    system_prompt: |
      將A/B文本標註多層主題，主題含label與代表詞。
    prompt: |
      請輸出JSON:
      {"A":[{"label":"...","terms":["..."],"confidence":0.x}], "B":[{"label":"...","terms":["..."],"confidence":0.x}]}
      A:
      {{ outputs.clean_A }}
      B:
      {{ outputs.clean_B }}

  - id: pos
    name: 斷詞與詞性標註器
    provider: grok
    model: grok-3-mini
    temperature: 0.2
    max_tokens: 2048
    output_key: pos_tags
    system_prompt: |
      你是詞性標註器，輸出精簡可讀格式。
    prompt: |
      對A與B各輸出：[token]/POS，保留前500項代表樣本。
      A文本：
      {{ outputs.clean_A }}
      B文本：
      {{ outputs.clean_B }}

  - id: sentiment
    name: 情感與情緒分析器
    provider: openai
    model: gpt-4o-mini
    temperature: 0.2
    max_tokens: 1024
    output_key: sentiment_json
    system_prompt: |
      進行情感(正/中/負)與情緒（喜/怒/哀/懼/驚/厭）機率分佈。
    prompt: |
      請輸出JSON:
      {"A":{"polarity":"...", "scores":{"pos":0.x,"neu":0.x,"neg":0.x}, "emotions":{"joy":0.x,"anger":0.x,"sadness":0.x,"fear":0.x,"surprise":0.x,"disgust":0.x}},
       "B":{"polarity":"...", "scores":{...}, "emotions":{...}}}
      A:
      {{ outputs.clean_A }}
      B:
      {{ outputs.clean_B }}

  - id: stance
    name: 立場與意圖分析器
    provider: gemini
    model: gemini-2.5-flash
    temperature: 0.3
    max_tokens: 2048
    output_key: stance_json
    system_prompt: |
      辨識作者立場（支持/反對/中立）與意圖（說服/告知/推銷/警示等）。
    prompt: |
      請輸出JSON:
      {"A":{"stance":"...","intent":"...","confidence":0.x},"B":{"stance":"...","intent":"...","confidence":0.x}}
      A:
      {{ outputs.clean_A }}
      B:
      {{ outputs.clean_B }}

  - id: tone
    name: 敘事框架與語氣分析器
    provider: openai
    model: gpt-5-nano
    temperature: 0.3
    max_tokens: 1024
    output_key: tone_json
    system_prompt: |
      萃取敘事框架（問題-解決、因果、對比）與語氣（正式、口語、煽動等）。
    prompt: |
      輸出JSON:
      {"A":{"frames":["..."],"tones":["..."]},"B":{"frames":["..."],"tones":["..."]}}
      A:
      {{ outputs.clean_A }}
      B:
      {{ outputs.clean_B }}

  - id: table_num
    name: 數值與表格抽取器
    provider: gemini
    model: gemini-2.5-flash
    temperature: 0.2
    max_tokens: 2048
    output_key: numbers_json
    system_prompt: |
      抽取表格、數值與單位，建立可比對欄位。
    prompt: |
      輸出JSON:
      {"A":{"tables":[...],"numbers":[{"value":...,"unit":"...","context":"..."}]},
       "B":{"tables":[...],"numbers":[...]}}
      A:
      {{ outputs.clean_A }}
      B:
      {{ outputs.clean_B }}

  - id: citations
    name: 引用與來源擷取器
    provider: openai
    model: gpt-4o-mini
    temperature: 0.2
    max_tokens: 1024
    output_key: citations_json
    system_prompt: |
      擷取參考文獻、連結、數據來源，並做正規化。
    prompt: |
      輸出JSON:
      {"A":[{"type":"url|book|paper","value":"..."}],"B":[{"type":"...","value":"..."}]}
      A:
      {{ outputs.clean_A }}
      B:
      {{ outputs.clean_B }}

  - id: fact_plan
    name: 事實核查規劃器
    provider: grok
    model: grok-3-mini
    temperature: 0.3
    max_tokens: 1024
    output_key: factcheck_plan
    system_prompt: |
      擬定需要核查的主張清單（top-N），指定核查方法。
    prompt: |
      產出待核查主張（A/B各不超過5條），含核查方向與資料來源建議。
      A/B主張抽樣來源：keywords, topics, numbers
      keywords:
      {{ outputs.keywords }}
      topics:
      {{ outputs.topics_json }}
      numbers:
      {{ outputs.numbers_json }}

  - id: fact_exec
    name: 事實核查執行器（輕量）
    provider: gemini
    model: gemini-2.5-flash
    temperature: 0.2
    max_tokens: 2048
    output_key: factcheck_results
    system_prompt: |
      根據規劃，進行常識與內部一致性核查（無外網存取），標記可信度。
    prompt: |
      請輸出JSON:
      {"A":[{"claim":"...","verdict":"SUPPORTED|CONTRADICTED|INSUFFICIENT","confidence":0.x}],
       "B":[{"claim":"...","verdict":"...","confidence":0.x}]}
      規劃：
      {{ outputs.factcheck_plan }}
      A:
      {{ outputs.clean_A }}
      B:
      {{ outputs.clean_B }}

  - id: contradiction
    name: 矛盾與不一致偵測
    provider: grok
    model: grok-4-fast-reaoning
    temperature: 0.2
    max_tokens: 2048
    output_key: contradictions
    system_prompt: |
      識別A/B之間的直接矛盾、定義衝突、數值差異與語義反轉。
    prompt: |
      以項目清單方式輸出矛盾點，附最小可重現片段（原句節錄）。

  - id: dedup
    name: 重複與冗詞刪減器
    provider: openai
    model: gpt-5-nano
    temperature: 0.2
    max_tokens: 1024
    output_key: concise_A_B
    system_prompt: |
      去除重複與贅詞，保留關鍵訊息與承上文關係。
    prompt: |
      請對A與B各給出精簡版本（各不超過300字）。
      A:
      {{ outputs.clean_A }}
      B:
      {{ outputs.clean_B }}

  - id: align_matrix
    name: 對齊矩陣產生器
    provider: gemini
    model: gemini-2.5-flash
    temperature: 0.2
    max_tokens: 2048
    output_key: alignment
    system_prompt: |
      建立A段落 vs B段落的相似度矩陣（0~1），並提供對齊配對建議。
    prompt: |
      參考structures：
      {{ outputs.structures }}
      請輸出JSON:
      {"labels_A":[...], "labels_B":[...], "matrix":[[...]], "pairs":[["A-1","B-3",0.82], ...]}

  - id: diff_summarizer
    name: 差異與重點摘要器
    provider: openai
    model: gpt-4o-mini
    temperature: 0.25
    max_tokens: 2048
    output_key: diffs
    system_prompt: |
      精要指出A/B的共通點、關鍵差異、缺漏與可能原因。
    prompt: |
      依據alignment、contradictions與keywords整合差異重點，輸出markdown。
      alignment:
      {{ outputs.alignment }}
      contradictions:
      {{ outputs.contradictions }}
      keywords:
      {{ outputs.keywords }}

  - id: risk_compliance
    name: 風險/合規/偏見審查器
    provider: gemini
    model: gemini-2.5-flash
    temperature: 0.25
    max_tokens: 2048
    output_key: risk_report
    system_prompt: |
      從偏見、歧視、隱私、法規合規角度檢視文本。
    prompt: |
      針對A與B分別列出風險項目、嚴重度(低/中/高)、建議因應。
      A:
      {{ outputs.clean_A }}
      B:
      {{ outputs.clean_B }}

  - id: hallucination_scan
    name: 假訊息/幻覺掃描器
    provider: openai
    model: gpt-4.1-mini
    temperature: 0.2
    max_tokens: 1536
    output_key: hallucinations
    system_prompt: |
      標記看似不可信、未有來源或矛盾之敘述。
    prompt: |
      以清單輸出可能幻覺陳述與理由，若無則回報「未發現」。

  - id: similarity_score
    name: 相似度與重合度評分器
    provider: gemini
    model: gemini-2.5-flash
    temperature: 0.2
    max_tokens: 1024
    output_key: sim_score
    system_prompt: |
      綜合語意與關鍵詞，給出整體相似度(0~1)與信心分數。
    prompt: |
      請輸出JSON: {"similarity":0.x,"confidence":0.x,"rationale":"..."}
      A:
      {{ outputs.clean_A }}
      B:
      {{ outputs.clean_B }}

  - id: kg_planner
    name: 知識圖規劃器
    provider: grok
    model: grok-3-mini
    temperature: 0.3
    max_tokens: 1536
    output_key: kg_plan
    system_prompt: |
      規劃節點類型（事件、實體、指標）與邊關係（因果、屬性、對應）。
    prompt: |
      根據NER與關鍵詞，輸出圖建構規劃。
      NER:
      {{ outputs.ner_json }}
      Keywords:
      {{ outputs.keywords }}

  - id: graph_builder
    name: 關鍵詞關聯圖建構器
    provider: gemini
    model: gemini-2.5-flash
    temperature: 0.25
    max_tokens: 2048
    output_key: graph_json
    system_prompt: |
      產生可視化用圖資料：nodes[{id,label,score}], links[{source,target,weight}]
    prompt: |
      請整合kg_plan與keywords，輸出{"nodes":[...],"links":[...]}。
      kg_plan:
      {{ outputs.kg_plan }}
      keywords:
      {{ outputs.keywords }}

  - id: cluster_taxonomy
    name: 叢集與主題層級化
    provider: openai
    model: gpt-4.1-mini
    temperature: 0.25
    max_tokens: 2048
    output_key: taxonomy
    system_prompt: |
      將主題分群並建立層級目錄（Level 1-3）。
    prompt: |
      請輸出JSON:
      {"taxonomy":[{"label":"...","children":[...]}]}

  - id: viz_script
    name: 可視化腳本產生器
    provider: gemini
    model: gemini-2.5-flash
    temperature: 0.3
    max_tokens: 1024
    output_key: viz_plan
    system_prompt: |
      規劃儀表板視圖：條形圖、熱力圖、詞網圖等所需欄位。
    prompt: |
      依據graph_json、similarity與keywords，給出可視化欄位規劃。
      graph_json:
      {{ outputs.graph_json }}
      similarity:
      {{ outputs.sim_score }}
      keywords:
      {{ outputs.keywords }}

  - id: final_synth
    name: 最終綜整摘要器
    provider: openai
    model: gpt-4o-mini
    temperature: 0.3
    max_tokens: 4096
    output_key: summary_markdown
    system_prompt: |
      你是高階研究助理。請用精確、條理分明、具可追溯性的方式撰寫專家級綜整。
      應含：背景、方法（代理鏈）、共通點、重要差異、風險、建議與結論。
    prompt: |
      資料：
      - diffs: {{ outputs.diffs }}
      - risk: {{ outputs.risk_report }}
      - hallucinations: {{ outputs.hallucinations }}
      - topics: {{ outputs.topics_json }}
      - factcheck: {{ outputs.factcheck_results }}
      - alignment: {{ outputs.alignment }}
      請輸出Markdown報告，附重點條列與表格摘要。

  - id: qa_generator
    name: QA 問答生成器
    provider: gemini
    model: gemini-2.5-flash
    temperature: 0.35
    max_tokens: 2048
    output_key: qa_pairs
    system_prompt: |
      基於總結與差異，提出高品質問答對（5-10組）。
    prompt: |
      summary_markdown:
      {{ outputs.summary_markdown }}

  - id: research_next
    name: 後續研究建議生成器
    provider: openai
    model: gpt-4.1-mini
    temperature: 0.35
    max_tokens: 2048
    output_key: next_steps
    system_prompt: |
      根據缺口、風險與矛盾，提出可執行的研究或驗證步驟。
    prompt: |
      請輸出條列式建議（每條含目的、資料與方法）。

  - id: metrics_aggregator
    name: 評測指標彙總器
    provider: grok
    model: grok-3-mini
    temperature: 0.2
    max_tokens: 1536
    output_key: metrics_summary
    system_prompt: |
      彙整相似度、主題覆蓋率、關鍵詞集中度、矛盾數量等為簡表。
    prompt: |
      可用資料：
      - sim_score: {{ outputs.sim_score }}
      - topics: {{ outputs.topics_json }}
      - keywords: {{ outputs.keywords }}
      - contradictions: {{ outputs.contradictions }}
      請輸出Markdown表格。

  - id: packaging
    name: 後處理與輸出包裝器
    provider: gemini
    model: gemini-2.5-flash
    temperature: 0.2
    max_tokens: 4096
    output_key: final_json
    system_prompt: |
      將先前結果彙整為前端可用JSON。
    prompt: |
      請輸出application/json，包含：
      {
        "summary_markdown": "...",
        "keywords": ["..."],
        "graph": {"nodes":[{"id":"...","label":"...","score":0.x}], "links":[{"source":"...","target":"...","weight":0.x}]},
        "similarity_matrix": [[...]],
        "labels": ["...","..."]
      }
      來源：
      - summary_markdown: {{ outputs.summary_markdown }}
      - keywords: 從 {{ outputs.keywords }} 取前20項詞，僅保留term
      - graph: 來自 {{ outputs.graph_json }}
      - similarity_matrix/labels: 來自 {{ outputs.alignment }}

Note: This pipeline purposefully mixes providers to showcase routing. You can change providers/models per agent in the Pipeline tab.

Advanced final prompt (already embedded in 最終綜整摘要器)
- Carefully structured to produce an expert-grade, verifiable Markdown report with sections for methods, commonalities, differences, risks, and recommendations.

How to deploy on Hugging Face Spaces (Streamlit)
- Add files: app.py, providers.py, chain.py, viz.py, agents.yaml, requirements.txt
- Set Space SDK to Streamlit.
- Define environment variables in Space Secrets (recommended), or rely on user input on the page:
  - GOOGLE_API_KEY
  - OPENAI_API_KEY
  - XAI_API_KEY
- Push repository; the app will read keys from environment. If absent, UI inputs are shown.

“Wow” status indicators and Interactive dashboard
- Per-agent step status badges (success/running/error), with elapsed time and provider/model telemetry.
- Toasts and a final completion capsule.
- Dashboard includes:
  - Markdown summary
  - Keyword frequency bars
  - Similarity heatmap
  - Interactive keyword graph (pyvis with drag)
- YAML editor with upload/download and inline validation.

Security and privacy
- Keys read from environment are never shown.
- If user inputs keys, they stay in session only.
- No external network calls beyond LLM providers; fact-check step is kept internal (no browsing) by design.

Notes
- For image OCR or vision tasks, you can extend ProviderRouter.call to pass image parts to Gemini or Grok and add agents with images fields.
- If you want streaming responses, adapt OpenAI/Gemini calls to stream and surface partial tokens in the status panel.

20 comprehensive follow-up questions
1) Do you want the pipeline to support browsing (live web search) for fact-checking, or keep it strictly offline as currently designed?
2) Should we add image and table OCR via Gemini Vision for PDF scans and embedded images, and orchestrate it as a pre-agent step?
3) Which default models would you like per provider for speed vs quality trade-offs, and should we auto-switch to smaller models for long chains?
4) Do you prefer stricter JSON schema validation for agent outputs (with schema enforcement and repair) to reduce parsing errors?
5) Should the dashboard allow users to click graph nodes to filter the summary and show related sentences dynamically?
6) Do you need export formats beyond YAML/JSON/Markdown (e.g., docx, PDF, HTML report with embedded visuals)?
7) Would you like to persist runs and traces to a lightweight database (e.g., SQLite) for history, versioning, and reproducibility?
8) Should we add a benchmarking mode to compare different pipelines/models on the same documents with aggregate metrics?
9) Do you want role-based controls (viewer vs editor) with restricted editing of agents.yaml in shared Spaces?
10) Should we add prompt libraries and presets, and a diff viewer for comparing prompt changes across versions?
11) Would you like partial re-run capability (resume from agent N) and caching to skip unchanged steps?
12) Should we integrate evaluation datasets and a test harness to validate pipeline quality automatically?
13) Do you need multilingual UI (English/Traditional Chinese toggle) with persisted preference?
14) Would you like advanced token budgeting that estimates cost/latency and enforces chain-wide limits?
15) Should we add a tool abstraction layer (e.g., calculators, regex extractors, semantic search) callable by agents?
16) Do you want cross-document citation linking that highlights where each claim originates in A or B with anchors?
17) Should the similarity heatmap include interactive selection that updates the aligned paragraph view below?
18) Are there compliance constraints requiring on-prem inference or proxy routing rather than direct API calls?
19) Would you like an automated agent quality report that flags brittle prompts and suggests improvements?
20) Should we schedule or batch runs (e.g., watch a folder/repo of documents) and auto-generate periodic comparison reports?

import os
import io
import json
import time
import yaml
import base64
import re
import pdfplumber
import streamlit as st
import streamlit_pdf_viewer as st_pdf_viewer
from typing import Optional, Dict, Any, List

# --- OCR & Image Processing Imports ---
from PyPDF2 import PdfReader
from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract
from wordcloud import WordCloud
import matplotlib.pyplot as plt


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
        images: Optional[List[bytes]] = None  # Accepts list of image bytes
    ) -> Dict[str, Any]:
        start = time.time()
        provider = (provider or "").lower()
        text = ""
        usage = {}

        if provider == "gemini":
            genai_client = self._get_gemini()
            model_obj = genai_client.GenerativeModel(model)
            
            contents = [user_prompt]
            if images:
                for img_bytes in images:
                    img = Image.open(io.BytesIO(img_bytes))
                    contents.append(img)

            resp = model_obj.generate_content(
                contents=contents,
                generation_config={"temperature": temperature, "max_output_tokens": max_tokens}
            )
            text = getattr(resp, "text", "") or ""
            usage = {"provider": "gemini", "model": model}

        elif provider == "openai":
            client = self._get_openai()
            
            messages = [{"role": "system", "content": system_prompt}]
            user_content = [{"type": "text", "text": user_prompt}]
            if images:
                for img_bytes in images:
                    b64_img = base64.b64encode(img_bytes).decode('utf-8')
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_img}"}
                    })
            messages.append({"role": "user", "content": user_content})

            resp = client.chat.completions.create(
                model=model, temperature=temperature, max_tokens=max_tokens,
                messages=messages
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
                 for idx, img_bytes in enumerate(images):
                    chat.append(xai_user(f"Image context {idx+1}", xai_image(img_bytes)))
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
def generate_wordcloud(text: str):
    if not text:
        return None
    try:
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        return fig
    except Exception:
        return None

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
    """
    Initializes the entire session state on the very first run.
    """
    if "initialized" in st.session_state:
        return

    st.session_state.initialized = True
    
    st.session_state.api_keys = {}
    st.session_state.key_sources = {}
    
    key_configs = {
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "XAI_API_KEY": os.getenv("XAI_API_KEY"),
    }
    
    for key, env_value in key_configs.items():
        st.session_state.api_keys[key] = env_value or ""
        st.session_state.key_sources[key] = "env" if env_value else "user"
            
    # --- Document and OCR State ---
    for i in ['1', '2']:
        setattr(st.session_state, f'doc{i}_text', "")
        setattr(st.session_state, f'doc{i}_bytes', None)
        setattr(st.session_state, f'doc{i}_page_count', 0)
        setattr(st.session_state, f'doc{i}_ocr_text', "")
        setattr(st.session_state, f'doc{i}_wordcloud', None)

    # --- Other State Variables ---
    st.session_state.ctx = {}
    st.session_state.agents = []
    st.session_state.yaml_text = ""
    st.session_state.trace = []
    st.session_state.outputs = {}
    st.session_state.results_json = {}
    st.session_state.run_metrics = []
    st.session_state.summary_markdown = ""
    st.session_state.custom_keywords = []

def render_sidebar():
    """Renders the sidebar with API key settings."""
    with st.sidebar:
        st.title("Settings")
        st.caption("Agentic AI Document Comparison (Streamlit)")
        st.subheader("API Keys")
        
        for key_name, label in [("GOOGLE_API_KEY", "Gemini"), ("OPENAI_API_KEY", "OpenAI"), ("XAI_API_KEY", "Grok")]:
            if st.session_state.key_sources.get(key_name) == "env":
                st.success(f"{label}: Using environment key")
            else:
                st.session_state.api_keys[key_name] = st.text_input(
                    f"{key_name} ({label})",
                    value=st.session_state.api_keys.get(key_name, ""),
                    type="password"
                )
        
        st.markdown("---")
        st.caption("Models will be selected per-agent in the Pipeline tab.")

def render_documents_tab(tab):
    """Renders the UI for document upload and preview."""
    with tab:
        st.header("Documents")
        col1, col2 = st.columns(2)

        def setup_doc_ui(doc_num):
            doc_key = f'doc{doc_num}'
            st.subheader(f"Document {chr(64 + doc_num)}") # A or B
            uploaded_file = st.file_uploader(f"Upload PDF/TXT for Document {chr(64 + doc_num)}", type=["pdf", "txt"], key=f"{doc_key}_upl")
            
            if uploaded_file:
                file_bytes = uploaded_file.getvalue()
                setattr(st.session_state, f'{doc_key}_bytes', file_bytes)
                
                if uploaded_file.type == "application/pdf":
                    try:
                        reader = PdfReader(io.BytesIO(file_bytes))
                        setattr(st.session_state, f'{doc_key}_page_count', len(reader.pages))
                    except Exception:
                        setattr(st.session_state, f'{doc_key}_page_count', 0)
                else:
                    setattr(st.session_state, f'{doc_key}_text', file_bytes.decode("utf-8", errors="ignore"))
            
            with st.expander(f"Preview Document {chr(64 + doc_num)}", expanded=True):
                doc_bytes = getattr(st.session_state, f'{doc_key}_bytes')
                if doc_bytes:
                    is_pdf = uploaded_file and uploaded_file.type == "application/pdf"
                    if is_pdf:
                        st_pdf_viewer.pdf_viewer(doc_bytes, height=400)
                    else:
                        st.text_area("Text Preview", getattr(st.session_state, f'{doc_key}_text'), height=300, disabled=True)
                else:
                    st.info("Upload a document to see a preview.")

        with col1:
            setup_doc_ui(1)
        with col2:
            setup_doc_ui(2)

def render_ocr_tab(tab):
    """Renders the UI for OCR processing."""
    with tab:
        st.header("Document OCR")
        st.info("Use this tab to extract text from scanned PDFs or images using OCR.")

        col1, col2 = st.columns(2)

        def setup_ocr_ui(doc_num):
            doc_key = f"doc{doc_num}"
            st.subheader(f"OCR for Document {chr(64+doc_num)}")
            
            if not getattr(st.session_state, f"{doc_key}_bytes"):
                st.warning(f"Please upload Document {chr(64+doc_num)} first.")
                return

            ocr_engine = st.selectbox("OCR Engine", 
                                      ["pdfplumber", "pytesseract", "gemini-1.5-flash", "gpt-4o-mini"], 
                                      key=f"{doc_key}_ocr_engine")
            
            page_count = getattr(st.session_state, f"{doc_key}_page_count")
            all_pages = list(range(1, page_count + 1))
            selected_pages = st.multiselect("Select pages to OCR", all_pages, default=all_pages[:5], key=f"{doc_key}_pages")

            if st.button("Run OCR", key=f"{doc_key}_run_ocr", disabled=not selected_pages):
                with st.spinner(f"Running OCR with {ocr_engine}..."):
                    doc_bytes = getattr(st.session_state, f"{doc_key}_bytes")
                    ocr_text = ""
                    try:
                        if ocr_engine == "pdfplumber":
                            with pdfplumber.open(io.BytesIO(doc_bytes)) as pdf:
                                for page_num in selected_pages:
                                    ocr_text += pdf.pages[page_num - 1].extract_text() + "\n"
                        
                        elif ocr_engine == "pytesseract":
                            try:
                                images = convert_from_bytes(doc_bytes, first_page=min(selected_pages), last_page=max(selected_pages))
                                page_map = {i + min(selected_pages): img for i, img in enumerate(images)}
                                for page_num in selected_pages:
                                    if page_num in page_map:
                                        ocr_text += pytesseract.image_to_string(page_map[page_num]) + "\n"
                            except Exception as e:
                                st.error(f"Pytesseract/Poppler error: {e}. Ensure they are installed and in your system's PATH.")

                        elif ocr_engine in ["gemini-1.5-flash", "gpt-4o-mini"]:
                            router = ProviderRouter(google_api_key=st.session_state.api_keys.get("GOOGLE_API_KEY"),
                                                    openai_api_key=st.session_state.api_keys.get("OPENAI_API_KEY"),
                                                    xai_api_key=st.session_state.api_keys.get("XAI_API_KEY"))
                            images_bytes = []
                            images = convert_from_bytes(doc_bytes, first_page=min(selected_pages), last_page=max(selected_pages))
                            page_map = {i + min(selected_pages): img for i, img in enumerate(images)}

                            for page_num in selected_pages:
                                with io.BytesIO() as output:
                                    page_map[page_num].save(output, format="PNG")
                                    images_bytes.append(output.getvalue())
                            
                            provider = "gemini" if "gemini" in ocr_engine else "openai"
                            resp = router.call(provider=provider, model=ocr_engine,
                                             system_prompt="You are an expert OCR agent.",
                                             user_prompt="Extract all text from the provided image(s).",
                                             images=images_bytes)
                            ocr_text = resp["text"]

                        setattr(st.session_state, f"{doc_key}_ocr_text", ocr_text)
                        setattr(st.session_state, f"{doc_key}_wordcloud", generate_wordcloud(ocr_text))

                    except Exception as e:
                        st.error(f"An error occurred during OCR: {e}")

            ocr_result = getattr(st.session_state, f"{doc_key}_ocr_text")
            if ocr_result:
                st.text_area("OCR Result", ocr_result, height=200)
                if st.button(f"Use this text for Document {chr(64+doc_num)}", key=f"{doc_key}_use_text"):
                    setattr(st.session_state, f"{doc_key}_text", ocr_result)
                    st.success(f"OCR text applied to Document {chr(64+doc_num)}.")
                
                wordcloud_fig = getattr(st.session_state, f"{doc_key}_wordcloud")
                if wordcloud_fig:
                    st.pyplot(wordcloud_fig)

        with col1:
            setup_ocr_ui(1)
        with col2:
            setup_ocr_ui(2)
            
# --- Other render functions (pipeline, run, summary, dashboard, yaml) remain the same ---
# ... (omitting for brevity, as they are unchanged from the previous version) ...

def render_pipeline_tab(tab):
    """Renders the UI for the Pipeline tab."""
    with tab:
        st.header("Agent Pipeline")
        colA, colB = st.columns([2, 1])
        with colA:
            upload_yaml = st.file_uploader("Upload agents.yaml", type=["yaml", "yml"])
            if upload_yaml:
                st.session_state.yaml_text = upload_yaml.read().decode("utf-8")
                data = yaml.safe_load(st.session_state.yaml_text)
                st.session_state.agents = data.get("pipeline", [])
            
            if not st.session_state.agents and os.path.exists(DEFAULT_YAML_PATH):
                with open(DEFAULT_YAML_PATH, "r", encoding="utf-8") as f:
                    st.session_state.yaml_text = f.read()
                    data = yaml.safe_load(st.session_state.yaml_text)
                    st.session_state.agents = data.get("pipeline", [])

            st.write("Active agents:", len(st.session_state.agents))
            for idx, ag in enumerate(st.session_state.agents):
                with st.expander(f"[{idx+1}] {ag.get('name', ag.get('id'))}", expanded=(idx < 3)):
                    prov_options = ["gemini", "openai", "grok"]
                    ag["provider"] = st.selectbox("Provider", prov_options, index=prov_options.index(ag.get("provider", "gemini")), key=f"prov_{idx}")
                    
                    model_options = {
                        "gemini": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"],
                        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
                        "grok": ["grok-1.5-flash", "grok-1.5"]
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
        keys = st.session_state.api_keys
        agents = st.session_state.agents
        
        missing_keys = set()
        providers_in_use = {ag.get("provider") for ag in agents}
        
        if "gemini" in providers_in_use and not keys.get("GOOGLE_API_KEY"):
            missing_keys.add("GOOGLE_API_KEY")
        if "openai" in providers_in_use and not keys.get("OPENAI_API_KEY"):
            missing_keys.add("OPENAI_API_KEY")
        if "grok" in providers_in_use and not keys.get("XAI_API_KEY"):
            missing_keys.add("XAI_API_KEY")

        if missing_keys:
            st.error(f"Missing API keys for providers in pipeline: {', '.join(missing_keys)}")
        
        if st.button("Run Agent Pipeline", type="primary", disabled=not (st.session_state.doc1_text and st.session_state.doc2_text)):
            st.session_state.trace, st.session_state.outputs, st.session_state.results_json, st.session_state.run_metrics = [], {}, {}, []
            status = st.status("Running agents...", expanded=True)
            router = ProviderRouter(google_api_key=keys.get("GOOGLE_API_KEY"), openai_api_key=keys.get("OPENAI_API_KEY"), xai_api_key=keys.get("XAI_API_KEY"))

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
            st.session_state.summary_markdown = st.session_state.outputs.get("summary_markdown", "No summary was generated.")
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

def render_summary_tab(tab):
    """Renders the UI for the Summary and Interactive Analysis tab."""
    with tab:
        st.header("Summary and Analysis")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Keyword Highlighting")
            default_keywords_str = st.text_input("Default Keywords (comma-separated)", "data, AI, model, analysis, risk")
            default_keywords = [k.strip() for k in default_keywords_str.split(",") if k.strip()]
            
            # Custom keywords
            with st.form("custom_keywords_form"):
                kw_text = st.text_input("Add Custom Keyword")
                kw_color = st.color_picker("Choose Color", "#FFDDC1")
                if st.form_submit_button("Add Keyword"):
                    if kw_text:
                        st.session_state.custom_keywords.append({"text": kw_text, "color": kw_color})

            for i, kw in enumerate(st.session_state.custom_keywords):
                cols = st.columns([0.6, 0.3, 0.1])
                cols[0].markdown(f"<span style='background-color:{kw['color']}; padding: 2px 5px; border-radius: 5px;'>{kw['text']}</span>", unsafe_allow_html=True)
                cols[1].write(kw["color"])
                if cols[2].button("X", key=f"del_kw_{i}"):
                    st.session_state.custom_keywords.pop(i)
                    st.rerun()

        with col2:
            st.subheader("Interactive Agent Analysis")
            if not st.session_state.agents:
                st.warning("No agents defined in pipeline.")
                return

            agent_options = {f"[{i+1}] {ag.get('name')}": i for i, ag in enumerate(st.session_state.agents)}
            selected_agent_name = st.selectbox("Select Agent for Analysis", options=list(agent_options.keys()))
            
            if selected_agent_name:
                agent_idx = agent_options[selected_agent_name]
                selected_agent = st.session_state.agents[agent_idx].copy()

                st.text_input("Model", selected_agent.get("model", ""), key="analysis_model", disabled=True)
                analysis_prompt = st.text_area("Analysis Prompt", "Analyze the following summary and provide your insights:\n\n---\n\n{{ summary }}", height=150)
                
                if st.button("Analyze Summary with Selected Agent"):
                    if not st.session_state.summary_markdown:
                        st.error("There is no summary to analyze.")
                    else:
                        router = ProviderRouter(
                            google_api_key=st.session_state.api_keys.get("GOOGLE_API_KEY"),
                            openai_api_key=st.session_state.api_keys.get("OPENAI_API_KEY"),
                            xai_api_key=st.session_state.api_keys.get("XAI_API_KEY")
                        )
                        final_prompt = render_template(analysis_prompt, {"summary": st.session_state.summary_markdown})
                        
                        with st.spinner(f"Running analysis with {selected_agent_name}..."):
                            resp = router.call(
                                provider=selected_agent["provider"],
                                model=selected_agent["model"],
                                system_prompt="You are a helpful analysis assistant.",
                                user_prompt=final_prompt,
                                temperature=selected_agent.get("temperature", 0.2),
                                max_tokens=selected_agent.get("max_tokens", 2048)
                            )
                            st.session_state.analysis_result = resp.get("text", "No result from analysis.")

        st.markdown("---")
        st.subheader("Generated Summary")
        st.session_state.summary_markdown = st.text_area("Edit summary content here:", st.session_state.summary_markdown, height=250)
        
        # Highlight keywords in the summary
        display_summary = st.session_state.summary_markdown
        all_keywords = [{"text": k, "color": "#FF6347"} for k in default_keywords] + st.session_state.custom_keywords
        
        for kw in all_keywords:
            pattern = re.compile(f"({re.escape(kw['text'])})", re.IGNORECASE)
            display_summary = pattern.sub(f"<span style='background-color:{kw['color']}; padding: 2px 5px; border-radius: 5px;'>\\1</span>", display_summary)
        
        st.markdown(display_summary, unsafe_allow_html=True)

        if "analysis_result" in st.session_state:
            st.subheader("Analysis Result")
            st.markdown(st.session_state.analysis_result)

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
    
    tabs = st.tabs([
        "Documents", "OCR", "Pipeline", "Run", "Summary", "Dashboard", "YAML"
    ])
    
    render_documents_tab(tabs[0])
    render_ocr_tab(tabs[1])
    render_pipeline_tab(tabs[2])
    render_run_tab(tabs[3])
    render_summary_tab(tabs[4])
    render_dashboard_tab(tabs[5])
    render_yaml_tab(tabs[6])

if __name__ == "__main__":
    main()

import json
import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast
import streamlit as st
from translation import translate_mbr


MODELDIR = './model'

st.set_page_config(
    page_title="Pocket Polyglot Mzansi",
    page_icon="🇿🇦",
    layout="wide",
    initial_sidebar_state=None,
    menu_items=None
)

if "model" not in st.session_state:
    with st.spinner("Loading model for the first time..."):
        st.session_state.tokenizer = NllbTokenizerFast.from_pretrained(MODELDIR)
        st.session_state.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODELDIR,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        _ = st.session_state.model.eval()
if "lang_map" not in st.session_state:
    with open("languages.json", "r") as f:
        st.session_state.lang_map = json.load(f)
if "mbr_result" not in st.session_state:
    st.session_state.mbr_result = ""
if "run_translation_state" not in st.session_state:
    st.session_state.run_translation_state = False

lang_map = st.session_state.lang_map
lang_codes = sorted(list(lang_map.keys()))
langs = [lang_map[code] for code in lang_codes]

def set_run_translation_state():
    st.session_state.run_translation_state = True

def run_translation():
    if (st.session_state.src_text is not None
        and st.session_state.src_lang is not None
        and st.session_state.tgt_lang is not None):
        st.session_state.mbr_result = translate_mbr(
            st.session_state.src_text, st.session_state.model, st.session_state.tokenizer,
            st.session_state.src_lang, st.session_state.tgt_lang, num_beams=5
        )
    st.rerun()

# Streamlit UI
st.html("""
    <style>
        .block-container {
            padding-top: 0rem;
        }
        .title-space {
            margin-bottom: 0.1rem;
        }
        h1 {
            padding-top: 0.1rem;
            font-size: 44px;
        }
    </style>
""")
# st.title("Pocket Polyglot Mzansi")
st.html('<h1 class="title-space">Pocket Polyglot Mzansi</h1>')
c0, c1, c2 = st.columns([1, 2, 2], gap="medium")
c0.image("assets/title-image-no-header.jpeg", width=200)

with c1:
    st.session_state.src_lang = st.pills(
        "Source Language",
        options=lang_codes,
        format_func=lambda x: lang_map[x],
        default="eng_Latn",
        on_change=set_run_translation_state,
    )
    st.session_state.src_text = st.text_area(
        "Source Text",
        value=None,
        placeholder="Enter text to translate...",
        label_visibility="collapsed",
        on_change=set_run_translation_state,
    )

with c2:
    st.session_state.tgt_lang = st.pills(
        "Target Language",
        options=lang_codes,
        format_func=lambda x: lang_map[x],
        default="zul_Latn",
        on_change=set_run_translation_state,
    )
    st.text(st.session_state.mbr_result)

do_translate = st.button(
    "Translate", type="primary", use_container_width=True,
    on_click=set_run_translation_state
)
if st.session_state.run_translation_state:
    st.session_state.run_translation_state = False
    run_translation()

# Pocket Polyglot Mzansi Demo

Demo of `Pocket Polyglot Mzansi`, a tiny 50M machine translation model for the following six South African languages:
- Afrikaans
- English
- isiXhosa
- isiZulu
- Setswana
- Sepedi

## Installation
We recommend installing in a virtual environment with Python >= 3.12.
```bash
git clone https://github.com/stefan027/pocket_polyglot_mzansi_demo.git
cd pocket_polyglot_mzansi_demo
pip install -r requirements.txt
```

## Running the Streamlit app
```bash
streamlit run app.py
```

## Other resources
- [Hosted demo on Streamlit Community Cloud](https://pocketpolyglotmzansi.streamlit.app/)
- Model weights on 🤗 Hugging Face: [Model (6 languages)](https://huggingface.co/stefan7/pocket_polyglot_mzansi_50M_6langs) | [Model (4 languages)](https://huggingface.co/stefan7/pocket_polyglot_mzansi_50M_4langs)
- [Slides of my talk at Deep Learning IndabaX South Africa 2025](https://docs.google.com/presentation/d/e/2PACX-1vQdIX1MOprBX4iE1y3iQb0_9ky1-dG__gwPR1RZSXPaK1GXp_Y5DdHX3fjVgwGdxsqmhRktWHmjKImk/pub?start=true&loop=false&delayms=3000)
- [COMING SOON] Main repo with training code
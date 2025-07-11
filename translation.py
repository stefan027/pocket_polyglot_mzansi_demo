from typing import List
import torch
import sacrebleu as scb


def translate(
    article, model, tokenizer, src_langs, tgt_langs, max_length=30, num_beams=1,
    do_sample=False, temperature=None, **kwargs
):
    """
    Translates a given text using a specified model and tokenizer.

    Args:
        article (`str` or `list` of `str`):
            The text to be translated from the source language to the target language.
        model (`transformers.PreTrainedModel`):
            The pre-trained model used for generating translations.
        tokenizer (`transformers.PreTrainedTokenizer`):
            The tokenizer used for encoding the input text and decoding the generated output.
        src_langs (`list ` of `str`):
            The source language codes for the tokenizer for each input text.
        tgt_langs (`list` of `str`):
            The target language codes for the tokenizer for each input text.
        max_length (`int`, optional):
            The maximum length of the generated translation. Default is `30`.
        num_beams (`int`, optional):
            The number of beams for beam search. Default is `1`. If set to `1`, it uses
            greedy decoding.
        do_sample (`bool`, optional):
            Whether to use sampling instead of greedy decoding. Default is `False`.
        temperature (`float`, optional):
            The temperature to use for sampling. Higher values (e.g., 1.0) result in more
            diverse outputs, while lower values (e.g., 0.5) make the output more deterministic.
            Default is `None`.

    Returns:
        list of `str`:
            The translated text(s) as a list of strings. If the input is a single string,
            the output will be a list with one translation.
    """
    if isinstance(article, str):
        article = [article]
    if isinstance(src_langs, str):
        src_langs = [src_langs]
    if isinstance(tgt_langs, str):
        tgt_langs = [tgt_langs]

    text_inputs = [f"{src} {t}</s>" for t, src in zip(article, src_langs)]
    inputs = tokenizer(
        text_inputs, return_tensors='pt', padding=True, truncation=True,
        max_length=128, add_special_tokens=False
    )
    decoder_input_ids = torch.tensor(
        [[tokenizer.convert_tokens_to_ids(l)] for l in tgt_langs]
    )
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
        decoder_input_ids = decoder_input_ids.to("cuda")

    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            decoder_input_ids=decoder_input_ids,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            **kwargs
        )
    translations = [
        tokenizer.decode(tokens, skip_special_tokens=True)
        for tokens in translated_tokens
    ]
    return translations


def mbr_decoding(candidates: List[str]) -> str:
    n = len(candidates)
    scores = []
    
    for i in range(n):
        bleu_sum = 0
        for j in range(n):
            if i != j:
                # bleu = sacrebleu.sentence_bleu(candidates[i], [candidates[j]]).score
                bleu = scb.sentence_chrf(candidates[i], [candidates[j]], word_order=2).score
                bleu_sum += bleu
        avg_bleu = bleu_sum / (n - 1)
        scores.append(avg_bleu)

    best_index = scores.index(max(scores))  # MBR: max expected utility
    return candidates[best_index]


def translate_mbr(
    article, model, tokenizer, src_langs, tgt_langs, max_length=256,
    num_beams=10, do_sample=False, temperature=None, **kwargs
):
    candidates = translate(
        article, model, tokenizer, src_langs, tgt_langs, max_length=max_length,
        num_beams=num_beams, do_sample=do_sample, temperature=temperature,
        num_return_sequences=num_beams, **kwargs
    )
    return mbr_decoding(candidates)

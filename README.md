# Multimodal Medical QA with BiomedCLIP

A Streamlit app that answers multiple-choice medical image questions with
[BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224),
and scores itself against the ground truth in
[`WorldMedQA/V`](https://huggingface.co/datasets/WorldMedQA/V).

Built for the AI in Health course at aivancity School of AI & Data
(Programme Grande École), 2026.

> This is a research and teaching demo. It is **not** a clinical tool and must not be used to
> inform any real decision about a patient.

---

## The idea, and its limit

BiomedCLIP is a CLIP-style vision–language model: it embeds images and text into a shared
space and measures similarity. It does **not** generate text, so it cannot "answer" a question
in the usual sense.

The app works around that by turning the task into retrieval. Each answer option is rendered
into a candidate prompt, every candidate is scored against the image, and the highest-scoring
option is taken as the prediction. That makes the result easy to interpret — you can see the
similarity score for every option, not just the winner — but it also means the model never
reasons over the question. It matches. Where the distractors are visually similar to the
correct answer, that distinction matters, and the scores show it.

---

## What it does

1. Loads `WorldMedQA/V` from Hugging Face, or from local processed TSV files.
2. Decodes and displays the medical image.
3. Scores the image against each question-and-answer candidate with BiomedCLIP.
4. Compares the top-scoring option with the dataset ground truth.
5. Runs a sampled batch evaluation for a quick accuracy figure.

Use the **Question and answer** prompt mode for the clearest matching behaviour, and
**Batch evaluation** for the accuracy check.

---

## Run it

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

### Model weights

Weights are **not** in this repository — the BiomedCLIP checkpoint is about 750 MB, far past
GitHub's 100 MB per-file limit. On first run `open_clip` downloads it from Hugging Face and
caches it locally, so the first prediction is slow and every later one is fast.

To use a local copy instead, place the files from the Hugging Face repo in `model/`:

```
model/
  open_clip_pytorch_model.bin
  open_clip_config.json
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  vocab.txt
```

`model/` is git-ignored.

### Dataset

`WorldMedQA/V` is pulled through `datasets.load_dataset()` on demand. The pre-processed TSV
exports used during development are also excluded here — they are ~75 MB each and are
reproducible from the Hugging Face dataset.

---

## Files

```
app.py                      the Streamlit application
biomed_clip_example.ipynb   walkthrough: loading the model, embedding, scoring
biomed-vlp-eval.svg         evaluation figure
requirements.txt
```

---

## Author

Samir Mahamat Saleh — [github.com/MoSaleh47](https://github.com/MoSaleh47)

## Licence

MIT — see [LICENSE](LICENSE). The licence covers this code only; BiomedCLIP and `WorldMedQA/V`
carry their own terms, which you should read before using either.

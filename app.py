import base64
import json
import os
import random
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image


DATASET_ID = "WorldMedQA/V"
MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
TEXT_ENCODER_REPO = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
APP_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = APP_DIR.parent
# Optional local copy of the weights, so the app can run without re-downloading
# ~750 MB from the hub. Keep it inside the repo (model/ is git-ignored); set
# BIOMEDCLIP_MODEL_DIR to point somewhere else. Falls back to MODEL_ID if absent.
MODEL_DIR = Path(os.environ.get("BIOMEDCLIP_MODEL_DIR", APP_DIR / "model"))
LOCAL_DATASETS = {
    "English TSV": APP_DIR / "brazil_english_processed.tsv",
    "Local TSV": APP_DIR / "brazil_local_processed.tsv",
}
OPTION_LETTERS = ("A", "B", "C", "D")


st.set_page_config(
    page_title="WorldMedQA-V with BiomedCLIP",
    page_icon="medkit",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .small-note { color: #5f6368; font-size: 0.85rem; }
    .metric-row { border-top: 1px solid #e7e7e7; padding-top: 0.75rem; }
    div[data-testid="stMetricValue"] { font-size: 1.45rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def decode_image(image_value: str) -> Image.Image:
    image_text = clean_text(image_value)
    if "," in image_text and image_text.lower().startswith("data:image"):
        image_text = image_text.split(",", 1)[1]
    image_bytes = base64.b64decode(image_text)
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def answer_options(row: pd.Series) -> dict[str, str]:
    return {letter: clean_text(row.get(letter, "")) for letter in OPTION_LETTERS}


def prompt_candidates(question: str, options: dict[str, str], style: str) -> list[str]:
    if style == "Answer only":
        return [f"this is a medical image showing {options[letter]}" for letter in OPTION_LETTERS]
    if style == "Question and answer":
        return [
            f"medical image. Question: {question} Answer: {options[letter]}"
            for letter in OPTION_LETTERS
        ]
    return [
        (
            "A biomedical image associated with the clinical question: "
            f"{question} The best answer is: {options[letter]}"
        )
        for letter in OPTION_LETTERS
    ]


@st.cache_data(show_spinner="Loading WorldMedQA-V data...")
def load_dataset_frame(source: str, local_file: str, max_rows: int | None) -> pd.DataFrame:
    if source == "Hugging Face":
        from datasets import load_dataset

        dataset = load_dataset(DATASET_ID, split="train")
        frame = dataset.to_pandas()
        if max_rows:
            frame = frame.head(max_rows)
    else:
        frame = pd.read_csv(local_file, sep="\t", dtype=str, nrows=max_rows or None)

    needed = ["index", "image", "question", "A", "B", "C", "D", "answer", "correct_option", "split"]
    for column in needed:
        if column not in frame.columns:
            frame[column] = ""

    frame = frame[needed].copy()
    frame["correct_option"] = frame["correct_option"].map(clean_text).str.upper()
    frame["split"] = frame["split"].map(clean_text)
    return frame


def load_local_biomedclip(local_dir: Path, device: str):
    import torch
    from open_clip import create_model_and_transforms, get_tokenizer
    from open_clip.factory import _MODEL_CONFIGS

    config_path = local_dir / "open_clip_config.json"
    weights_path = local_dir / "open_clip_pytorch_model.bin"
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    model_name = "biomedclip_local"
    text_cfg = config["model_cfg"].setdefault("text_cfg", {})
    text_snapshot = find_hf_snapshot(TEXT_ENCODER_REPO)
    if text_snapshot:
        text_cfg["hf_model_name"] = str(text_snapshot)
    text_cfg["hf_tokenizer_name"] = str(local_dir)

    _MODEL_CONFIGS[model_name] = config["model_cfg"]
    model, _, preprocess = create_model_and_transforms(
        model_name=model_name,
        pretrained=str(weights_path),
        pretrained_hf=False,
        **{f"image_{key}": value for key, value in config["preprocess_cfg"].items()},
    )
    tokenizer = get_tokenizer(model_name)
    model.to(torch.device(device))
    model.eval()
    return model, preprocess, tokenizer


def find_hf_snapshot(repo_id: str) -> Path | None:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_cache = cache_root / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not repo_cache.exists():
        return None

    snapshots = [path for path in repo_cache.iterdir() if (path / "config.json").exists()]
    if not snapshots:
        return None
    return max(snapshots, key=lambda path: path.stat().st_mtime)


@st.cache_resource(show_spinner="Loading BiomedCLIP model...")
def load_model(use_local_if_available: bool, device: str):
    import torch
    from open_clip import create_model_from_pretrained, get_tokenizer

    local_weights = MODEL_DIR / "open_clip_pytorch_model.bin"
    if use_local_if_available and local_weights.exists():
        model, preprocess, tokenizer = load_local_biomedclip(MODEL_DIR, device)
        return model, preprocess, tokenizer, "local weights"

    model, preprocess = create_model_from_pretrained(MODEL_ID)
    tokenizer = get_tokenizer(MODEL_ID)
    model.to(torch.device(device))
    model.eval()
    return model, preprocess, tokenizer, "Hugging Face Hub"


def predict_answer(
    model,
    preprocess,
    tokenizer,
    image: Image.Image,
    question: str,
    options: dict[str, str],
    prompt_style: str,
    device: str,
) -> pd.DataFrame:
    import torch

    candidates = prompt_candidates(question, options, prompt_style)
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    text_tensor = tokenizer(candidates, context_length=256).to(device)

    with torch.no_grad():
        image_features, text_features, logit_scale = model(image_tensor, text_tensor)
        probabilities = (logit_scale * image_features @ text_features.t()).softmax(dim=-1)[0]

    rows = []
    for letter, score, prompt in zip(OPTION_LETTERS, probabilities.detach().cpu().tolist(), candidates):
        rows.append(
            {
                "option": letter,
                "answer": options[letter],
                "score": float(score),
                "prompt": prompt,
            }
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def run_batch_eval(
    frame: pd.DataFrame,
    sample_size: int,
    model,
    preprocess,
    tokenizer,
    prompt_style: str,
    device: str,
) -> pd.DataFrame:
    if sample_size >= len(frame):
        positions = list(range(len(frame)))
    else:
        positions = random.sample(range(len(frame)), sample_size)

    progress = st.progress(0, text="Scoring examples...")
    rows = []
    for step, pos in enumerate(positions, start=1):
        row = frame.iloc[pos]
        image = decode_image(row["image"])
        options = answer_options(row)
        scores = predict_answer(
            model,
            preprocess,
            tokenizer,
            image,
            clean_text(row["question"]),
            options,
            prompt_style,
            device,
        )
        prediction = scores.iloc[0]["option"]
        correct = clean_text(row["correct_option"])
        rows.append(
            {
                "dataset_index": clean_text(row["index"]),
                "prediction": prediction,
                "correct_option": correct,
                "is_correct": prediction == correct,
                "confidence": scores.iloc[0]["score"],
                "question": clean_text(row["question"]),
            }
        )
        progress.progress(step / len(positions), text=f"Scored {step}/{len(positions)} examples")
    progress.empty()
    return pd.DataFrame(rows)


st.title("WorldMedQA-V with BiomedCLIP")
st.caption("Research demo for multiple-choice medical image QA. Not for clinical decision-making.")

with st.sidebar:
    st.header("Data")
    data_source = st.radio("Dataset source", ["Local TSV", "Hugging Face"], horizontal=True)
    local_label = st.selectbox("Local file", list(LOCAL_DATASETS.keys()), disabled=data_source != "Local TSV")
    max_rows = st.number_input("Rows to load", min_value=50, max_value=2000, value=1200, step=50)

    st.header("Model")
    device_default = "cuda"
    try:
        import torch

        device_default = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device_default = "cpu"
    device = st.selectbox("Device", ["cpu", "cuda"], index=0 if device_default == "cpu" else 1)
    use_local = st.checkbox("Use local weights if present", value=True)
    local_weights = MODEL_DIR / "open_clip_pytorch_model.bin"
    if local_weights.exists():
        st.caption(f"Local BiomedCLIP weights found: {local_weights.stat().st_size / 1_000_000:.0f} MB")
    else:
        st.caption("Local BiomedCLIP weights not found; the app will use Hugging Face Hub.")
    prompt_style = st.selectbox(
        "Text prompt",
        ["Question and answer", "Clinical caption", "Answer only"],
        help="BiomedCLIP scores image-text similarity. The question and each answer choice become candidate texts.",
    )

    st.header("Evaluation")
    batch_size = st.slider("Batch examples", min_value=3, max_value=50, value=10)


local_path = str(LOCAL_DATASETS[local_label])
if data_source == "Local TSV" and not Path(local_path).exists():
    st.error(f"Local dataset not found: {local_path}")
    st.stop()

try:
    df = load_dataset_frame(data_source, local_path, int(max_rows))
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

if df.empty:
    st.error("Dataset loaded, but it contains no rows.")
    st.stop()

if "sample_pos" not in st.session_state:
    st.session_state.sample_pos = 0

with st.sidebar:
    if st.button("Random example", use_container_width=True):
        st.session_state.sample_pos = random.randrange(len(df))
    st.session_state.sample_pos = st.number_input(
        "Row position",
        min_value=0,
        max_value=len(df) - 1,
        value=min(st.session_state.sample_pos, len(df) - 1),
        step=1,
    )

row = df.iloc[int(st.session_state.sample_pos)]
question = clean_text(row["question"])
options = answer_options(row)
correct_option = clean_text(row["correct_option"])
image = decode_image(row["image"])

left, right = st.columns([0.9, 1.1], gap="large")
with left:
    st.image(image, caption=f"Dataset index {clean_text(row['index'])}", use_container_width=True)
    st.markdown(
        f"<div class='small-note'>Split: {clean_text(row['split']) or 'unknown'}</div>",
        unsafe_allow_html=True,
    )

with right:
    st.subheader("Question")
    st.write(question)
    st.subheader("Choices")
    option_frame = pd.DataFrame(
        [{"option": letter, "answer": options[letter]} for letter in OPTION_LETTERS]
    )
    st.dataframe(option_frame, hide_index=True, use_container_width=True)
    st.info(f"Ground truth: {correct_option}", icon=":material/check_circle:")

model = preprocess = tokenizer = model_source = None
run_prediction = st.button("Run VLM prediction", type="primary")

if run_prediction:
    try:
        model, preprocess, tokenizer, model_source = load_model(use_local, device)
        scores = predict_answer(model, preprocess, tokenizer, image, question, options, prompt_style, device)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    prediction = scores.iloc[0]["option"]
    is_correct = prediction == correct_option

    metric_cols = st.columns(3)
    metric_cols[0].metric("Prediction", prediction)
    metric_cols[1].metric("Ground truth", correct_option)
    metric_cols[2].metric("Result", "Correct" if is_correct else "Incorrect")
    st.caption(f"Model loaded from {model_source}.")

    chart_data = scores.set_index("option")["score"]
    st.bar_chart(chart_data)
    st.dataframe(
        scores[["option", "answer", "score"]],
        hide_index=True,
        use_container_width=True,
        column_config={"score": st.column_config.ProgressColumn("score", min_value=0.0, max_value=1.0)},
    )

with st.expander("Batch evaluation"):
    st.write("Run a small random sample to compare model predictions with ground truth labels.")
    if st.button("Run batch evaluation"):
        try:
            model, preprocess, tokenizer, model_source = load_model(use_local, device)
            batch_results = run_batch_eval(
                df,
                batch_size,
                model,
                preprocess,
                tokenizer,
                prompt_style,
                device,
            )
        except Exception as exc:
            st.error(f"Batch evaluation failed: {exc}")
            st.stop()

        accuracy = batch_results["is_correct"].mean()
        st.metric("Accuracy on sampled rows", f"{accuracy:.1%}")
        st.caption(f"Model loaded from {model_source}.")
        st.dataframe(batch_results, hide_index=True, use_container_width=True)

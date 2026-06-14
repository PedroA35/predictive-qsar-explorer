# Predictive Therapeutic Target Explorer and QSAR (Cheminformatics)
# Phase 1 + 2 + 4 + 5 — PubChem lookup + RDKit 2D viz + QSAR prediction + Multi-target comparison

import streamlit as st
import requests
import numpy as np
import joblib
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw, rdFingerprintGenerator

# ── Constants ──────────────────────────────────────────────────────────────────
FP_RADIUS = 2
FP_NBITS  = 2048

TARGETS = {
    "Thrombin (CHEMBL204)": {
        "model_path": "model_thrombin.pkl",
        "description": "Serine protease involved in blood coagulation.",
        "disease": "Cardiovascular / Thrombosis",
        "emoji": "🩸",
    },
    "BACE-1 (CHEMBL4822)": {
        "model_path": "model_bace1.pkl",
        "description": "Beta-secretase 1, protease involved in amyloid-β production.",
        "disease": "Alzheimer's Disease",
        "emoji": "🧠",
    },
    "CDK2 (CHEMBL301)": {
        "model_path": "model_cdk2.pkl",
        "description": "Cyclin-dependent kinase 2, involved in cell cycle regulation.",
        "disease": "Cancer (Cell Cycle)",
        "emoji": "🔬",
    },
    "EGFR (CHEMBL203)": {
        "model_path": "model_egfr.pkl",
        "description": "Epidermal growth factor receptor, a key cancer drug target.",
        "disease": "Cancer (Growth Factor)",
        "emoji": "🎯",
    },
}

MODEL_TYPE = {
    "model_thrombin.pkl": "RF",
    "model_bace1.pkl":    "GB",
    "model_cdk2.pkl":     "RF",
    "model_egfr.pkl":     "RF",
}

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_NBITS)

# ── Helper functions ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        return None, f"Model file `{model_path}` not found."
    try:
        return joblib.load(path), None
    except Exception as e:
        return None, f"Failed to load model: {e}"

def smiles_to_fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(morgan_gen.GetFingerprint(mol)).reshape(1, -1)

def predict_activity(smiles: str, model):
    fp = smiles_to_fingerprint(smiles)
    if fp is None:
        return None, None, "Invalid SMILES — could not generate fingerprint."
    proba = model.predict_proba(fp)[0]
    label = model.predict(fp)[0]
    return label, proba, None

def name_to_smiles(name: str):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(name.strip())}/property/CanonicalSMILES/JSON"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, timeout=8, headers=headers)
        if r.status_code == 200:
            props = r.json()["PropertyTable"]["Properties"][0]
            smiles = (
                props.get("CanonicalSMILES")
                or props.get("IsomericSMILES")
                or props.get("ConnectivitySMILES")
            )
            if smiles:
                return smiles, None
            return None, f"No SMILES found for '{name}'."
        return None, f"'{name}' not found in PubChem (status {r.status_code})."
    except requests.exceptions.ConnectionError:
        return None, "No internet connection. Please check your network."
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try again."
    except Exception as e:
        return None, f"Unexpected error: {e}"

def get_synonyms(name: str):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(name.strip())}/synonyms/JSON"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, timeout=8, headers=headers)
        if r.status_code == 200:
            all_synonyms = r.json()["InformationList"]["Information"][0]["Synonym"]
            return [
                s for s in all_synonyms
                if len(s) <= 40
                and not s[0].isdigit()
                and s[:4].count("-") == 0
                and s.isascii()
                and s.lower() != name.strip().lower()
            ][:8]
        return []
    except:
        return []

def smiles_to_image(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=(400, 300))

def show_2d_structure(img, caption: str):
    st.markdown("**2D Structure:**")
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.image(img, caption=caption)
    with st.expander("🔬 How to read the 2D structure"):
        st.markdown("""
**Atom colours:**
- ⚫ **Black / no label** — Carbon (C)
- 🔵 **Blue** — Nitrogen (N)
- 🔴 **Red** — Oxygen (O)
- 🟡 **Yellow** — Sulfur (S)
- 🟢 **Green** — Fluorine (F) or Chlorine (Cl)
- ⚪ **White** — Hydrogen (H), usually omitted

**Bonds:** single line = single bond · double line = double bond · triple line = triple bond

**Rings:** hexagon = 6-membered ring · circle inside ring = aromatic
        """)

# ── Single-target prediction display ──────────────────────────────────────────
def show_prediction(smiles: str, model, target_name: str):
    with st.spinner("Running QSAR prediction..."):
        label, proba, error = predict_activity(smiles, model)
    if error:
        st.error(error)
        return
    prob_active   = proba[1]
    prob_inactive = proba[0]
    st.divider()
    st.subheader(f"🎯 QSAR Prediction — {target_name}")
    col1, col2 = st.columns(2)
    with col1:
        if label == 1:
            st.success("**Active** ✅")
        else:
            st.error("**Inactive** ❌")
    with col2:
        st.metric("Probability of Activity", f"{prob_active*100:.1f}%")
    st.progress(float(prob_active), text=f"Active: {prob_active*100:.1f}%  |  Inactive: {prob_inactive*100:.1f}%")
    with st.expander("ℹ️ How to interpret this result"):
        if label == 1:
            st.markdown(f"**In simple terms:** The model predicts this molecule is likely to **inhibit {target_name}** at a biologically relevant concentration.")
        else:
            st.markdown(f"**In simple terms:** The model predicts this molecule is **unlikely to significantly inhibit {target_name}** at typical drug concentrations.")
        st.markdown("---")
        st.markdown(f"""
**Technical details:**
- **Active** = predicted IC₅₀ < 10 µM for **{target_name}**
- **Inactive** = predicted IC₅₀ ≥ 10 µM
- **IC₅₀**: concentration needed to inhibit 50% of target activity — lower = more potent
- **Probability bar**: values above 80% indicate high-confidence predictions
- This is a **computational prediction** — always validate experimentally
        """)

# ── Multi-target comparison table ─────────────────────────────────────────────
def show_comparison_table(smiles: str, selected_targets: list):
    st.divider()
    st.subheader("📊 Multi-Target Comparison")

    results = []
    errors  = []

    with st.spinner("Running predictions across selected targets..."):
        for target_name in selected_targets:
            cfg   = TARGETS[target_name]
            model, model_error = load_model(cfg["model_path"])
            if model_error:
                errors.append(f"⚠️ {target_name}: {model_error}")
                continue
            label, proba, err = predict_activity(smiles, model)
            if err:
                errors.append(f"⚠️ {target_name}: {err}")
                continue
            results.append({
                "emoji":        cfg["emoji"],
                "target":       target_name,
                "disease":      cfg["disease"],
                "model_type":   MODEL_TYPE.get(cfg["model_path"], "—"),
                "label":        label,
                "prob_active":  proba[1],
                "prob_inactive":proba[0],
            })

    for e in errors:
        st.warning(e)

    if not results:
        st.error("No predictions could be made. Check that model files are present.")
        return

    # Sort by probability of activity descending
    results.sort(key=lambda x: x["prob_active"], reverse=True)

    # ── Table header ──────────────────────────────────────────────────────────
    col_t, col_d, col_m, col_r, col_p, col_b = st.columns([2.4, 2, 1.5, 1.1, 1.2, 2.5])
    col_t.markdown("**Target**")
    col_d.markdown("**Disease Area**")
    col_m.markdown("**Model**")
    col_r.markdown("**Result**")
    col_p.markdown("**P(active) %**")
    col_b.markdown("**Confidence**")
    st.markdown("<hr style='margin:4px 0 8px 0'>", unsafe_allow_html=True)

    # ── Table rows ────────────────────────────────────────────────────────────
    for r in results:
        col_t, col_d, col_m, col_r, col_p, col_b = st.columns([2.4, 2, 1.5, 1.1, 1.2, 2.5])
        col_t.markdown(f"{r['emoji']} {r['target']}")
        col_d.markdown(r["disease"])
        col_m.markdown(f"`{r['model_type']}`")
        if r["label"] == 1:
            col_r.markdown("✅ **Active**")
        else:
            col_r.markdown("❌ Inactive")
        col_p.markdown(f"**{r['prob_active']*100:.1f}%**")
        col_b.progress(float(r["prob_active"]))

    # ── Summary insight ───────────────────────────────────────────────────────
    st.divider()
    active_targets = [r for r in results if r["label"] == 1]
    if active_targets:
        best = active_targets[0]
        st.success(
            f"🏆 **Highest predicted activity:** {best['emoji']} {best['target']} "
            f"— {best['prob_active']*100:.1f}% probability of inhibition"
        )
        if len(active_targets) > 1:
            others = ", ".join(f"{r['emoji']} {r['target']}" for r in active_targets[1:])
            st.info(f"Also predicted **active** against: {others}")
    else:
        st.warning("This molecule was predicted **inactive** against all selected targets.")

    with st.expander("ℹ️ How to read the comparison table"):
        st.markdown("""
**In simple terms:**
Each row shows how likely this molecule is to block a given biological target.
The table is sorted from most to least promising — the higher the bar, the stronger the predicted
interaction. If the molecule scores **Active** on multiple targets, it may affect more than one
biological pathway simultaneously.

---

**Technical details:**
- **P(active) %**: probability (0–100%) that the molecule inhibits the target at IC₅₀ < 10 µM —
  the lower the IC₅₀, the more potent the compound needs to be to qualify as active
- **IC₅₀** (Inhibitory Concentration 50%): concentration required to inhibit 50% of the target's
  activity; values below 10 µM are considered pharmacologically relevant
- **Results are sorted** from highest to lowest predicted probability of activity
- **Model type** reflects the best-performing algorithm selected during training for each target
- A molecule predicted **active against multiple targets** may indicate **polypharmacology**
  (intentional multi-target design) or **off-target effects** (unintended binding)
- This is a **computational screening tool** — use results to prioritise candidates,
  not as final biological conclusions; always validate experimentally
        """)


# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QSAR Explorer",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 Predictive Therapeutic Target & QSAR Explorer")
st.markdown("""
Analyze chemical molecules by **name** or **SMILES** string.
Visualize the 2D structure and predict biological activity against therapeutic targets.
""")
st.divider()

# ── Mode selector ──────────────────────────────────────────────────────────────
st.subheader("🔬 Prediction Mode")
mode = st.radio(
    "Choose prediction mode:",
    options=["Single target", "Multi-target comparison"],
    horizontal=True,
    help="Single target: detailed prediction for one enzyme. Multi-target: side-by-side comparison table."
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE TARGET MODE
# ══════════════════════════════════════════════════════════════════════════════
if mode == "Single target":

    st.subheader("🎯 Select Therapeutic Target")
    selected_target = st.selectbox(
        "Target:",
        options=list(TARGETS.keys()),
        help="Select the biological target to predict activity against."
    )
    target_cfg = TARGETS[selected_target]
    st.caption(f"{target_cfg['emoji']} {target_cfg['description']}  |  **{target_cfg['disease']}**")
    model, model_error = load_model(target_cfg["model_path"])
    if model_error:
        st.warning(f"⚠️ {model_error}")
    st.divider()

    st.subheader("🔍 Search Molecule")
    input_mode = st.radio("Search by:", options=["Molecule name", "SMILES string"], horizontal=True)

    if input_mode == "Molecule name":
        name_input = st.text_input(
            label="🔎 Enter molecule name:",
            placeholder="e.g. aspirin, ibuprofen, acetaminophen, caffeine",
        )
        if st.button("Search & Predict", type="primary"):
            if not name_input.strip():
                st.warning("Please enter a molecule name before searching.")
            else:
                with st.spinner("Looking up in PubChem..."):
                    smiles_result, error = name_to_smiles(name_input.strip())
                if error:
                    st.error(error)
                else:
                    st.success(f"Molecule found for **{name_input}**!")
                    with st.spinner("Fetching synonyms..."):
                        synonyms = get_synonyms(name_input.strip())
                    if synonyms:
                        st.markdown("**Other searchable names:**")
                        st.markdown("  ".join(f"`{s}`" for s in synonyms))
                    st.markdown("**SMILES:**")
                    st.code(smiles_result, language=None)
                    img = smiles_to_image(smiles_result)
                    if img:
                        show_2d_structure(img, name_input)
                    if model:
                        show_prediction(smiles_result, model, selected_target)
                    else:
                        st.warning("Model not loaded — prediction unavailable.")
    else:
        smiles_input = st.text_input(
            label="🔬 Enter molecule SMILES:",
            placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O  (Aspirin)",
        )
        if st.button("Analyze & Predict", type="primary"):
            if not smiles_input.strip():
                st.warning("Please enter a SMILES string before analyzing.")
            else:
                st.code(smiles_input, language=None)
                img = smiles_to_image(smiles_input.strip())
                if img:
                    show_2d_structure(img, "Molecule")
                else:
                    st.error("Invalid SMILES string — could not render structure.")
                    st.stop()
                if model:
                    show_prediction(smiles_input.strip(), model, selected_target)
                else:
                    st.warning("Model not loaded — prediction unavailable.")

# ══════════════════════════════════════════════════════════════════════════════
# MULTI-TARGET COMPARISON MODE
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.subheader("🎯 Select Targets to Compare")

    all_target_names = list(TARGETS.keys())

    n_targets = st.slider(
        "How many targets do you want to compare?",
        min_value=2,
        max_value=len(all_target_names),
        value=len(all_target_names),
        help="Drag to choose how many targets appear in the comparison."
    )

    selected_targets = st.multiselect(
        f"Select exactly {n_targets} target(s):",
        options=all_target_names,
        default=all_target_names[:n_targets],
        format_func=lambda t: f"{TARGETS[t]['emoji']} {t}",
        max_selections=n_targets,
    )

    if len(selected_targets) < n_targets:
        st.info(f"Select {n_targets - len(selected_targets)} more target(s) to proceed.")

    # Show target info cards
    if selected_targets:
        cols = st.columns(len(selected_targets))
        for col, t in zip(cols, selected_targets):
            cfg = TARGETS[t]
            col.markdown(f"**{cfg['emoji']} {t.split('(')[0].strip()}**")
            col.caption(cfg["disease"])

    st.divider()
    st.subheader("🔍 Search Molecule")
    input_mode = st.radio("Search by:", options=["Molecule name", "SMILES string"], horizontal=True)

    run_disabled = len(selected_targets) < n_targets

    if input_mode == "Molecule name":
        name_input = st.text_input(
            label="🔎 Enter molecule name:",
            placeholder="e.g. imatinib, gefitinib, warfarin, donepezil",
        )
        if st.button("Search & Compare", type="primary", disabled=run_disabled):
            if not name_input.strip():
                st.warning("Please enter a molecule name.")
            else:
                with st.spinner("Looking up in PubChem..."):
                    smiles_result, error = name_to_smiles(name_input.strip())
                if error:
                    st.error(error)
                else:
                    st.success(f"Molecule found for **{name_input}**!")
                    with st.spinner("Fetching synonyms..."):
                        synonyms = get_synonyms(name_input.strip())
                    if synonyms:
                        st.markdown("**Other searchable names:**")
                        st.markdown("  ".join(f"`{s}`" for s in synonyms))
                    st.markdown("**SMILES:**")
                    st.code(smiles_result, language=None)
                    img = smiles_to_image(smiles_result)
                    if img:
                        show_2d_structure(img, name_input)
                    show_comparison_table(smiles_result, selected_targets)
    else:
        smiles_input = st.text_input(
            label="🔬 Enter molecule SMILES:",
            placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O  (Aspirin)",
        )
        if st.button("Analyze & Compare", type="primary", disabled=run_disabled):
            if not smiles_input.strip():
                st.warning("Please enter a SMILES string.")
            else:
                st.code(smiles_input, language=None)
                img = smiles_to_image(smiles_input.strip())
                if img:
                    show_2d_structure(img, "Molecule")
                else:
                    st.error("Invalid SMILES string — could not render structure.")
                    st.stop()
                show_comparison_table(smiles_input.strip(), selected_targets)

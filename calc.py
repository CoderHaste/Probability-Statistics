import streamlit as st
import numpy as np
import pandas as pd

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StatLab · Correlation & Regression",
    page_icon="📐",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #0d0f14;
    --surface:   #13161e;
    --card:      #181c27;
    --border:    #252b3b;
    --accent:    #4fd1c5;
    --accent2:   #f6ad55;
    --text:      #e2e8f0;
    --muted:     #718096;
    --danger:    #fc8181;
    --success:   #68d391;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem !important; max-width: 1280px; }

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 900;
    letter-spacing: -1px;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 3px;
    color: var(--muted);
    text-transform: uppercase;
    margin-top: 0.4rem;
}
.hero-bar {
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2), transparent);
    margin: 1.4rem 0 2.2rem;
    border: none;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 12px !important;
    padding: 6px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: var(--muted) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    letter-spacing: 1px !important;
    padding: 10px 24px !important;
    border: none !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e3a5f, #1e3a4a) !important;
    color: var(--accent) !important;
    box-shadow: 0 2px 12px rgba(79,209,197,0.15) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 0 !important; }

.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 3px;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    margin-top: 1.8rem;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #38b2ac 100%) !important;
    color: #0d0f14 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    letter-spacing: 1px !important;
    font-weight: 500 !important;
    padding: 10px 28px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 16px rgba(79,209,197,0.2) !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 24px rgba(79,209,197,0.4) !important;
    transform: translateY(-1px) !important;
}

.result-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.result-card-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 3px;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.result-value {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
    margin: 0;
}
.result-label { font-size: 0.8rem; color: var(--muted); margin-top: 0.3rem; }

.equation-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent2);
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    font-family: 'DM Mono', monospace;
    font-size: 1.05rem;
    color: var(--accent2);
    letter-spacing: 0.5px;
    margin: 0.8rem 0;
}
.interp-box {
    background: #0f2027;
    border: 1px solid #1a3a4a;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    font-size: 0.88rem;
    color: #a0c4cf;
    line-height: 1.7;
    margin-top: 1rem;
}
.steps-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    line-height: 2;
    white-space: pre-wrap;
    margin-top: 0.8rem;
}
.warn-box {
    background: rgba(252,129,129,0.08);
    border: 1px solid rgba(252,129,129,0.3);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: var(--danger);
    font-size: 0.85rem;
    margin: 0.8rem 0;
}
.sum-table {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin-top: 1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
}
.sum-row {
    display: flex;
    justify-content: space-between;
    padding: 0.35rem 0;
    border-bottom: 1px solid var(--border);
    color: var(--text);
}
.sum-row:last-child { border-bottom: none; }
.sum-key { color: var(--accent2); }
.sum-val { color: var(--accent); font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<p class="hero-sub">Probability & Statistics · Problem Solver</p>
<h1 class="hero-title">StatLab</h1>
<hr class="hero-bar">
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_values(text: str, allow_categorical=False):
    import re
    tokens = re.split(r'[\s,]+', text.strip())
    vals = []
    labels = []   # original labels if categorical
    for t in tokens:
        if t == '':
            continue
        try:
            v = float(t)
            vals.append(v)
            labels.append(t)
        except ValueError:
            if allow_categorical and t.isalpha():
                # Map A→1, B→2, ... preserving case
                v = ord(t.upper()) - ord('A') + 1
                vals.append(v)
                labels.append(t.upper())
            else:
                return None, None, f"Cannot parse '{t}'."
    if not vals:
        return None, None, "No values entered."
    return np.array(vals), labels, None


def strength_label(r):
    ar = abs(r)
    if ar >= 0.9: return "Very Strong"
    elif ar >= 0.7: return "Strong"
    elif ar >= 0.5: return "Moderate"
    elif ar >= 0.3: return "Weak"
    else: return "Very Weak"


def direction(r):
    return "Positive" if r >= 0 else "Negative"


def pearson(a, b):
    a_m, b_m = np.mean(a), np.mean(b)
    num = np.sum((a - a_m) * (b - b_m))
    den = np.sqrt(np.sum((a - a_m) ** 2) * np.sum((b - b_m) ** 2))
    return (num / den if den != 0 else 0.0), a_m, b_m, num, den


def render_summations(sums: dict):
    rows = "".join(
        f'<div class="sum-row"><span class="sum-key">{k}</span><span class="sum-val">{v}</span></div>'
        for k, v in sums.items()
    )
    st.markdown(f'<div class="sum-table">{rows}</div>', unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["  📊  CORRELATION COEFFICIENT  ", "  📈  LINEAR REGRESSION  "])

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 · CORRELATION COEFFICIENT                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab1:
    st.markdown("")
    st.markdown("""
    Calculate the **Pearson correlation coefficient**.
    - **2 variables**: relationship between **Y and X₁**
    - **3 variables**: relationships between **Y & X₁**, **Y & X₂**, and **X₁ & X₂**

    Y accepts letters (A, B, C …) which are converted to ranks (1, 2, 3 …).
    """)

    st.markdown('<p class="section-label">Enter your data</p>', unsafe_allow_html=True)
    st.caption("Separate values with commas or spaces. All series must have the same length.")

    col_y, col_x1, col_x2 = st.columns(3)
    with col_y:
        st.markdown("**Y** (dependent — letters or numbers)")
        y_raw = st.text_area("Y values", placeholder="e.g. A, B, C, D  or  2.1, 3.5", height=160, key="corr_y", label_visibility="collapsed")
    with col_x1:
        st.markdown("**X₁** (predictor 1)")
        x1_raw = st.text_area("X1 values", placeholder="e.g. 1.0, 2.0, 3.0, 4.0", height=160, key="corr_x1", label_visibility="collapsed")
    with col_x2:
        st.markdown("**X₂** (predictor 2, optional)")
        x2_raw = st.text_area("X2 values", placeholder="e.g. 0.5, 1.5, 2.5, 3.5", height=160, key="corr_x2", label_visibility="collapsed")

    show_steps_corr = st.checkbox("Show calculation steps", key="steps_corr")
    st.markdown("")
    calc_corr = st.button("⟶  Calculate Correlation", key="btn_corr")

    if calc_corr:
        errors = []
        Y, y_labels, err = parse_values(y_raw, allow_categorical=True)
        if err: errors.append(f"Y: {err}")
        X1, _, err = parse_values(x1_raw)
        if err: errors.append(f"X₁: {err}")

        has_x2 = x2_raw.strip() != ""
        if has_x2:
            X2, _, err = parse_values(x2_raw)
            if err: errors.append(f"X₂: {err}")
        else:
            X2 = None

        if errors:
            for e in errors:
                st.markdown(f'<div class="warn-box">⚠ {e}</div>', unsafe_allow_html=True)
        else:
            lengths = [len(Y), len(X1)] + ([len(X2)] if X2 is not None else [])
            if len(set(lengths)) != 1:
                st.markdown('<div class="warn-box">⚠ All series must have the same number of values.</div>', unsafe_allow_html=True)
            elif len(Y) < 2:
                st.markdown('<div class="warn-box">⚠ At least 2 data points required.</div>', unsafe_allow_html=True)
            else:
                n = len(Y)
                r_yx1, y_mean, x1_mean, num_yx1, den_yx1 = pearson(Y, X1)

                # ── Summation table ──
                st.markdown('<p class="section-label">Summations</p>', unsafe_allow_html=True)
                sums = {
                    "N": n,
                    "ΣY":    f"{np.sum(Y):.4f}",
                    "ΣX₁":   f"{np.sum(X1):.4f}",
                    "ΣX₁²":  f"{np.sum(X1**2):.4f}",
                    "ΣX₁Y":  f"{np.sum(X1*Y):.4f}",
                }
                if has_x2:
                    sums["ΣX₂"]   = f"{np.sum(X2):.4f}"
                    sums["ΣX₂²"]  = f"{np.sum(X2**2):.4f}"
                    sums["ΣX₂Y"]  = f"{np.sum(X2*Y):.4f}"
                    sums["ΣX₁X₂"] = f"{np.sum(X1*X2):.4f}"
                render_summations(sums)

                # ── Data preview table with derived columns ──
                st.markdown('<p class="section-label">Data table with derived values</p>', unsafe_allow_html=True)
                preview = {"Y (orig)": y_labels, "Y (num)": Y, "X₁": X1, "X₁²": X1**2, "X₁·Y": X1*Y}
                if has_x2:
                    preview["X₂"]    = X2
                    preview["X₂²"]   = X2**2
                    preview["X₂·Y"]  = X2*Y
                    preview["X₁·X₂"] = X1*X2
                df_prev = pd.DataFrame(preview)
                # Add totals row
                totals = {"Y (orig)": "Σ", "Y (num)": np.sum(Y), "X₁": np.sum(X1),
                          "X₁²": np.sum(X1**2), "X₁·Y": np.sum(X1*Y)}
                if has_x2:
                    totals["X₂"]    = np.sum(X2)
                    totals["X₂²"]   = np.sum(X2**2)
                    totals["X₂·Y"]  = np.sum(X2*Y)
                    totals["X₁·X₂"] = np.sum(X1*X2)
                df_prev = pd.concat([df_prev, pd.DataFrame([totals])], ignore_index=True)
                st.dataframe(df_prev, use_container_width=True)

                # ── Correlation results ──
                st.markdown('<p class="section-label">Correlation Results</p>', unsafe_allow_html=True)

                if has_x2:
                    r_yx2, _, x2_mean, num_yx2, den_yx2 = pearson(Y, X2)
                    r_x1x2, _, _, num_x1x2, den_x1x2 = pearson(X1, X2)
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"""<div class="result-card">
                            <div class="result-card-title">r(Y, X₁)</div>
                            <p class="result-value">{r_yx1:.4f}</p>
                            <div class="result-label">{direction(r_yx1)} · {strength_label(r_yx1)}</div>
                        </div>""", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""<div class="result-card">
                            <div class="result-card-title">r(Y, X₂)</div>
                            <p class="result-value">{r_yx2:.4f}</p>
                            <div class="result-label">{direction(r_yx2)} · {strength_label(r_yx2)}</div>
                        </div>""", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"""<div class="result-card">
                            <div class="result-card-title">r(X₁, X₂)</div>
                            <p class="result-value">{r_x1x2:.4f}</p>
                            <div class="result-label">{direction(r_x1x2)} · {strength_label(r_x1x2)}</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown(f"""<div class="interp-box">
                        📌 <strong>Y vs X₁:</strong> r = {r_yx1:.4f} →
                        <strong>{strength_label(r_yx1)} {direction(r_yx1)}</strong> relationship.<br>
                        📌 <strong>Y vs X₂:</strong> r = {r_yx2:.4f} →
                        <strong>{strength_label(r_yx2)} {direction(r_yx2)}</strong> relationship.<br>
                        📌 <strong>X₁ vs X₂:</strong> r = {r_x1x2:.4f} →
                        <strong>{strength_label(r_x1x2)} {direction(r_x1x2)}</strong> relationship.
                    </div>""", unsafe_allow_html=True)

                    if show_steps_corr:
                        steps = f"""PEARSON CORRELATION — Y vs X₁
n = {n}   Ȳ = {y_mean:.4f}   X̄₁ = {x1_mean:.4f}
Σ(Y-Ȳ)(X₁-X̄₁) = {num_yx1:.4f}
√[Σ(Y-Ȳ)²·Σ(X₁-X̄₁)²] = {den_yx1:.4f}
r(Y,X₁) = {r_yx1:.4f}

PEARSON CORRELATION — Y vs X₂
X̄₂ = {x2_mean:.4f}
Σ(Y-Ȳ)(X₂-X̄₂) = {num_yx2:.4f}
√[Σ(Y-Ȳ)²·Σ(X₂-X̄₂)²] = {den_yx2:.4f}
r(Y,X₂) = {r_yx2:.4f}

PEARSON CORRELATION — X₁ vs X₂
Σ(X₁-X̄₁)(X₂-X̄₂) = {num_x1x2:.4f}
√[Σ(X₁-X̄₁)²·Σ(X₂-X̄₂)²] = {den_x1x2:.4f}
r(X₁,X₂) = {r_x1x2:.4f}"""
                        st.markdown(f'<div class="steps-box">{steps}</div>', unsafe_allow_html=True)

                else:
                    # 2-variable case
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"""<div class="result-card">
                            <div class="result-card-title">r(Y, X₁)</div>
                            <p class="result-value">{r_yx1:.4f}</p>
                            <div class="result-label">{direction(r_yx1)} · {strength_label(r_yx1)}</div>
                        </div>""", unsafe_allow_html=True)
                    with c2:
                        r2 = r_yx1 ** 2
                        st.markdown(f"""<div class="result-card">
                            <div class="result-card-title">r² (Coefficient of Determination)</div>
                            <p class="result-value">{r2:.4f}</p>
                            <div class="result-label">{r2*100:.1f}% variance explained</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown(f"""<div class="interp-box">
                        📌 r = {r_yx1:.4f} → <strong>{strength_label(r_yx1)} {direction(r_yx1)}</strong> linear relationship between Y and X₁.<br>
                        {'As X₁ increases, Y tends to increase.' if r_yx1 > 0 else 'As X₁ increases, Y tends to decrease.'}
                        X₁ explains <strong>{r_yx1**2*100:.1f}%</strong> of the variation in Y.
                    </div>""", unsafe_allow_html=True)

                    if show_steps_corr:
                        steps = f"""PEARSON CORRELATION — Y vs X₁
n = {n}
Ȳ  = {y_mean:.4f}    X̄₁ = {x1_mean:.4f}
ΣY  = {np.sum(Y):.4f}   ΣX₁ = {np.sum(X1):.4f}
ΣX₁² = {np.sum(X1**2):.4f}   ΣX₁Y = {np.sum(X1*Y):.4f}

Σ(Y-Ȳ)(X₁-X̄₁) = {num_yx1:.4f}
√[Σ(Y-Ȳ)²·Σ(X₁-X̄₁)²] = {den_yx1:.4f}
r = {num_yx1:.4f} / {den_yx1:.4f} = {r_yx1:.4f}
r² = {r_yx1**2:.4f}"""
                        st.markdown(f'<div class="steps-box">{steps}</div>', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 · LINEAR REGRESSION                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab2:
    st.markdown("")
    st.markdown("""
    Find the **simple linear regression** equation using the least squares method.

    **Formula used:**
    - **b = (N·Σxy − Σx·Σy) / (N·Σx² − (Σx)²)**
    - **a = (Σy·Σx² − Σx·Σxy) / (N·Σx² − (Σx)²)**
    - **Equation: ŷ = a + b·x**
    """)

    st.markdown('<p class="section-label">Enter your data</p>', unsafe_allow_html=True)
    st.caption("Enter X (independent) and Y (dependent) values. Separate with commas or spaces.")

    rc_x, rc_y = st.columns(2)
    with rc_x:
        st.markdown("**X** (independent variable)")
        rx_raw = st.text_area("X", placeholder="e.g. 1, 2, 3, 4, 5", height=160, key="reg_x", label_visibility="collapsed")
    with rc_y:
        st.markdown("**Y** (dependent variable)")
        ry_raw = st.text_area("Y", placeholder="e.g. 2.1, 3.8, 5.0, 6.5, 8.2", height=160, key="reg_y", label_visibility="collapsed")

    show_steps_reg = st.checkbox("Show calculation steps", key="steps_reg")
    st.markdown("")
    calc_reg = st.button("⟶  Find Regression Equation", key="btn_reg")

    if calc_reg:
        errors = []
        RX, _, err = parse_values(rx_raw)
        if err: errors.append(f"X: {err}")
        RY, _, err = parse_values(ry_raw)
        if err: errors.append(f"Y: {err}")

        if errors:
            for e in errors:
                st.markdown(f'<div class="warn-box">⚠ {e}</div>', unsafe_allow_html=True)
        else:
            if len(RX) != len(RY):
                st.markdown('<div class="warn-box">⚠ X and Y must have the same number of values.</div>', unsafe_allow_html=True)
            elif len(RX) < 2:
                st.markdown('<div class="warn-box">⚠ At least 2 data points required.</div>', unsafe_allow_html=True)
            else:
                n   = len(RX)
                Sx  = np.sum(RX)
                Sy  = np.sum(RY)
                Sx2 = np.sum(RX ** 2)
                Sy2 = np.sum(RY ** 2)
                Sxy = np.sum(RX * RY)

                denom = n * Sx2 - Sx ** 2

                if denom == 0:
                    st.markdown('<div class="warn-box">⚠ X values are all identical — regression undefined.</div>', unsafe_allow_html=True)
                else:
                    a = (Sy * Sx2 - Sx * Sxy) / denom
                    b = (n * Sxy - Sx * Sy) / denom

                    Y_hat   = a + b * RX
                    SS_res  = np.sum((RY - Y_hat) ** 2)
                    SS_tot  = np.sum((RY - np.mean(RY)) ** 2)
                    r2      = 1 - SS_res / SS_tot if SS_tot != 0 else 0.0
                    se      = np.sqrt(SS_res / (n - 2)) if n > 2 else 0.0

                    sign_b = "+" if b >= 0 else "−"
                    eq_str = f"ŷ = {a:.4f} {sign_b} {abs(b):.4f}·x"

                    # ── Summation table ──
                    st.markdown('<p class="section-label">Summations</p>', unsafe_allow_html=True)
                    sums = {
                        "N":    n,
                        "Σx":   f"{Sx:.4f}",
                        "Σy":   f"{Sy:.4f}",
                        "Σx²":  f"{Sx2:.4f}",
                        "Σy²":  f"{Sy2:.4f}",
                        "Σxy":  f"{Sxy:.4f}",
                        "(Σx)²":f"{Sx**2:.4f}",
                        "N·Σx²":f"{n*Sx2:.4f}",
                        "Denominator (N·Σx²−(Σx)²)": f"{denom:.4f}",
                    }
                    render_summations(sums)

                    # ── Data table ──
                    st.markdown('<p class="section-label">Data table with derived values</p>', unsafe_allow_html=True)
                    df_data = pd.DataFrame({
                        "x":   RX,
                        "y":   RY,
                        "x²":  RX**2,
                        "y²":  RY**2,
                        "x·y": RX*RY,
                    })
                    totals_row = {
                        "x":   Sx,  "y":  Sy,
                        "x²":  Sx2, "y²": Sy2,
                        "x·y": Sxy,
                    }
                    df_data = pd.concat([df_data, pd.DataFrame([totals_row])], ignore_index=True)
                    # Label last row
                    df_data.index = list(range(1, n + 1)) + ["Σ"]
                    st.dataframe(df_data.style.format("{:.4f}"), use_container_width=True)

                    # ── Equation & metrics ──
                    st.markdown('<p class="section-label">Regression Equation</p>', unsafe_allow_html=True)
                    st.markdown(f"""<div class="result-card">
                        <div class="result-card-title">Simple Linear Regression (Least Squares)</div>
                        <div class="equation-box">{eq_str}</div>
                        <div class="result-label">
                            a (intercept) = {a:.4f} &nbsp;|&nbsp; b (slope) = {b:.4f}
                        </div>
                    </div>""", unsafe_allow_html=True)

                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(f"""<div class="result-card">
                            <div class="result-card-title">R² (Goodness of Fit)</div>
                            <p class="result-value">{r2:.4f}</p>
                            <div class="result-label">{r2*100:.1f}% variance explained</div>
                        </div>""", unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""<div class="result-card">
                            <div class="result-card-title">Std Error of Estimate</div>
                            <p class="result-value">{se:.4f}</p>
                            <div class="result-label">Average prediction error</div>
                        </div>""", unsafe_allow_html=True)
                    with m3:
                        st.markdown(f"""<div class="result-card">
                            <div class="result-card-title">n (observations)</div>
                            <p class="result-value">{n}</p>
                            <div class="result-label">Data points used</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown(f"""<div class="interp-box">
                        📌 <strong>Equation:</strong> {eq_str}<br>
                        • For every 1-unit increase in x, y {'increases' if b > 0 else 'decreases'} by
                          <strong>{abs(b):.4f}</strong> units.<br>
                        • When x = 0, predicted y = <strong>{a:.4f}</strong> (intercept a).<br>
                        • The model explains <strong>{r2*100:.1f}%</strong> of total variation in y.
                    </div>""", unsafe_allow_html=True)

                    # ── Calculation steps ──
                    if show_steps_reg:
                        steps = f"""SIMPLE LINEAR REGRESSION — LEAST SQUARES
n = {n}

Summations:
  Σx   = {Sx:.4f}
  Σy   = {Sy:.4f}
  Σx²  = {Sx2:.4f}
  Σxy  = {Sxy:.4f}

Denominator = N·Σx² − (Σx)²
           = {n}×{Sx2:.4f} − ({Sx:.4f})²
           = {n*Sx2:.4f} − {Sx**2:.4f}
           = {denom:.4f}

b = (N·Σxy − Σx·Σy) / Denominator
  = ({n}×{Sxy:.4f} − {Sx:.4f}×{Sy:.4f}) / {denom:.4f}
  = ({n*Sxy:.4f} − {Sx*Sy:.4f}) / {denom:.4f}
  = {n*Sxy - Sx*Sy:.4f} / {denom:.4f}
  = {b:.4f}

a = (Σy·Σx² − Σx·Σxy) / Denominator
  = ({Sy:.4f}×{Sx2:.4f} − {Sx:.4f}×{Sxy:.4f}) / {denom:.4f}
  = ({Sy*Sx2:.4f} − {Sx*Sxy:.4f}) / {denom:.4f}
  = {Sy*Sx2 - Sx*Sxy:.4f} / {denom:.4f}
  = {a:.4f}

Equation: {eq_str}

R²  = {r2:.4f}   ({r2*100:.1f}% variance explained)
SE  = {se:.4f}"""
                        st.markdown(f'<div class="steps-box">{steps}</div>', unsafe_allow_html=True)

                    # ── Fitted values ──
                    st.markdown('<p class="section-label">Fitted values</p>', unsafe_allow_html=True)
                    df_fit = pd.DataFrame({
                        "x": RX,
                        "y (actual)": RY,
                        "ŷ (predicted)": Y_hat,
                        "Residual (y−ŷ)": RY - Y_hat,
                    })
                    df_fit.index = range(1, n + 1)
                    st.dataframe(df_fit.style.format("{:.4f}"), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem; padding-top:1.5rem; border-top:1px solid #252b3b;
     font-family:'DM Mono',monospace; font-size:0.7rem; color:#4a5568; text-align:center;
     letter-spacing:2px;">
STATLAB · PROBABILITY & STATISTICS SOLVER · PEARSON CORRELATION · LINEAR REGRESSION
</div>
""", unsafe_allow_html=True)
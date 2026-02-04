import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

from online_constrained_kmeans import OnlineConstrainedKMeans

FEATURES_DIR = "features_out"

DATASETS = {
    "animals": 3,  # clases esperadas
    "fruits": 4,
}

DESCRIPTORS = ["Momentos de Hu", "HOG", "SIFT"]  # (Embeddings lo dejamos aparte si luego lo conectas)


@dataclass
class LoadedData:
    X: np.ndarray
    y_true: Optional[np.ndarray]
    df_raw: pd.DataFrame


def _is_lfs_pointer(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(120)
        return "git-lfs" in head and "oid sha256" in head
    except Exception:
        return False


def load_csv_features(descriptor: str, dataset: str) -> LoadedData:
    file_map = {
        "Momentos de Hu": f"X_hu_{dataset}.csv",
        "HOG": f"X_hog_{dataset}.csv",
        "SIFT": f"X_sift_{dataset}.csv",
    }
    path = os.path.join(FEATURES_DIR, file_map[descriptor])

    if not os.path.exists(path):
        st.error(f"No existe el archivo: {path}")
        st.stop()

    if _is_lfs_pointer(path):
        st.error(
            "Tus CSV parecen punteros de **Git LFS**.\n\n"
            "Solución:\n"
            "1) Instala Git LFS\n"
            "2) En terminal dentro del repo:\n"
            "   - `git lfs install`\n"
            "   - `git lfs pull`\n"
        )
        st.stop()

    df = pd.read_csv(path)

    # columnas meta (si existen)
    meta_cols = ["dataset", "class", "file"]
    X = df.drop(columns=meta_cols, errors="ignore").to_numpy(dtype=np.float32)
    y = df["class"].to_numpy() if "class" in df.columns else None

    return LoadedData(X=X, y_true=y, df_raw=df)


def compute_max_sizes(y_true: Optional[np.ndarray], k: int, mode: str, uniform_max: int) -> np.ndarray:
    """
    mode:
      - 'ground_truth': cupo por cluster = distribución real del dataset (si y_true existe)
      - 'uniform': todos los clusters con el mismo cupo
    """
    if mode == "ground_truth":
        if y_true is None:
            raise ValueError("No hay 'class' (y_true) en el CSV para calcular cupos por ground truth.")
        classes, counts = np.unique(y_true, return_counts=True)
        if len(classes) != k:
            raise ValueError(f"Ground truth tiene {len(classes)} clases, pero k={k}. Usa k correcto o modo Uniforme.")
        order = np.argsort(classes)
        return counts[order].astype(int)

    return np.full(k, int(uniform_max), dtype=int)


def pca_2d_deterministic(X: np.ndarray) -> np.ndarray:
    """
    PCA determinístico: svd_solver='full' (sin random_state).
    """
    return PCA(n_components=2, svd_solver="full").fit_transform(X)


def init_stream_state(X_scaled: np.ndarray, y_true: Optional[np.ndarray]):
    # Sin semilla: el stream es determinístico, en orden 0..N-1
    st.session_state.stream_idx = np.arange(len(X_scaled))
    st.session_state.ptr = 0
    st.session_state.labels = []
    st.session_state.history = []
    st.session_state.y_true_stream = y_true


def do_one_step(X_scaled: np.ndarray, y_true: Optional[np.ndarray]):
    if st.session_state.ptr >= len(X_scaled):
        return False

    i = int(st.session_state.stream_idx[st.session_state.ptr])
    x = X_scaled[i]
    y_t = y_true[i] if y_true is not None else None

    label, info = st.session_state.model.partial_fit_one(x)

    st.session_state.labels.append(int(label))
    st.session_state.history.append(
        {
            "t": int(st.session_state.ptr),
            "index": i,
            "y_true": y_t,
            "assigned_cluster": int(label),
            "cluster_sizes_now": info["counts"].tolist(),
            "fallback_used": bool(info["fallback_used"]),
            "tried_order": info["tried_order"],
        }
    )
    st.session_state.ptr += 1
    return True


# ---------------- UI ----------------
st.set_page_config(page_title="Online Clustering con Restricción", layout="wide")

st.markdown(
    """
<style>
.big-title {font-size: 30px; font-weight: 800; margin-bottom: 4px;}
.subtle {color: #666; margin-top: 0px;}
.card {
  padding: 14px; border-radius: 16px; border: 1px solid #eee;
  background: white; box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}
.badge {
  display:inline-block; padding: 4px 10px; border-radius: 999px;
  border: 1px solid #eee; background: #fafafa; font-size: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">Sistema de Clustering Online con Restricción de Tamaño</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtle">Demostración clara: <b>instancia por instancia</b> + <b>cupo por cluster</b> + métricas + visualización.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Configuración")

    dataset = st.selectbox("Dataset", list(DATASETS.keys()))
    descriptor = st.selectbox("Método de extracción", DESCRIPTORS)

    default_k = DATASETS[dataset]
    k = st.number_input("Número de clusters (k)", min_value=2, max_value=15, value=int(default_k), step=1)

    restriction_mode_label = st.radio("Restricción de tamaño", ["Por Ground Truth", "Uniforme"], index=0)
    mode = "ground_truth" if restriction_mode_label == "Por Ground Truth" else "uniform"
    uniform_max = st.number_input("Cupo máximo por cluster (Uniforme)", min_value=1, value=50, step=1)

    st.divider()
    reset_btn = st.button("🔁 Reiniciar proceso", use_container_width=True)
    step_btn = st.button("➡️ Procesar 1 instancia", use_container_width=True)
    run_all_btn = st.button("▶️ Procesar todo", use_container_width=True)

data = load_csv_features(descriptor, dataset)
X = data.X
y_true = data.y_true

X_scaled = StandardScaler().fit_transform(X)

# cupos
try:
    max_sizes = compute_max_sizes(y_true, int(k), mode, int(uniform_max))
except Exception as e:
    st.error(str(e))
    st.stop()

# init session
if "model" not in st.session_state or reset_btn:
    st.session_state.model = OnlineConstrainedKMeans(n_clusters=int(k), max_sizes=max_sizes)
    init_stream_state(X_scaled, y_true)

# acciones
if step_btn:
    do_one_step(X_scaled, y_true)

if run_all_btn:
    prog = st.progress(0)
    total = len(X_scaled)
    while st.session_state.ptr < total:
        do_one_step(X_scaled, y_true)
        prog.progress(int(100 * st.session_state.ptr / total))

# -------- Tabs --------
tab1, tab2, tab3, tab4 = st.tabs(["🧪 Online (Paso a Paso)", "📦 Cupos y Clusters", "📈 Métricas y PCA", "🧠 Guía para defensa"])

# -------- TAB 1 --------
with tab1:
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])

    processed = int(st.session_state.ptr)
    total = int(len(X_scaled))
    counts_now = st.session_state.model.counts_.astype(int)
    full_clusters = int(np.sum(counts_now >= max_sizes))

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Instancias procesadas", f"{processed} / {total}")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Clusters llenos", f"{full_clusters} / {int(k)}")
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("Método")
        st.markdown(f'<span class="badge">{descriptor}</span>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("Restricción")
        st.markdown(f'<span class="badge">{restriction_mode_label}</span>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    if len(st.session_state.history) == 0:
        st.info("Presiona **Procesar 1 instancia** para ver el clustering online en vivo.")
    else:
        last = st.session_state.history[-1]
        if last["fallback_used"]:
            st.warning("⚠️ Todos los clusters estaban llenos. Revisa que la suma de cupos sea ≥ N.")
        else:
            st.success(
                f"✅ Última instancia (t={last['t']}) asignada a cluster **{last['assigned_cluster']}** | "
                f"orden de intento (cercano→lejano): {last['tried_order']}"
            )

    st.write("### Historial (últimas 30 instancias)")
    df_hist = pd.DataFrame(st.session_state.history)
    st.dataframe(df_hist.tail(30), use_container_width=True)

# -------- TAB 2 --------
with tab2:
    st.write("### Restricción de tamaño (cupo) vs tamaño actual")
    df_sizes = pd.DataFrame({
        "cluster": list(range(int(k))),
        "tamaño_actual": counts_now,
        "cupo_maximo": max_sizes.astype(int),
        "faltan_para_lleno": (max_sizes.astype(int) - counts_now).clip(min=0),
    })
    st.dataframe(df_sizes, use_container_width=True)

    st.write("### Vista rápida (barras)")
    chart_df = df_sizes.set_index("cluster")[["tamaño_actual", "cupo_maximo"]]
    st.bar_chart(chart_df, use_container_width=True)

    st.info(
        "📌 Regla del algoritmo: asigna la instancia al cluster más cercano **que tenga cupo**.\n\n"
        "Esto demuestra la **restricción de tamaño** de forma visual."
    )

# -------- TAB 3 --------
with tab3:
    if processed < int(k) + 2:
        st.info("Procesa más instancias para habilitar métricas y PCA (mínimo k+2).")
    else:
        processed_idx = st.session_state.stream_idx[:processed]
        X_proc = X_scaled[processed_idx]
        labels_proc = np.array(st.session_state.labels, dtype=int)

        left, right = st.columns([1, 1])

        with left:
            st.write("### Métricas internas / externas")
            if len(np.unique(labels_proc)) > 1:
                sil = silhouette_score(X_proc, labels_proc)  # determinístico (sin muestreo)
                st.metric("Silhouette (interno)", f"{sil:.4f}")
            else:
                st.warning("Silhouette requiere al menos 2 clusters distintos.")

            if y_true is not None:
                y_proc = y_true[processed_idx]
                ari = adjusted_rand_score(y_proc, labels_proc)
                nmi = normalized_mutual_info_score(y_proc, labels_proc)
                st.metric("ARI (externo)", f"{ari:.4f}")
                st.metric("NMI (externo)", f"{nmi:.4f}")
            else:
                st.info("No hay y_true en el CSV, no se calculan métricas externas.")

        with right:
            st.write("### PCA (2D) para visualizar agrupamiento")
            X2 = pca_2d_deterministic(X_proc)
            df_pca = pd.DataFrame({
                "PC1": X2[:, 0],
                "PC2": X2[:, 1],
                "cluster": labels_proc.astype(str),
            })
            st.scatter_chart(df_pca, x="PC1", y="PC2", color="cluster")

        st.info("📌 PCA aquí es solo visualización para tu defensa: muestra separación/solapamiento entre clusters.")

# -------- TAB 4 --------
with tab4:
    st.write("### Cómo defender el proyecto (en 60–90 segundos)")

    st.markdown(
        """
**1) Qué hace el sistema**
- Recibe datos **uno por uno** (online).
- Para cada instancia calcula su distancia a cada centroide.
- Asigna al cluster más cercano **que tenga cupo**.
- Actualiza el centroide **en ese instante** (sin recalcular todo).

**2) Dónde se ve lo “online”**
- En la pestaña **Online (Paso a Paso)**: botón *Procesar 1 instancia* + historial con t=0,1,2…

**3) Dónde se ve la restricción**
- En **Cupos y Clusters**: tabla y barras (tamaño_actual vs cupo_maximo)

**4) Método de extracción**
- Cambias Hu/HOG/SIFT desde el sidebar y el proceso es el mismo: solo cambian los vectores.

**5) Qué comparas**
- Con PCA y métricas (Silhouette, ARI, NMI) justificas qué descriptor separa mejor.
"""
    )

    st.success("✅ No se usa semilla ni random_state. El stream es determinístico por orden de datos.")

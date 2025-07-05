import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import locale

# =============================================================================
# Konfigurasi Halaman dan Judul
# =============================================================================
st.set_page_config(
    page_title="Dashboard Pembiayaan Puskesmas",
    layout="wide"
)

st.title("Dashboard Visualisasi Data Deskriptif Pembiayaan Puskesmas")

# =============================================================================
# Fungsi untuk Memuat dan Memproses Data
# =============================================================================
@st.cache_data
def load_data():
    df = pd.read_csv('dataset_puskesmas_cleaned_with_features.csv')
    return df

df_cleaned = load_data()

# =============================================================================
# Ringkasan Kinerja
# =============================================================================
st.header("Ringkasan Kinerja Pembiayaan")

total_alokasi = df_cleaned['alokasi'].sum()
total_realisasi = df_cleaned['realisasi'].sum()
avg_persentase_realisasi = (total_realisasi / total_alokasi) * 100
jumlah_puskesmas = df_cleaned['nama_puskesmas'].nunique()

def format_number(num):
    if num >= 1e12:
        return f"Rp {num / 1e12:.2f}T"
    if num >= 1e9:
        return f"Rp {num / 1e9:.2f}M"
    if num >= 1e6:
        return f"Rp {num / 1e6:.2f}Jt"
    return f"Rp {num:,.0f}"

col1, col2, col3, col4 = st.columns([2, 2, 1.5, 1.5])
col1.metric("Total Alokasi Dana", format_number(total_alokasi))
col2.metric("Total Realisasi Dana", format_number(total_realisasi))
col3.metric("Rata-rata Persentase Realisasi", f"{avg_persentase_realisasi:.2f}%")
col4.metric("Jumlah Puskesmas Terdata", jumlah_puskesmas)

# =============================================================================
# Visualisasi Utama (Sankey & Perbandingan Total)
# =============================================================================
st.header("Alur dan Perbandingan Anggaran")

col1, col2 = st.columns(2)

with col1:
    df_agg = df_cleaned.groupby(['sumber_dana_grouped', 'provinsi']).agg({
        'alokasi': 'sum', 'realisasi': 'sum'
    }).reset_index()
    df_agg['sisa_anggaran'] = df_agg['alokasi'] - df_agg['realisasi']

    sumber_dana_nodes = df_agg['sumber_dana_grouped'].unique().tolist()
    provinsi_nodes = df_agg['provinsi'].unique().tolist()
    status_nodes = ['Total Realisasi', 'Total Sisa Anggaran']
    all_labels = sumber_dana_nodes + provinsi_nodes + status_nodes
    label_indices = {label: i for i, label in enumerate(all_labels)}

    sources, targets, values, link_colors = [], [], [], []

    for _, row in df_agg.iterrows():
        sources.append(label_indices[row['sumber_dana_grouped']])
        targets.append(label_indices[row['provinsi']])
        values.append(row['alokasi'])
        link_colors.append('rgba(0, 119, 200, 0.6)')

    for _, row in df_agg.iterrows():
        if row['realisasi'] > 0:
            sources.append(label_indices[row['provinsi']])
            targets.append(label_indices['Total Realisasi'])
            values.append(row['realisasi'])
            link_colors.append('rgba(0, 182, 0, 0.6)')
        if row['sisa_anggaran'] > 0:
            sources.append(label_indices[row['provinsi']])
            targets.append(label_indices['Total Sisa Anggaran'])
            values.append(row['sisa_anggaran'])
            link_colors.append('rgba(255, 0, 0, 0.6)')

    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20,
            line=dict(color="black", width=0.5),
            label=all_labels,
            color=["#307EC7"] * len(sumber_dana_nodes) + ["#FFD700"] * len(provinsi_nodes) + ["#55a868", "#d62728"]
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors
        ))])
    fig_sankey.update_layout(title_text="Alur Dana dari Sumber ke Provinsi dan Status Akhir")
    st.plotly_chart(fig_sankey, use_container_width=True)

with col2:
    totals_df = pd.DataFrame({
        'Kategori': ['Total Alokasi', 'Total Realisasi'],
        'Jumlah': [total_alokasi, total_realisasi]
    })
    fig_bar_total = px.bar(totals_df, x='Kategori', y='Jumlah', title="Perbandingan Total Alokasi vs. Realisasi",
                           color='Kategori',
                           color_discrete_map={'Total Alokasi': '#4c72b0', 'Total Realisasi': '#55a868'})
    fig_bar_total.update_layout(yaxis_title="Jumlah Dana", xaxis_title="")
    st.plotly_chart(fig_bar_total, use_container_width=True)

# =============================================================================
# Peringkat Puskesmas
# =============================================================================
st.header("Peringkat Kinerja Puskesmas")

col1, col2 = st.columns(2)

with col1:
    top_10_df = df_cleaned.sort_values(by='persentase_realisasi', ascending=False).head(10)
    fig_top10 = px.bar(top_10_df, x='persentase_realisasi', y='nama_puskesmas', orientation='h',
                       title="10 Puskesmas dengan Realisasi Tertinggi",
                       color='persentase_realisasi', color_continuous_scale='viridis')
    fig_top10.update_layout(xaxis_title="Persentase Realisasi (%)", yaxis_title="Nama Puskesmas",
                            coloraxis_showscale=False, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_top10, use_container_width=True)

with col2:
    bottom_10_df = df_cleaned.sort_values(by='persentase_realisasi', ascending=True).head(10)
    fig_bottom10 = px.bar(bottom_10_df, x='persentase_realisasi', y='nama_puskesmas', orientation='h',
                          title="10 Puskesmas dengan Realisasi Terendah",
                          color='persentase_realisasi', color_continuous_scale='Reds_r')
    fig_bottom10.update_layout(xaxis_title="Persentase Realisasi (%)", yaxis_title="Nama Puskesmas",
                               coloraxis_showscale=False, yaxis={'categoryorder': 'total descending'})
    st.plotly_chart(fig_bottom10, use_container_width=True)

# =============================================================================
# Visualisasi Detail: Scatter & Histogram
# =============================================================================
st.header("Analisis Detail")

col1, col2 = st.columns(2)

with col1:
    fig_scatter, ax = plt.subplots(figsize=(14, 8), facecolor='none')
    fig_scatter.patch.set_alpha(0.0)
    ax.set_facecolor('none')

    sns.scatterplot(
        data=df_cleaned, x='alokasi', y='realisasi', hue='sumber_dana_grouped',
        s=100, alpha=0.6, edgecolor='black', linewidth=0.5, ax=ax)

    max_val = max(df_cleaned['alokasi'].max(), df_cleaned['realisasi'].max())
    ax.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='Realisasi 100%')

    ax.set_title('Hubungan antara Alokasi dan Realisasi Dana (Disederhanakan)', fontsize=16, pad=20, color='white')
    ax.set_xlabel('Alokasi Dana (Rp)', color='white')
    ax.set_ylabel('Realisasi Dana (Rp)', color='white')

    legend = ax.legend(title='Kelompok Sumber Dana', loc='upper left', facecolor='black', framealpha=0.5)
    legend.get_title().set_color('white')
    for text in legend.get_texts():
        text.set_color('white')

    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.ticklabel_format(style='plain', axis='both')
    ax.grid(True, linestyle='--', alpha=0.3)

    st.pyplot(fig_scatter)

with col2:
    fig_hist, ax = plt.subplots(figsize=(12, 7), facecolor='none')
    fig_hist.patch.set_alpha(0.0)
    ax.set_facecolor('none')

    sns.histplot(df_cleaned['persentase_realisasi'], bins=30, kde=True, color='coral', ax=ax)
    mean_value = df_cleaned['persentase_realisasi'].mean()
    ax.axvline(mean_value, color='red', linestyle='--', label=f'Rata-rata: {mean_value:.2f}%')

    ax.set_title('Distribusi Persentase Realisasi di Semua Puskesmas', fontsize=16, pad=20, color='white')
    ax.set_xlabel('Persentase Realisasi (%)', color='white')
    ax.set_ylabel('Jumlah Puskesmas', color='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')

    legend = ax.legend(facecolor='black', framealpha=0.5, edgecolor='white')
    for text in legend.get_texts():
        text.set_color('white')

    st.pyplot(fig_hist)

# =============================================================================
# Tabel Data dan Ekspor
# =============================================================================
st.header("Detail Data Pembiayaan")

columns_to_display = [
    'nama_puskesmas', 'provinsi', 'sumber_dana_grouped', 'performance_group',
    'alokasi', 'realisasi', 'sisa_anggaran', 'persentase_realisasi'
]
locale.setlocale(locale.LC_ALL, '')
df_display = df_cleaned[columns_to_display].copy()

def format_rupiah(x):
    return f"Rp {x:,.0f}".replace(",", ".")

df_display['Alokasi (Rp)'] = df_display['alokasi'].apply(format_rupiah)
df_display['Realisasi (Rp)'] = df_display['realisasi'].apply(format_rupiah)
df_display['Sisa Anggaran (Rp)'] = df_display['sisa_anggaran'].apply(format_rupiah)
df_display['Persentase Realisasi'] = df_display['persentase_realisasi'].apply(lambda x: f"{x:.2f}%")

df_display = df_display.rename(columns={
    'nama_puskesmas': 'Nama Puskesmas',
    'provinsi': 'Provinsi',
    'sumber_dana_grouped': 'Kelompok Sumber Dana',
    'performance_group': 'Kategori Kinerja'
})

final_columns = [
    'Nama Puskesmas', 'Provinsi', 'Kelompok Sumber Dana', 'Kategori Kinerja',
    'Alokasi (Rp)', 'Realisasi (Rp)', 'Sisa Anggaran (Rp)', 'Persentase Realisasi'
]

# Pastikan kolom tidak mengandung NaN dan semua dalam bentuk string
df_display['Provinsi'] = df_display['Provinsi'].fillna('Tidak diketahui').astype(str)
df_display['Kelompok Sumber Dana'] = df_display['Kelompok Sumber Dana'].fillna('Tidak diketahui').astype(str)

# =========================
# Fitur Filter
# =========================
prov_filter = st.multiselect(
    "Filter berdasarkan Provinsi", 
    options=sorted(df_display['Provinsi'].unique())
)

sumber_filter = st.multiselect(
    "Filter berdasarkan Sumber Dana", 
    options=sorted(df_display['Kelompok Sumber Dana'].unique())
)

# =========================
# Terapkan Filter
# =========================
filtered_df = df_display.copy()

if prov_filter:
    filtered_df = filtered_df[filtered_df['Provinsi'].isin(prov_filter)]

if sumber_filter:
    filtered_df = filtered_df[filtered_df['Kelompok Sumber Dana'].isin(sumber_filter)]

# =========================
# Tampilkan Tabel
# =========================
st.dataframe(filtered_df[final_columns], use_container_width=True)

# =========================
# Fitur Ekspor CSV
# =========================
csv = filtered_df[final_columns].to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Unduh Tabel dalam CSV",
    data=csv,
    file_name='data_pembiayaan_puskesmas.csv',
    mime='text/csv'
)


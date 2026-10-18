"""Estilo e funções de gráfico usados nos notebooks do projeto."""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

AZUL, LARANJA = "#2a78d6", "#eb6834"
TEXTO, TEXTO_SECUNDARIO, GRADE = "#0b0b0b", "#52514e", "#e4e3df"
FUNDO_ROTULO = dict(facecolor="white", edgecolor="none", pad=1)  # rótulo legível quando cruza a linha da média


def aplicar_estilo():
    plt.rcParams.update({
        "figure.dpi": 110, "font.size": 10, "figure.constrained_layout.use": True,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.edgecolor": GRADE, "axes.axisbelow": True,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRADE, "grid.linewidth": 0.8,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "text.color": TEXTO, "axes.labelcolor": TEXTO_SECUNDARIO,
        "xtick.color": TEXTO_SECUNDARIO, "ytick.color": TEXTO_SECUNDARIO, "ytick.left": False,
    })


def pct(v):
    return f"{v:.1%}".replace(".", ",")


def barras(ax, valores, titulo, formato=pct, media=None):
    """Uma barra por grupo, com o valor escrito no topo. `media` desenha a linha da taxa geral."""
    ax.bar([str(i) for i in valores.index], valores.values, width=0.4, color=AZUL)
    for i, v in enumerate(valores.values):
        ax.annotate(formato(v), (i, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", color=TEXTO, bbox=FUNDO_ROTULO)
    if media is not None:
        ax.axhline(media, color=TEXTO_SECUNDARIO, linewidth=1, label=f"média geral ({pct(media)})")
        ax.legend(frameon=False, loc="best", fontsize=9)
    if formato is pct:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, max(valores.max(), media or 0) * 1.18)
    ax.set_title(titulo)


def barras_agrupadas(ax, tabela, titulo, destacar=()):
    """Barras horizontais, duas por linha da tabela (uma por coluna). Valores escritos só nas linhas destacadas."""
    y = np.arange(len(tabela))
    altura = 0.36
    for j, (coluna, cor) in enumerate(zip(tabela.columns, [AZUL, LARANJA])):
        posicoes = y + (j - 0.5) * (altura + 0.02)
        ax.barh(posicoes, tabela[coluna], height=altura, color=cor, label=str(coluna))
        for p, (grupo, v) in zip(posicoes, tabela[coluna].items()):
            if grupo in destacar:
                ax.annotate(pct(v), (v, p), xytext=(3, 0), textcoords="offset points",
                            va="center", color=TEXTO, fontsize=9)
    ax.set_yticks(y, [str(i) for i in tabela.index])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, tabela.max().max() * 1.15)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="best", fontsize=9)
    ax.set_title(titulo)


def barras_horizontais(ax, valores, titulo, formato=pct, media=None, nome_media="média geral"):
    """Uma barra horizontal por grupo, na ordem recebida, com o valor na ponta."""
    ax.barh([str(i) for i in valores.index], valores.values, height=0.5, color=AZUL)
    for i, v in enumerate(valores.values):
        ax.annotate(formato(v), (v, i), xytext=(3, 0), textcoords="offset points",
                    va="center", color=TEXTO, fontsize=9, bbox=FUNDO_ROTULO)
    if media is not None:
        ax.axvline(media, color=TEXTO_SECUNDARIO, linewidth=1, label=f"{nome_media} ({formato(media)})")
        ax.legend(frameon=False, loc="lower right", fontsize=9)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    if formato is pct:
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, max(valores.max(), media or 0) * 1.2)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_title(titulo)

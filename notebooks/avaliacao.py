"""Métricas e o ponto de comparação (tabela segmento × antecedência) usados nos notebooks 03 e 04."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

FAIXAS_ANTECEDENCIA = [-1, 7, 30, 90, 180, 365, np.inf]


def taxa_no_topo(y, prob, fracao=0.10):
    """Entre as reservas com maior nota (as 10% mais arriscadas), a fração que cancelou.
    Reservas empatadas no corte entram pela média delas."""
    y, prob = np.asarray(y), np.asarray(prob)
    k = int(len(prob) * fracao)
    corte = np.sort(prob)[-k]
    acima, empate = prob > corte, prob == corte
    return (y[acima].sum() + (k - acima.sum()) * y[empate].mean()) / k


def metricas(y, prob):
    """ROC-AUC, PR-AUC e a fração que cancela entre as 10% reservas com maior nota."""
    return {
        "ROC-AUC": roc_auc_score(y, prob),
        "PR-AUC": average_precision_score(y, prob),
        "cancelam no top 10%": taxa_no_topo(y, prob),
    }


def nota_da_tabela(X_treino, y_treino, X):
    """Nota de cada reserva de X: a taxa de cancelamento do treino no mesmo segmento × faixa de antecedência."""

    def grupo(base):
        return base["market_segment"].astype(str) + " | " + pd.cut(base["lead_time"], FAIXAS_ANTECEDENCIA).astype(str)

    taxa_por_grupo = pd.Series(np.asarray(y_treino), index=X_treino.index).groupby(grupo(X_treino)).mean()
    return grupo(X).map(taxa_por_grupo).fillna(np.mean(y_treino)).to_numpy()

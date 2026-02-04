import numpy as np


class OnlineConstrainedKMeans:
    """
    K-Means ONLINE (instancia por instancia) con RESTRICCIÓN DE TAMAÑO.

    - Llega una instancia x
    - Calcula distancia a centroides
    - Elige el cluster más cercano que tenga cupo
    - Actualiza el centroide incrementalmente
    """

    def __init__(self, n_clusters: int, max_sizes):
        self.k = int(n_clusters)
        self.max_sizes = np.asarray(max_sizes, dtype=int)

        if len(self.max_sizes) != self.k:
            raise ValueError("max_sizes debe tener longitud igual a n_clusters")
        if np.any(self.max_sizes <= 0):
            raise ValueError("max_sizes debe tener valores positivos")

        self.centroids_ = None
        self.counts_ = np.zeros(self.k, dtype=int)

    def reset(self):
        self.centroids_ = None
        self.counts_ = np.zeros(self.k, dtype=int)

    def partial_fit_one(self, x: np.ndarray):
        """Procesa UNA instancia y retorna (cluster_asignado, info_debug)."""
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)[0]

        # Inicialización simple: al inicio, todos los centroides empiezan con la primera instancia
        if self.centroids_ is None:
            self.centroids_ = np.tile(x, (self.k, 1)).astype(np.float32)

        dists = np.linalg.norm(self.centroids_ - x, axis=1)
        order = np.argsort(dists)  # clusters de más cercano a más lejano

        chosen = None
        tried = []
        for c in order:
            c = int(c)
            tried.append(c)
            if self.counts_[c] < self.max_sizes[c]:
                chosen = c
                break

        # Si TODOS están llenos (no debería pasar si cupos >= N), igual asigna al más cercano
        fallback_used = False
        if chosen is None:
            chosen = int(order[0])
            fallback_used = True

        # Actualización online del centroide (media incremental)
        self.counts_[chosen] += 1
        eta = 1.0 / float(self.counts_[chosen])
        self.centroids_[chosen] = (1 - eta) * self.centroids_[chosen] + eta * x

        info = {
            "chosen": chosen,
            "tried_order": tried,
            "dists": dists.astype(float),
            "counts": self.counts_.copy(),
            "max_sizes": self.max_sizes.copy(),
            "fallback_used": fallback_used,
        }
        return chosen, info

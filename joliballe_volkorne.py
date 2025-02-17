import pandas as pd
from pyvis.network import Network
from scipy.optimize import linear_sum_assignment
from collections import deque
from typing import Dict, Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Volkorne:
    """
    Représente un Volkorne.
    """
    def __init__(
        self,
        vol_id: str,
        sex: str,
        partenaire: str = "NAN",
        father: Optional["Volkorne"] = None,
        mother: Optional["Volkorne"] = None
    ) -> None:
        self.id: str = vol_id
        self.sex: str = sex
        self.partenaire: str = partenaire
        self.father: Optional[Volkorne] = father
        self.mother: Optional[Volkorne] = mother

    def __repr__(self) -> str:
        f_id = self.father.id if self.father else None
        m_id = self.mother.id if self.mother else None
        return (
            f"Volkorne(id={self.id}, sex={self.sex}, "
            f"father={f_id}, mother={m_id}, partenaire={self.partenaire})"
        )


class Elevage:
    """
    Gère l'ensemble des Volkornes, leur chargement, leurs relations familiales,
    ainsi que le calcul de consanguinité et l'appariement optimal.
    """
    def __init__(self) -> None:
        self.volkornes_dict: Dict[str, Volkorne] = {}
        self.raw_parents_dict: Dict[str, str] = {}

    def load_volkornes_from_csv(self, df: pd.DataFrame) -> None:
        """
        Charge les Volkornes depuis un DataFrame, crée les objets et
        relie leurs parents dans un second temps.
        """
        df.columns = df.columns.str.upper()
        for _, row in df.iterrows():
            vol_id = str(row['NOM']).strip()
            sex = str(row['SEXE']).strip().upper()
            if sex == "NAN":
                continue

            partenaire = "NAN"
            if 'PARTENAIRE' in df.columns and not pd.isna(row['PARTENAIRE']):
                partenaire = str(row['PARTENAIRE'])

            raw_parents = ""
            if 'PARENTS' in df.columns and not pd.isna(row['PARENTS']):
                raw_parents = str(row['PARENTS']).strip()

            vol = Volkorne(vol_id=vol_id, sex=sex, partenaire=partenaire)
            self.volkornes_dict[vol_id] = vol
            self.raw_parents_dict[vol_id] = raw_parents

        for vol_id, raw_parents in self.raw_parents_dict.items():
            if "+" in raw_parents:
                parent_a_id, parent_b_id = raw_parents.split("+")
                parent_a_id = parent_a_id.strip()
                parent_b_id = parent_b_id.strip()
                parent_a = self.volkornes_dict.get(parent_a_id)
                parent_b = self.volkornes_dict.get(parent_b_id)
                if parent_a and parent_b:
                    self.volkornes_dict[vol_id].father = parent_a
                    self.volkornes_dict[vol_id].mother = parent_b
            else:
                self.volkornes_dict[vol_id].father = None
                self.volkornes_dict[vol_id].mother = None

    def get_volkorne(self, vol_id: str) -> Optional[Volkorne]:
        return self.volkornes_dict.get(vol_id)

    def all_volkornes(self) -> List[Volkorne]:
        return list(self.volkornes_dict.values())

    def repr_volkornes(self) -> None:
        for vol in self.volkornes_dict.values():
            print(vol)

    def display_family_tree(self, notebook: bool = False) -> None:
        """
        Affiche l'arbre généalogique complet (hiérarchique) via pyvis.
        """
        net = Network(height="750px", width="100%", directed=True, notebook=notebook)
        for vol_id, vol in self.volkornes_dict.items():
            label = f"{vol.id}\n({vol.sex})"
            title = f"ID: {vol.id}<br>Sexe: {vol.sex}<br>Partenaire: {vol.partenaire}"
            color = "lightblue" if vol.sex == "MALE" else "pink"
            net.add_node(n_id=vol_id, label=label, title=title, color=color)

        for vol_id, vol in self.volkornes_dict.items():
            if vol.father:
                net.add_edge(vol.father.id, vol.id)
            if vol.mother:
                net.add_edge(vol.mother.id, vol.id)

        net.set_options('''
        {
          "layout": {
            "hierarchical": {
              "enabled": true,
              "levelSeparation": 150,
              "nodeSpacing": 200,
              "direction": "UD",
              "sortMethod": "directed"
            }
          },
          "physics": {"enabled": false}
        }
        ''')
        net.show("family_tree.html", notebook=notebook)

    def _get_ancestors_ids(self, vol_id: str, visited: Optional[set] = None) -> set:
        if visited is None:
            visited = set()
        if vol_id not in self.volkornes_dict or vol_id in visited:
            return visited
        visited.add(vol_id)
        vol = self.volkornes_dict[vol_id]
        if vol.father:
            self._get_ancestors_ids(vol.father.id, visited)
        if vol.mother:
            self._get_ancestors_ids(vol.mother.id, visited)
        return visited

    def _get_ancestors_with_distance(self, vol: Volkorne) -> Dict[str, int]:
        """
        Renvoie {id_ancetre: distance_generations} via un parcours BFS.
        """
        ancestors: Dict[str, int] = {}
        queue = deque([(vol, 0)])
        while queue:
            current, dist = queue.popleft()
            if dist > 0:
                ancestors[current.id] = dist
            if current.father:
                queue.append((current.father, dist + 1))
            if current.mother:
                queue.append((current.mother, dist + 1))
        return ancestors

    def relatedness(self, v1: Optional[Volkorne], v2: Optional[Volkorne]) -> float:
        if v1 is None or v2 is None:
            return 0.0
        if v1.id == v2.id:
            return 1.0
        anc1 = self._get_ancestors_with_distance(v1)
        anc2 = self._get_ancestors_with_distance(v2)
        common_anc = set(anc1.keys()) & set(anc2.keys())
        r = 0.0
        for ca in common_anc:
            r += 1.0 / (2 ** (anc1[ca] + anc2[ca]))
        return r

    def get_candidates(self) -> Tuple[List[Volkorne], List[Volkorne]]:
        """
        Renvoie deux listes (males, females) : ceux qui n'ont pas encore de partenaire ("NAN").
        """
        # On ne sélectionne que ceux qui n'ont pas encore de partenaire
        candidates = [v for v in self.volkornes_dict.values() if v.partenaire == "NAN"]
        males = [v for v in candidates if v.sex == "MALE"]
        females = [v for v in candidates if v.sex == "FEMELLE"]
        return males, females

    def build_cost_matrix(
        self,
        males: List[Volkorne],
        females: List[Volkorne]
    ) -> np.ndarray:
        """
        Construit la matrice de consanguinité (mâle, femelle).
        """
        matrix: List[List[float]] = []
        for m in males:
            row: List[float] = []
            for f in females:
                row.append(self.relatedness(m, f))
            matrix.append(row)
        return np.array(matrix)

    def solve_min_cost_matching(self, cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applique l'algo Hongrois pour minimiser la somme dans cost_matrix.
        Retourne (row_ind, col_ind).
        """
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return row_ind, col_ind

    def interpret_matching(
        self,
        males: List[Volkorne],
        females: List[Volkorne],
        row_ind: np.ndarray,
        col_ind: np.ndarray,
        cost_matrix: np.ndarray
    ) -> List[Tuple[Volkorne, Volkorne, float]]:
        """
        Transforme la solution (row_ind, col_ind) en liste (mâle, femelle, consanguinité).
        """
        pairs: List[Tuple[Volkorne, Volkorne, float]] = []
        for i in range(len(row_ind)):
            m_index = row_ind[i]
            f_index = col_ind[i]
            male = males[m_index]
            fem = females[f_index]
            cons = cost_matrix[m_index, f_index]
            pairs.append((male, fem, float(cons)))
        pairs.sort(key=lambda x: x[2])
        return pairs

    def find_optimal_pairs(self) -> List[Tuple[Volkorne, Volkorne, float]]:
        """
        Détermine les appariements optimaux en minimisant la consanguinité.
        """
        males, females = self.get_candidates()
        if not males or not females:
            return []

        cost_matrix = self.build_cost_matrix(males, females)
        row_ind, col_ind = self.solve_min_cost_matching(cost_matrix)
        return self.interpret_matching(males, females, row_ind, col_ind, cost_matrix)


if __name__ == "__main__":
    elevage = Elevage()
    csv_path = "data/volkornes.csv"
    df = pd.read_csv(csv_path)
    elevage.load_volkornes_from_csv(df)

    elevage.repr_volkornes()

    # Exemple d'usage
    pairs = elevage.find_optimal_pairs()
    for male, fem, cons in pairs:
        print(f"Appariement: {male.id} x {fem.id} => consanguinité = {cons:.3f}")

    # Visualisation de la matrice
    males, females = elevage.get_candidates()
    matrix = elevage.build_cost_matrix(males, females)
    male_ids = [m.id for m in males]
    female_ids = [f.id for f in females]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        cmap="viridis",
        xticklabels=female_ids,
        yticklabels=male_ids
    )
    plt.xlabel("Femelles (ID)")
    plt.ylabel("Mâles (ID)")
    plt.title("Matrice de consanguinité")
    plt.show()

    # Arbre généalogique
    # elevage.display_family_tree(notebook=False)

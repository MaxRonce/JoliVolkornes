# How to use 

Make sure to have python 

## Install dependencies

```bash
pip install -r requirements.txt
```
## Download the volkornes data csv from your google drive

Paste it in data folder as ```volkornes.csv```

The .csv must follow the following format:
Columns : NOM	SEXE	POSSIB. GEN	PARTENAIRE	RESULTAT COUPLE	Nb_Repro	COULEUR	Parents

- NOM : Name of the volkorne, **MUST BE UNIQUE**
- SEX : MALE | FEMELLE
- Nb_Repro : Number of reproduction (0 if not reproduced, <=2 if reproduced)
- COULEUR : Color of the volkorne
- Parents : Name of the parents, separated by a "+", if the parents don't follow the format, the volkorne will be considered as a first generation volkorne

## Run the script

```bash
python joliballe_volkorne.py
```



# Week 3 — RESTful Inference Service

**Varolline AI Engineer internship · Week 3** — *Expose model inference
engine via FastAPI REST endpoints with response caching.*

A FastAPI service serving the Week-2 Titanic survival model (vendored
training code rebuilds the artifacts from the raw CSV, so this repo is
self-contained).

## Repository layout

```
week3-inference-api/
├── app/
│   ├── __init__.py
│   └── main.py                 # FastAPI app: /health, /predict, /predict/batch
├── src/preprocessing.py        # preprocessing logic (needed to unpickle the pipeline)
├── scripts/build_artifacts.py  # trains the model, writes artifacts/
├── artifacts/
│   ├── model.joblib            # winning end-to-end pipeline
│   ├── preprocessor.joblib     # fitted preprocessing pipeline
│   └── model_card.json
├── data/raw/titanic.csv        # vendored raw data
├── tests/test_api.py           # 5 TestClient tests
├── requirements.txt
├── LICENSE (MIT)
└── README.md
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_artifacts.py   # (re)build artifacts — already committed
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Run the tests:

```bash
pytest tests/ -v
```

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | `{"status": "ok", "model": "titanic-survival", "version": "1.0.0"}` |
| `POST` | `/predict` | Single passenger → `{"survived": bool, "probability": float, "cached": bool}` |
| `POST` | `/predict/batch` | Up to 1000 passengers → `{"predictions": [...], "cache_hits": n}` |

- **Response caching**: `cachetools.LRUCache` (1024 entries) keyed on the
  canonical JSON payload. Repeat requests return instantly with
  `"cached": true`.
- **`X-Process-Time`** header on every response.
- **Validation**: pydantic v2 — `Pclass ∈ {1,2,3}`, `Sex ∈ {male,female}`,
  `Embarked ∈ {C,Q,S}`; missing/invalid fields → `422` with a clear message.
- **Errors**: inference failures return `500 {"error": ...}` (no raw tracebacks).

## curl examples

```bash
# health
curl http://localhost:8000/health

# single prediction
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "Pclass": 1, "Name": "Example, Mrs. Rose", "Sex": "female", "Age": 29,
  "SibSp": 0, "Parch": 0, "Ticket": "PC 99999", "Fare": 100.0,
  "Cabin": "C85", "Embarked": "S"
}'
# -> {"survived": true, "probability": 0.96, "cached": false}

# batch prediction
curl -X POST http://localhost:8000/predict/batch -H "Content-Type: application/json" -d '[
  {"Pclass": 1, "Name": "A, Mrs. X", "Sex": "female", "Age": 29, "SibSp": 0,
   "Parch": 0, "Ticket": "T1", "Fare": 100.0, "Embarked": "S"},
  {"Pclass": 3, "Name": "B, Mr. Y", "Sex": "male", "Age": 22, "SibSp": 0,
   "Parch": 0, "Ticket": "T2", "Fare": 7.25, "Embarked": "S"}
]'
```

## Tests

`tests/test_api.py` (all passing): health check, single predict, batch
predict, validation error (422), and cache-hit behaviour (second identical
request returns `"cached": true` with identical results).

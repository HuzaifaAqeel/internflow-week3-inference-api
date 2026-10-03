"""API tests — run with ``pytest tests/`` (or ``python -m pytest``)."""

from fastapi.testclient import TestClient

from app.main import app

ROSE = {
    "Pclass": 1,
    "Name": "Test, Mrs. Rose Example",
    "Sex": "female",
    "Age": 29,
    "SibSp": 0,
    "Parch": 0,
    "Ticket": "PC 99999",
    "Fare": 100.0,
    "Cabin": "C85",
    "Embarked": "S",
}

JACK = {
    "Pclass": 3,
    "Name": "Test, Mr. Jack Example",
    "Sex": "male",
    "Age": 22,
    "SibSp": 0,
    "Parch": 0,
    "Ticket": "A/5 99999",
    "Fare": 7.25,
    "Cabin": None,
    "Embarked": "S",
}


def _client():
    return TestClient(app)


def test_health():
    with _client() as client:
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == "titanic-survival"


def test_predict_single():
    with _client() as client:
        r = client.post("/predict", json=ROSE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["survived"] is True  # 1st-class female: model should say survived
    assert 0.0 <= body["probability"] <= 1.0
    assert body["cached"] is False
    assert "X-Process-Time" in r.headers


def test_predict_batch():
    with _client() as client:
        r = client.post("/predict/batch", json=[ROSE, JACK])
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["predictions"]) == 2
    assert body["predictions"][1]["survived"] is False  # 3rd-class male
    assert all(0.0 <= p["probability"] <= 1.0 for p in body["predictions"])


def test_predict_validation_error():
    bad = dict(ROSE)
    del bad["Sex"]  # required field missing
    with _client() as client:
        r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_cache_hit():
    with _client() as client:
        first = client.post("/predict", json=JACK).json()
        second = client.post("/predict", json=JACK).json()
    assert first["cached"] is False
    assert second["cached"] is True
    assert first["survived"] == second["survived"]
    assert first["probability"] == second["probability"]

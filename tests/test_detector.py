import pytest
from finbert_ai_detector import FinbertAIDetector

@pytest.fixture(scope="module")
def detector():
    # Load once for all tests to save time
    # This might take a bit during the first run as it downloads weights
    return FinbertAIDetector()

def test_predict_single_text(detector):
    text = "The Tax Cuts and Jobs Act enacted in 2017 in the United States, significantly changed the tax rules applicable to U.S.-domiciled corporations."
    result = detector.predict(text)
    
    # Check the dictionary structure
    assert "text" in result
    assert "ai_probability" in result
    assert "human_probability" in result
    assert "label" in result
    
    assert result["text"] == text
    assert isinstance(result["ai_probability"], float)
    assert isinstance(result["human_probability"], float)
    assert result["label"] in ["AI-generated", "Human-written"]
    
    # Probabilities should sum to approximately 1.0
    total_prob = result["ai_probability"] + result["human_probability"]
    assert pytest.approx(total_prob, 0.01) == 1.0

def test_predict_batch(detector):
    texts = [
        "Company revenue grew by 15% due to increased demand in the European market.",
        "A machine learning model generated this text based on recent financial statements."
    ]
    
    results = detector.predict_batch(texts)
    
    assert isinstance(results, list)
    assert len(results) == 2
    
    for i, res in enumerate(results):
        assert res["text"] == texts[i]
        assert "ai_probability" in res
        assert "human_probability" in res
        assert "label" in res

def test_empty_batch(detector):
    assert detector.predict_batch([]) == []

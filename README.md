# FinBERT AI Detector

This is an easy-to-use Python package for the [msperlin/finbert-ai-detector](https://huggingface.co/msperlin/finbert-ai-detector) model.

The model is designed specifically to detect AI-generated text in financial documents, such as corporate annual reports (e.g., 10-K filings). It is fine-tuned from `yiyanghkust/finbert-pretrain`.

The model is used in the working paper:
> Perlin, Marcelo and Foguesatto, Cristian and Karagrigoriou Galanos, Aliki and Affonso, Felipe, The use of AI in 10-K Filings: An Empirical Analysis of S&P 500 Reports (January 21, 2026). Available at SSRN: [https://ssrn.com/abstract=6108946](https://ssrn.com/abstract=6108946) or [http://dx.doi.org/10.2139/ssrn.6108946](http://dx.doi.org/10.2139/ssrn.6108946)

## Features
- **GPU automatically supported:** CUDA and Apple Silicon (MPS) are fully supported if available.
- **Easy Interface:** Detect AI-generated text with a single method call.
- **Batched Inference:** Run predictions efficiently on huge datasets using batched inputs.

## Installation

Install using pip:

```bash
pip install finbert-ai-detector
```

## Quick Start
```python
from finbert_ai_detector import FinbertAIDetector

# Initialize the detector (downloads the model if not cached)
detector = FinbertAIDetector()

# Example text
text = "The Tax Cuts and Jobs Act enacted in 2017 in the United States, significantly changed the tax rules applicable to U.S.-domiciled corporations. Changes such as lower corporate tax rates, full expensing for qualified property, taxation of offshore earnings, limitations on interest expense deductions, and changes to the municipal bond tax exemption may impact demand for our products and services."

# Predict a single text
result = detector.predict(text)
print(f"Prediction: {result['label']}")
print(f"AI Probability: {result['ai_probability']:.2%}")
```

## Batched Prediction
For analyzing multiple documents or sentences quickly, use batched inference:

```python
from finbert_ai_detector import FinbertAIDetector

detector = FinbertAIDetector()

texts = [
    "Company revenue grew by 15% due to increased demand in the European market.",
    "A machine learning model generated this text based on recent financial statements."
]

results = detector.predict_batch(texts)
for result in results:
    print(f"Text: {result['text']}")
    print(f"AI Probability: {result['ai_probability']:.2%}")
    print("---")
```

## Intended Use & Limitations
- **Intended Usage:** Analyzing formal financial reports, press releases, corporate filings, and similar structured financial disclosures.
- **Limitations:** The model is optimized specifically for the formal, complex tone of financial documents. Its accuracy may be lower when applied to texts outside the financial domain, such as social media posts, casual emails, news articles, or creative text.
- **Length Constraint:** The underlying standard FinBERT architecture implies a maximum sequence length of 512 tokens. Texts longer than this will be truncated prior to sequence prediction.

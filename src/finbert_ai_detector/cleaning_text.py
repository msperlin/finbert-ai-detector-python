import re

def clean_financial_text(text):
    # 1. Standardize whitespace
    text = re.sub(r'\\s+', ' ', text)

    # 2. Remove typical PDF conversion artifacts (e.g., Page X of Y)
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)

    # 3. Keep common financial/punctuation non-ASCII, strip the rest
    # This regex keeps basic ASCII + common currency + accented Latin-1
    # Adjust if you're specifically looking for smart quotes as AI markers
    text = "".join(i for i in text if ord(i) < 128 or i in "€£¥©®™áéíóúçÁÉÍÓÚÇ")

    # 20260326 dont converto to lower case
    #text = text.lower()

    # 4. Strip leading/trailing whitespace
    text = text.strip()

    return text
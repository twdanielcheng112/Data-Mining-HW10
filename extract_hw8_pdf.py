from pypdf import PdfReader
import os

file_path = "HW8/scikitLearn 操作記錄單 2.pdf"
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    print(text)

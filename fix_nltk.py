"""
Quick script to download required NLTK data
Run this first if you encounter NLTK errors
"""

import nltk # type: ignore
import ssl

def download_nltk_data():
    """Download all required NLTK data"""
    
    # Handle SSL certificate issues
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    print("Downloading NLTK data...")
    
    # List of required packages
    packages = [
        'punkt',
        'punkt_tab',  
        'stopwords',
        'averaged_perceptron_tagger',
        'wordnet'
    ]
    
    for package in packages:
        try:
            print(f"Downloading {package}...")
            nltk.download(package, quiet=False)
            print(f"✓ {package} downloaded successfully")
        except Exception as e:
            print(f"✗ Failed to download {package}: {e}")
    
    
    print("\nTesting NLTK functionality...")
    try:
        from nltk.tokenize import word_tokenize, sent_tokenize # type: ignore
        from nltk.corpus import stopwords # type: ignore
        
        test_text = "This is a test sentence. This is another sentence!"
        words = word_tokenize(test_text)
        sentences = sent_tokenize(test_text)
        stops = stopwords.words('english')
        
        print(f"✓ Word tokenization: {len(words)} words")
        print(f"✓ Sentence tokenization: {len(sentences)} sentences") 
        print(f"✓ Stop words: {len(stops)} stop words loaded")
        print("\nNLTK is ready to use!")
        
    except Exception as e:
        print(f"✗ NLTK test failed: {e}")
        print("The main script will use fallback methods.")

if __name__ == "__main__":
    download_nltk_data()
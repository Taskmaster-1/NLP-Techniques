"""
Blackcoffer Data Extraction and Text Analysis Solution

This script performs:
1. Data extraction from URLs using BeautifulSoup
2. Text analysis including sentiment analysis and readability metrics
3. Output generation in specified format
"""

import pandas as pd # type: ignore
import requests # type: ignore
from bs4 import BeautifulSoup # type: ignore
import re
import string
import os
import sys
from urllib.parse import urlparse
import time
import warnings
warnings.filterwarnings('ignore')

try:
    import nltk # type: ignore
    from nltk.tokenize import sent_tokenize, word_tokenize # type: ignore
    from nltk.corpus import stopwords # type: ignore
    NLTK_AVAILABLE = True
    
    # Downloading required NLTK data
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)  
        nltk.download('stopwords', quiet=True)
        
        # Testing if NLTK actually works
        test_text = "This is a test sentence. This is another sentence."
        test_tokens = word_tokenize(test_text)
        test_sentences = sent_tokenize(test_text)
        print("NLTK is working properly.")
        
    except Exception as e:
        print(f"Warning: NLTK data download or test failed: {e}")
        print("Using built-in fallback methods.")
        NLTK_AVAILABLE = False
        
except ImportError:
    print("NLTK not available. Using built-in text processing methods.")
    NLTK_AVAILABLE = False

class TextAnalyzer:
    def __init__(self, stop_words_folder="StopWords", master_dict_folder="MasterDictionary"):
        """Initialize the TextAnalyzer with required dictionaries and stop words"""
        self.stop_words_folder = stop_words_folder
        self.master_dict_folder = master_dict_folder
        self.stop_words = set()
        self.positive_words = set()
        self.negative_words = set()
        
        self.load_stop_words()
        self.load_master_dictionary()
    
    def load_stop_words(self):
        """Load stop words from all files in StopWords folder"""
        try:
            if os.path.exists(self.stop_words_folder):
                for filename in os.listdir(self.stop_words_folder):
                    if filename.endswith('.txt'):
                        filepath = os.path.join(self.stop_words_folder, filename)
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            words = f.read().lower().split()
                            self.stop_words.update(words)
            
            
            if NLTK_AVAILABLE:
                try:
                    nltk_stop_words = set(stopwords.words('english'))
                    self.stop_words.update(nltk_stop_words)
                except:
                    pass
            else:
                basic_stop_words = {
                    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                    'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
                    'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                    'further', 'then', 'once'
                }
                self.stop_words.update(basic_stop_words)
                
            print(f"Loaded {len(self.stop_words)} stop words")
        except Exception as e:
            print(f"Error loading stop words: {e}")
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    def load_master_dictionary(self):
        """Load positive and negative words from MasterDictionary folder"""
        try:
            pos_file = os.path.join(self.master_dict_folder, "positive-words.txt")
            if os.path.exists(pos_file):
                with open(pos_file, 'r', encoding='utf-8', errors='ignore') as f:
                    pos_words = f.read().lower().split()
                    self.positive_words = set(word for word in pos_words if word not in self.stop_words)
            
            neg_file = os.path.join(self.master_dict_folder, "negative-words.txt")
            if os.path.exists(neg_file):
                with open(neg_file, 'r', encoding='utf-8', errors='ignore') as f:
                    neg_words = f.read().lower().split()
                    self.negative_words = set(word for word in neg_words if word not in self.stop_words)
            
            print(f"Loaded {len(self.positive_words)} positive words and {len(self.negative_words)} negative words")
        except Exception as e:
            print(f"Error loading master dictionary: {e}")
            self.positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'positive'}
            self.negative_words = {'bad', 'terrible', 'awful', 'horrible', 'negative', 'worst', 'hate'}

class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def extract_article_text(self, url, url_id):
        """Extract article title and text from URL"""
        try:
            print(f"Extracting text from: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Removing unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "sidebar", "advertisement"]):
                element.decompose()
            
            # Extracting titles
            title = ""
            title_tags = soup.find_all(['h1', 'title'])
            if title_tags:
                title = title_tags[0].get_text().strip()
            
            # Extracting article text
            article_text = ""

            # Different selectors for article content
            selectors = [
                'article',
                '.post-content',
                '.entry-content', 
                '.article-content',
                '.content',
                'main',
                '.post',
                '.article'
            ]
            
            content_found = False
            for selector in selectors:
                content_elements = soup.select(selector)
                if content_elements:
                    article_text = content_elements[0].get_text()
                    content_found = True
                    break
            
            # If no specific content area found, extract from body
            if not content_found:
                body = soup.find('body')
                if body:
                    article_text = body.get_text()
            
            # Cleaning
            article_text = re.sub(r'\s+', ' ', article_text).strip()
            
            # Combine title and article text
            full_text = f"{title}\n\n{article_text}"
            
            
            filename = f"{url_id}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            print(f"Successfully extracted and saved: {filename}")
            return full_text
            
        except Exception as e:
            print(f"Error extracting from {url}: {e}")
            return ""
    
    def extract_from_input_file(self, input_file="Input.xlsx"):
        """Extract articles from all URLs in input file"""
        try:
            df = pd.read_excel(input_file)
            extracted_texts = {}
            
            for index, row in df.iterrows():
                url_id = row['URL_ID']
                url = row['URL']
                
                text = self.extract_article_text(url, url_id)
                extracted_texts[url_id] = text
                
                time.sleep(1)
            
            return extracted_texts
        except Exception as e:
            print(f"Error processing input file: {e}")
            return {}

class TextAnalysisProcessor:
    def __init__(self, analyzer):
        self.analyzer = analyzer
    
    def simple_sent_tokenize(self, text):
        """Simple sentence tokenizer as fallback"""
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def simple_word_tokenize(self, text):
        """Simple word tokenizer as fallback"""
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        return [word for word in words if word]
    
    def clean_text(self, text):
        """Clean text by removing stop words and punctuation"""
        if NLTK_AVAILABLE:
            try:
                words = word_tokenize(text.lower())
            except Exception as e:
                print(f"NLTK tokenization failed: {e}. Using fallback method.")
                words = self.simple_word_tokenize(text.lower())
        else:
            words = self.simple_word_tokenize(text.lower())
        cleaned_words = [
            word for word in words 
            if word not in string.punctuation and 
            word not in self.analyzer.stop_words and 
            word.isalpha()
        ]
        
        return cleaned_words
    
    def count_syllables(self, word):
        """Count syllables in a word"""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        prev_char_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_char_was_vowel:
                    syllable_count += 1
                prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False
        
        if word.endswith(('es', 'ed')):
            syllable_count -= 1
        return max(1, syllable_count)
    
    def is_complex_word(self, word):
        """Check if word is complex (more than 2 syllables)"""
        return self.count_syllables(word) > 2
    
    def count_personal_pronouns(self, text):
        """Count personal pronouns, excluding 'US' as country"""
        pronouns = r'\b(I|we|my|ours|us)\b'
        matches = re.findall(pronouns, text, re.IGNORECASE)
        
        filtered_matches = []
        for match in matches:
            if match.lower() == 'us':
                context = re.search(r'\b(?:in|from|to)\s+us\b', text, re.IGNORECASE)
                if not context:
                    filtered_matches.append(match)
            else:
                filtered_matches.append(match)
        
        return len(filtered_matches)
    
    def analyze_text(self, text):
        """Perform complete text analysis"""
        if not text or len(text.strip()) == 0:
            return self.get_default_scores()
        
        cleaned_words = self.clean_text(text)
        
        if NLTK_AVAILABLE:
            try:
                all_words = word_tokenize(text.lower())
                sentences = sent_tokenize(text)
            except Exception as e:
                print(f"NLTK tokenization failed: {e}. Using fallback method.")
                all_words = self.simple_word_tokenize(text.lower())
                sentences = self.simple_sent_tokenize(text)
        else:
            all_words = self.simple_word_tokenize(text.lower())
            sentences = self.simple_sent_tokenize(text)
        
        # Calculating sentiment scores
        positive_score = sum(1 for word in cleaned_words if word in self.analyzer.positive_words)
        negative_score = sum(1 for word in cleaned_words if word in self.analyzer.negative_words)
        
        # Calculating polarity score
        total_sentiment = positive_score + negative_score
        if total_sentiment > 0:
            polarity_score = (positive_score - negative_score) / (total_sentiment + 0.000001)
        else:
            polarity_score = 0
        
        # Calculating subjectivity score
        total_words_cleaned = len(cleaned_words)
        if total_words_cleaned > 0:
            subjectivity_score = total_sentiment / (total_words_cleaned + 0.000001)
        else:
            subjectivity_score = 0
        
        # Calculating readability metrics
        word_count = len(cleaned_words)
        sentence_count = len(sentences)
        
        if sentence_count > 0:
            avg_sentence_length = word_count / sentence_count
            avg_words_per_sentence = avg_sentence_length
        else:
            avg_sentence_length = 0
            avg_words_per_sentence = 0
        
        # Complex word analysis
        complex_words = [word for word in cleaned_words if self.is_complex_word(word)]
        complex_word_count = len(complex_words)
        
        if word_count > 0:
            percentage_complex_words = complex_word_count / word_count
        else:
            percentage_complex_words = 0
    
        fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
        
        # Syllable analysis
        total_syllables = sum(self.count_syllables(word) for word in cleaned_words)
        if word_count > 0:
            syllables_per_word = total_syllables / word_count
        else:
            syllables_per_word = 0
        
        # Personal pronouns
        personal_pronouns = self.count_personal_pronouns(text)
        
        # Average word length
        total_characters = sum(len(word) for word in cleaned_words)
        if word_count > 0:
            avg_word_length = total_characters / word_count
        else:
            avg_word_length = 0
        
        return {
            'POSITIVE SCORE': positive_score,
            'NEGATIVE SCORE': negative_score,
            'POLARITY SCORE': polarity_score,
            'SUBJECTIVITY SCORE': subjectivity_score,
            'AVG SENTENCE LENGTH': avg_sentence_length,
            'PERCENTAGE OF COMPLEX WORDS': percentage_complex_words,
            'FOG INDEX': fog_index,
            'AVG NUMBER OF WORDS PER SENTENCE': avg_words_per_sentence,
            'COMPLEX WORD COUNT': complex_word_count,
            'WORD COUNT': word_count,
            'SYLLABLE PER WORD': syllables_per_word,
            'PERSONAL PRONOUNS': personal_pronouns,
            'AVG WORD LENGTH': avg_word_length
        }
    
    def get_default_scores(self):
        """Return default scores for empty/invalid text"""
        return {
            'POSITIVE SCORE': 0,
            'NEGATIVE SCORE': 0,
            'POLARITY SCORE': 0,
            'SUBJECTIVITY SCORE': 0,
            'AVG SENTENCE LENGTH': 0,
            'PERCENTAGE OF COMPLEX WORDS': 0,
            'FOG INDEX': 0,
            'AVG NUMBER OF WORDS PER SENTENCE': 0,
            'COMPLEX WORD COUNT': 0,
            'WORD COUNT': 0,
            'SYLLABLE PER WORD': 0,
            'PERSONAL PRONOUNS': 0,
            'AVG WORD LENGTH': 0
        }

def main():
    """Main execution function"""
    print("Starting Blackcoffer Text Analysis...")
    
    analyzer = TextAnalyzer()
    scraper = WebScraper()
    processor = TextAnalysisProcessor(analyzer)
    
    # Step 1: Extracting articles from URLs
    print("\n=== STEP 1: DATA EXTRACTION ===")
    extracted_texts = scraper.extract_from_input_file("Input.xlsx")
    
    if not extracted_texts:
        print("No texts were extracted. Please check your Input.xlsx file and internet connection.")
        return
    
    # Step 2: Analyzing extracted texts
    print("\n=== STEP 2: TEXT ANALYSIS ===")
    results = []
    
    # Loading input data for URL_ID and URL information
    try:
        input_df = pd.read_excel("Input.xlsx")
    except:
        print("Error: Could not load Input.xlsx")
        return
    
    for index, row in input_df.iterrows():
        url_id = row['URL_ID']
        url = row['URL']
        
        print(f"Analyzing article: {url_id}")
        
        text = extracted_texts.get(url_id, "")
        analysis_results = processor.analyze_text(text)
        
        result_row = {
            'URL_ID': url_id,
            'URL': url,
            **analysis_results
        }
        results.append(result_row)
    
    print("\n=== STEP 3: SAVING RESULTS ===")
    output_df = pd.DataFrame(results)
    
    # Save as both CSV and Excel
    output_df.to_csv("Output_Results.csv", index=False)
    output_df.to_excel("Output_Results.xlsx", index=False)
    
    print("Analysis complete!")
    print(f"Results saved to Output_Results.csv and Output_Results.xlsx")
    print(f"Processed {len(results)} articles")
    

    print("\n=== SUMMARY STATISTICS ===")
    numeric_columns = [col for col in output_df.columns if col not in ['URL_ID', 'URL']]
    print(output_df[numeric_columns].describe())

if __name__ == "__main__":
    main()
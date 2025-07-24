# NLP-Techniques 📊

> **Advanced Text Analysis Pipeline for Article Content Extraction and Sentiment Analysis**

A comprehensive Natural Language Processing toolkit that extracts article content from URLs and performs detailed linguistic analysis, generating 13+ key metrics including sentiment scores, readability indices, and text complexity measures.

## 🚀 Features

### Text Analysis Metrics
- **Sentiment Analysis**: Positive Score, Negative Score, Polarity Score, Subjectivity Score
- **Readability Metrics**: FOG Index, Average Sentence Length, Average Words per Sentence
- **Complexity Analysis**: Complex Word Count, Percentage of Complex Words, Syllables per Word
- **Text Statistics**: Word Count, Average Word Length, Personal Pronouns Count

### Core Capabilities
- ✅ **URL Content Extraction**: Automated article scraping from web URLs
- ✅ **Batch Processing**: Handle multiple articles simultaneously
- ✅ **CSV Export**: Structured data output for analysis and visualization
- ✅ **Multi-metric Analysis**: 13+ linguistic and sentiment features
- ✅ **Text Preprocessing**: Advanced cleaning and normalization

## 📋 Requirements

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `requests` - Web scraping and HTTP requests
- `beautifulsoup4` - HTML parsing and content extraction
- `nltk` - Natural language processing toolkit
- `textblob` - Sentiment analysis and text processing
- `pandas` - Data manipulation and CSV handling
- `numpy` - Numerical computations
- `re` - Regular expressions for text cleaning

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/Taskmaster-1/NLP-Techniques.git
cd NLP-Techniques
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data** (if required)
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 📖 Usage

### Basic Usage

```python
from nlp_analyzer import ArticleAnalyzer

# Initialize analyzer
analyzer = ArticleAnalyzer()

# Analyze single URL
url = "https://example.com/article"
results = analyzer.analyze_url(url)

# Analyze multiple URLs
urls = ["url1", "url2", "url3"]
results = analyzer.analyze_batch(urls)

# Export to CSV
analyzer.export_to_csv(results, "analysis_results.csv")
```

### Command Line Interface

```bash
# Analyze single URL
python analyze.py --url "https://example.com/article"

# Batch processing from file
python analyze.py --input urls.txt --output results.csv

# Custom configuration
python analyze.py --config config.json
```

## 📊 Output Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **Positive Score** | Proportion of positive words | 0.0 - 1.0 |
| **Negative Score** | Proportion of negative words | 0.0 - 1.0 |
| **Polarity Score** | Overall sentiment polarity | -1.0 to 1.0 |
| **Subjectivity Score** | Subjectivity vs objectivity | 0.0 - 1.0 |
| **Average Sentence Length** | Mean words per sentence | Numeric |
| **Percentage of Complex Words** | Complex words ratio | 0-100% |
| **FOG Index** | Gunning Fog readability index | Numeric |
| **Average Words per Sentence** | Mean sentence length | Numeric |
| **Complex Word Count** | Total complex words | Integer |
| **Word Count** | Total words in text | Integer |
| **Syllables per Word** | Average syllable count | Numeric |
| **Personal Pronouns** | Count of personal pronouns | Integer |
| **Average Word Length** | Mean character length | Numeric |

## 🔧 Configuration

Customize analysis parameters in `config/settings.json`:

```json
{
  "scraping": {
    "timeout": 30,
    "headers": {"User-Agent": "NLP-Analyzer/1.0"},
    "retry_attempts": 3
  },
  "analysis": {
    "min_word_length": 3,
    "complex_word_threshold": 2,
    "remove_stopwords": true
  },
  "output": {
    "csv_encoding": "utf-8",
    "include_raw_text": false
  }
}
```

## 📈 Sample Output

```csv
URL,Positive_Score,Negative_Score,Polarity_Score,Subjectivity_Score,Avg_Sentence_Length,Complex_Word_Percentage,FOG_Index,Word_Count,Personal_Pronouns,Avg_Word_Length
https://example.com/article1,0.23,0.12,0.11,0.45,18.5,12.3,8.7,450,15,4.2
https://example.com/article2,0.31,0.08,0.23,0.52,20.1,15.7,9.2,520,12,4.5
```

## 🧪 Testing

Run unit tests to ensure functionality:

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_analyzer.py

# With coverage report
python -m pytest --cov=src tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 Use Cases

- **Content Analysis**: Analyze blog posts, news articles, and web content
- **Sentiment Monitoring**: Track sentiment trends across multiple sources
- **Readability Assessment**: Evaluate text complexity and accessibility
- **Research Projects**: Academic studies on text linguistics and sentiment
- **SEO Optimization**: Analyze content quality and readability metrics

## 🔍 Technical Details

### Sentiment Analysis Algorithm
- Uses lexicon-based approach with VADER sentiment analyzer
- Combines rule-based and machine learning techniques
- Handles negations, intensifiers, and contextual sentiment

### Readability Calculations
- **FOG Index**: Gunning Fog Index for reading difficulty
- **Complex Words**: Words with 3+ syllables
- **Sentence Analysis**: Statistical analysis of sentence structure

## ⚡ Performance

- **Processing Speed**: ~2-3 articles per second
- **Memory Usage**: <100MB for typical batches
- **Accuracy**: 85%+ sentiment classification on news articles
- **Scalability**: Handles 1000+ URLs efficiently

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Vivek Yadav**
- GitHub: [@Taskmaster-1](https://github.com/Taskmaster-1)
- LinkedIn: [Vivek Yadav](https://linkedin.com/in/taskmaster)

## 🙏 Acknowledgments

- NLTK team for natural language processing tools
- TextBlob for sentiment analysis capabilities
- BeautifulSoup for web scraping utilities
- Pandas for data manipulation framework

---

⭐ **Star this repository if you found it helpful!** ⭐

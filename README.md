# 🛒 Sales Expert RAG System with NVIDIA AI

![GitHub](https://img.shields.io/github/license/yourusername/sales-expert-rag)
![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![NVIDIA](https://img.shields.io/badge/NVIDIA%20AI-Powered-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-red)

A state-of-the-art Sales Analysis RAG (Retrieval-Augmented Generation) system powered by NVIDIA's AI models. This application allows users to analyze sales data from PDF documents using natural language queries, providing intelligent insights through advanced LLM and embedding models.

## 🎥 Demo

[![Sales Expert RAG Demo](https://img.shields.io/badge/Watch-Demo%20Video-blue)](your_demo_video_link)

[Watch the full demo video here](your_demo_video_link)

## ✨ Features

- 📊 Interactive Sales Data Analysis
- 🤖 Powered by NVIDIA's Advanced LLM Models
- 📑 PDF Document Processing & Indexing
- 💡 Natural Language Query Interface
- 📈 Real-time Performance Monitoring
- 🔄 Response Quality Evaluation
- 📊 User Feedback System
- 🎯 Detailed Analytics Dashboard

## 🛠️ Technology Stack

- **NVIDIA AI Models**: NeMo and NV-Embed
- **Python**: Core programming language
- **Streamlit**: Web interface
- **LlamaIndex**: Document indexing and querying
- **SQLite**: Metrics storage
- **PyMuPDF**: PDF processing

## 🚀 Quick Start

### Prerequisites

1. Python 3.9+
2. NVIDIA AI API Key ([Get it here](#getting-nvidia-api-key))
3. Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sales-expert-rag.git
cd sales-expert-rag
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file:
```bash
NVIDIA_API_KEY=your_nvidia_api_key_here
```

5. Run the application:
```bash
streamlit run app.py
```

## 🔑 Getting NVIDIA API Key

1. Visit [NVIDIA AI Foundation Models](https://www.nvidia.com/en-us/ai-data-science/foundation-models/)
2. Click on "Get Started"
3. Sign up or log in to your NVIDIA account
4. Navigate to the API Keys section
5. Generate a new API key
6. Copy the key and add it to your `.env` file

For detailed instructions, visit [NVIDIA's API Documentation](https://docs.nvidia.com/ai-foundation-models/index.html)

## 📖 Usage Guide

1. **Upload Documents**
   - Click on "Upload your Sales PDFs"
   - Select one or more PDF files containing sales data
   - Wait for processing

2. **Ask Questions**
   - Type your question in natural language
   - Example queries:
     - "What were the top-selling products last quarter?"
     - "Show me the sales trend for the past 6 months"
     - "Compare regional performance"

3. **Analyze Results**
   - View the detailed response
   - Check response quality metrics
   - Provide feedback
   - Download analysis results

4. **Monitor Performance**
   - View system metrics
   - Check response times
   - Monitor context quality
   - Track user satisfaction

## 📊 Monitoring & Evaluation

The system includes comprehensive monitoring features:

- Response time tracking
- Context quality assessment
- Answer relevancy scoring
- User feedback collection
- Detailed performance metrics

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NVIDIA AI for providing the foundation models
- LlamaIndex  for the indexing framework
- Streamlit  for the amazing web framework
- Thanks to all.
---

Made with ❤️ by Faisal

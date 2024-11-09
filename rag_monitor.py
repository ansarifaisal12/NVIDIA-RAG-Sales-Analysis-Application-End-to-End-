import pandas as pd
from datetime import datetime
import numpy as np
from typing import Dict, List, Optional
import json
import os
from dataclasses import dataclass, asdict
import sqlite3
from pathlib import Path

@dataclass
class RAGMetrics:
    """Data class to store RAG evaluation metrics"""
    query_id: str
    timestamp: str
    query: str
    response: str
    response_time: float
    token_count: int
    context_quality: float
    answer_relevancy: float
    user_feedback: Optional[int] = None
    source_documents: Optional[List[str]] = None

class RAGMonitor:
    def __init__(self, db_path: str = "rag_metrics.db"):
        """Initialize the RAG monitoring system"""
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        """Create database and tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rag_metrics (
                query_id TEXT PRIMARY KEY,
                timestamp TEXT,
                query TEXT,
                response TEXT,
                response_time REAL,
                token_count INTEGER,
                context_quality REAL,
                answer_relevancy REAL,
                user_feedback INTEGER,
                source_documents TEXT
            )
        ''')
        
        conn.commit()
        conn.close()

    def evaluate_context_quality(self, query: str, context_chunks: List[str]) -> float:
        """Evaluate the quality of retrieved context"""
        # Implement semantic similarity between query and context chunks
        # This is a simplified version - you might want to use your NVIDIA embeddings here
        try:
            # Basic length-based scoring for demonstration
            total_score = sum(len(set(query.lower().split()) & set(chunk.lower().split())) 
                            for chunk in context_chunks)
            return min(total_score / len(query.split()), 1.0)
        except Exception:
            return 0.0

    def evaluate_answer_relevancy(self, query: str, response: str) -> float:
        """Evaluate the relevancy of the answer to the query"""
        try:
            # Basic word overlap scoring - replace with more sophisticated metrics
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            overlap = len(query_words & response_words)
            return min(overlap / len(query_words), 1.0)
        except Exception:
            return 0.0

    def log_interaction(self, metrics: RAGMetrics):
        """Log RAG interaction metrics to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert source_documents list to JSON string
        metrics_dict = asdict(metrics)
        if metrics_dict['source_documents']:
            metrics_dict['source_documents'] = json.dumps(metrics_dict['source_documents'])
        
        cursor.execute('''
            INSERT OR REPLACE INTO rag_metrics
            (query_id, timestamp, query, response, response_time, token_count,
             context_quality, answer_relevancy, user_feedback, source_documents)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', tuple(metrics_dict.values()))
        
        conn.commit()
        conn.close()

    def get_metrics_summary(self) -> Dict:
        """Get summary statistics of RAG performance"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM rag_metrics", conn)
        conn.close()

        if len(df) == 0:
            return {}

        return {
            "total_queries": len(df),
            "avg_response_time": df["response_time"].mean(),
            "avg_context_quality": df["context_quality"].mean(),
            "avg_answer_relevancy": df["answer_relevancy"].mean(),
            "avg_user_feedback": df["user_feedback"].mean(),
            "queries_last_24h": len(df[pd.to_datetime(df["timestamp"]) > 
                                    pd.Timestamp.now() - pd.Timedelta(days=1)])
        }

    def generate_report(self) -> pd.DataFrame:
        """Generate a detailed report of RAG performance"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM rag_metrics", conn)
        conn.close()
        
        return df

class ResponseEvaluator:
    def __init__(self):
        """Initialize the response evaluator"""
        self.criteria = {
            "completeness": self._evaluate_completeness,
            "coherence": self._evaluate_coherence,
            "relevance": self._evaluate_relevance
        }

    def _evaluate_completeness(self, response: str) -> float:
        """Evaluate if the response is complete"""
        # Basic implementation - you might want to enhance this
        has_conclusion = any(word in response.lower() 
                           for word in ["therefore", "conclusion", "finally", "in summary"])
        sentence_count = len(response.split('.'))
        return min((sentence_count / 3) * 0.7 + (0.3 if has_conclusion else 0), 1.0)

    def _evaluate_coherence(self, response: str) -> float:
        """Evaluate the coherence of the response"""
        sentences = response.split('.')
        if len(sentences) < 2:
            return 0.5
        
        # Check for transition words
        transition_words = {"however", "moreover", "furthermore", "additionally", "thus"}
        transition_count = sum(1 for word in transition_words 
                             if word.lower() in response.lower())
        
        return min(0.4 + (transition_count * 0.2), 1.0)

    def _evaluate_relevance(self, query: str, response: str) -> float:
        """Evaluate the relevance of the response to the query"""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words & response_words)
        return min(overlap / len(query_words), 1.0)

    def evaluate_response(self, query: str, response: str) -> Dict[str, float]:
        """Evaluate the response based on multiple criteria"""
        scores = {}
        for criterion, eval_func in self.criteria.items():
            if criterion == "relevance":
                scores[criterion] = eval_func(query, response)
            else:
                scores[criterion] = eval_func(response)
        
        # Calculate overall score
        scores["overall"] = np.mean(list(scores.values()))
        return scores
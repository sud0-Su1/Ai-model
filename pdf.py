import os
import re
import math
import numpy as np
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading

# Check for required packages and install if missing
try:
    import PyPDF2
except ModuleNotFoundError:
    import sys
    import subprocess
    print("PyPDF2 module not found. Installing PyPDF2...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
    import PyPDF2
    print("PyPDF2 installed successfully!")

try:
    import tkinter as tk
    from tkinter import filedialog, ttk, scrolledtext
except ModuleNotFoundError:
    print("Note: Tkinter is required for the GUI version.")
    print("If running in a notebook environment, we'll use a simplified interface.")
    HAS_TKINTER = False
else:
    HAS_TKINTER = True


class PDFQuestionAnsweringSystem:
    def __init__(self):
        """Initialize the PDF QA system with improved NLP capabilities"""
        self.documents = {}
        self.document_index = {}
        self.idf_scores = {}
        self.vocabulary = set()
        self.chunk_size = 1000  # Reduced for better precision
        self.overlap = 200      # Optimized overlap
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
            'be', 'been', 'being', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 
            'about', 'of', 'this', 'that', 'these', 'those', 'it', 'its', 'as',
            'from', 'have', 'has', 'had', 'not', 'what', 'when', 'where', 'who',
            'which', 'how', 'why', 'would', 'could', 'should', 'will', 'shall'
        }  # Expanded stop words list
        
        # Create cache for document queries to speed up repeated questions
        self.query_cache = {}

    def preprocess_text(self, text):
        """Optimized text preprocessing"""
        # Convert to lowercase and remove special characters in one pass
        text = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
        
        # Tokenize efficiently
        tokens = [token for token in text.split() 
                 if token not in self.stop_words and len(token) > 1]
        
        return tokens

    def create_chunks(self, text):
        """Create overlapping chunks from text with improved handling"""
        # Handle empty text
        if not text:
            return []
            
        chunks = []
        # Ensure chunks break at sentence boundaries where possible
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += sentence + " "
            else:
                # Add current chunk if it's not empty
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with current sentence
                current_chunk = sentence + " "
                
                # If sentence is longer than chunk_size, split it
                if len(sentence) > self.chunk_size:
                    # Split long sentences into overlapping chunks
                    words = sentence.split()
                    for i in range(0, len(words), self.chunk_size // 10):
                        chunk = " ".join(words[i:i + self.chunk_size // 5])
                        if chunk:
                            chunks.append(chunk)
                    current_chunk = ""
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def load_pdf(self, pdf_path, callback=None):
        """Load and process a PDF file with threading support"""
        try:
            pdf_name = os.path.basename(pdf_path)
            
            start_time = time.time()
            
            # Extract text from PDF - optimized extraction
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Parallel processing for large PDFs
                def extract_page_text(page_num):
                    try:
                        page = pdf_reader.pages[page_num]
                        return page.extract_text()
                    except Exception as e:
                        return f"[Error on page {page_num}: {str(e)}]"
                
                # Use ThreadPoolExecutor for parallel page extraction
                with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 8)) as executor:
                    page_texts = list(executor.map(extract_page_text, 
                                               range(len(pdf_reader.pages))))
                
                full_text = " ".join(page_texts)
            
            # Store the full text
            self.documents[pdf_name] = full_text
            
            # Create chunks
            chunks = self.create_chunks(full_text)
            
            if callback:
                callback(f"Processing {len(chunks)} chunks from {pdf_name}...")
            
            # Process each chunk with parallel processing
            doc_data = []
            
            def process_chunk(chunk_data):
                idx, chunk = chunk_data
                tokens = self.preprocess_text(chunk)
                term_freq = Counter(tokens)
                
                return {
                    'id': f"{pdf_name}_chunk_{idx}",
                    'text': chunk,
                    'tokens': tokens,
                    'term_freq': term_freq
                }
            
            with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 8)) as executor:
                doc_data = list(executor.map(process_chunk, 
                                          enumerate(chunks)))
            
            # Update vocabulary in one pass
            for chunk_data in doc_data:
                self.vocabulary.update(chunk_data['tokens'])
            
            # Store in document index
            self.document_index[pdf_name] = doc_data
            
            # Update IDF scores
            self._update_idf_scores()
            
            processing_time = time.time() - start_time
            
            if callback:
                callback(f"Successfully processed {pdf_name} in {processing_time:.2f} seconds")
            
            return True, f"Processed {pdf_name} in {processing_time:.2f} seconds"

        except Exception as e:
            error_msg = f"Error loading PDF: {str(e)}"
            if callback:
                callback(error_msg)
            return False, error_msg

    def _update_idf_scores(self):
        """Update the IDF scores more efficiently"""
        # Count documents containing each term
        doc_freq = defaultdict(int)
        total_docs = 0
        
        # First collect all unique terms by document
        for doc_name, chunks in self.document_index.items():
            total_docs += len(chunks)
            
            for chunk in chunks:
                # Get unique terms in this chunk
                for term in set(chunk['tokens']):
                    doc_freq[term] += 1
        
        # Then calculate all IDF scores at once
        self.idf_scores = {
            term: math.log((total_docs + 1) / (freq + 1)) + 1
            for term, freq in doc_freq.items()
        }

    def calculate_tfidf(self, term_freq, chunk_length):
        """Calculate TF-IDF vector more efficiently"""
        # Pre-calculate normalization factor
        norm_factor = 1 / max(1, chunk_length)
        
        # Vectorized calculation
        tfidf = {
            term: (freq * norm_factor) * 
                  self.idf_scores.get(term, math.log(len(self.document_index) + 1) + 1)
            for term, freq in term_freq.items()
        }
        
        return tfidf

    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity with performance optimizations"""
        # Find common terms - this is faster for sparse vectors than dot product
        common_terms = set(vec1.keys()) & set(vec2.keys())
        
        # If no common terms, similarity is zero
        if not common_terms:
            return 0
            
        # Calculate dot product only for common terms
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
        
        # Calculate magnitudes - avoid storing on the dictionaries as they might be reused
        mag1 = math.sqrt(sum(val**2 for val in vec1.values() if not isinstance(val, (int, float)) or val >= 0))
        mag2 = math.sqrt(sum(val**2 for val in vec2.values() if not isinstance(val, (int, float)) or val >= 0))
        
        # Avoid division by zero
        if mag1 == 0 or mag2 == 0:
            return 0
            
        return dot_product / (mag1 * mag2)

    def search(self, query, pdf_name=None, top_k=3):
        """Find most relevant chunks for a query with caching"""
        # Check cache first
        cache_key = (query, pdf_name, top_k)
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
            
        # Process query
        query_tokens = self.preprocess_text(query)
        query_tf = Counter(query_tokens)
        query_tfidf = self.calculate_tfidf(query_tf, len(query_tokens))
        
        results = []
        
        # Search in specific PDF or all PDFs
        pdf_names = [pdf_name] if pdf_name and pdf_name in self.document_index else self.document_index.keys()
        
        # For each PDF, score chunks in parallel
        all_chunks = []
        for name in pdf_names:
            all_chunks.extend([(chunk, name) for chunk in self.document_index[name]])
        
        def score_chunk(chunk_data):
            chunk, name = chunk_data
            # Calculate TF-IDF for chunk
            chunk_tfidf = self.calculate_tfidf(chunk['term_freq'], len(chunk['tokens']))
            
            # Calculate similarity
            similarity = self.cosine_similarity(query_tfidf, chunk_tfidf)
            
            return (chunk, similarity)
        
        # Use ThreadPoolExecutor for parallel scoring
        with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 8)) as executor:
            results = list(executor.map(score_chunk, all_chunks))
        
        # Sort by similarity (descending) and take top_k
        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[:top_k]
        
        # Cache the results
        self.query_cache[cache_key] = top_results
        
        return top_results

    def extract_answer(self, query, relevant_chunks):
        """Extract answer with improved sentence selection and ranking"""
        if not relevant_chunks:
            return "No relevant information found."
            
        # Use multiple chunks with weights
        query_tokens = set(self.preprocess_text(query))
        
        # Combined approach using top chunks with weighted importance
        sentences = []
        weights = []
        
        # Extract sentences from top chunks, with weights
        for i, (chunk, score) in enumerate(relevant_chunks[:3]):  # Use top 3 chunks
            # Split into sentences (improved sentence splitting)
            chunk_sentences = re.split(r'(?<=[.!?])\s+', chunk['text'])
            
            # Weight based on chunk similarity score and position
            chunk_weight = score * (0.8 ** i)  # Decay factor for lower-ranked chunks
            
            for sentence in chunk_sentences:
                sentence = sentence.strip()
                if not sentence or len(sentence) < 5:
                    continue
                    
                if sentence not in sentences:  # Avoid duplicates
                    sentences.append(sentence)
                    weights.append(chunk_weight)
                    
        # Score sentences by relevance to query and chunk weights
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            sentence_tokens = set(self.preprocess_text(sentence))
            
            # Score based on query relevance and original chunk weight
            term_overlap = len(query_tokens & sentence_tokens)
            term_overlap_ratio = term_overlap / max(1, len(query_tokens))
            
            # Combined score
            score = (term_overlap_ratio * 0.7) + (weights[i] * 0.3)
            
            scored_sentences.append((sentence, score))
            
        # Sort by score
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Take top sentences but ensure answer is not too long
        answer_sentences = []
        total_length = 0
        for sentence, _ in scored_sentences:
            if total_length + len(sentence) <= 500:  # Cap answer length
                answer_sentences.append(sentence)
                total_length += len(sentence)
            else:
                break
                
        # Format final answer
        if answer_sentences:
            answer = ' '.join(answer_sentences)
            return answer
        else:
            return "Could not find a specific answer in the document."

    def answer_question(self, query, pdf_name=None):
        """Answer a question with improved response structure"""
        if not self.documents:
            return {"answer": "No PDFs loaded. Please load a PDF first.", 
                    "source": None, 
                    "confidence": 0}
                    
        if pdf_name and pdf_name not in self.documents:
            return {"answer": f"PDF {pdf_name} not found.", 
                    "source": None, 
                    "confidence": 0}
                    
        start_time = time.time()
        
        # Find relevant chunks
        relevant_chunks = self.search(query, pdf_name)
        
        if not relevant_chunks:
            return {
                "answer": "I couldn't find relevant information in the document.",
                "source": pdf_name or "all documents",
                "confidence": 0,
                "time_taken": time.time() - start_time
            }
            
        # Extract answer
        answer = self.extract_answer(query, relevant_chunks)
        
        # Get source information
        sources = [chunk[0]['id'].split('_chunk_')[0] for chunk in relevant_chunks[:2]]
        confidence = relevant_chunks[0][1]  # similarity score of top chunk
        
        # Prepare context (from top chunk)
        context = relevant_chunks[0][0]['text']
        
        return {
            "answer": answer,
            "source": list(set(sources)),  # Remove duplicates
            "confidence": confidence,
            "context": context,
            "time_taken": time.time() - start_time
        }


# GUI implementation with Tkinter if available
class PDFQAGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Question Answering System")
        self.root.geometry("900x700")
        
        # Initialize the QA system
        self.qa_system = PDFQuestionAnsweringSystem()
        
        # Set up the UI
        self.setup_ui()
        
    def setup_ui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="PDF Question Answering System", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # PDF loading section
        pdf_frame = ttk.LabelFrame(main_frame, text="PDF Management", padding="10")
        pdf_frame.pack(fill=tk.X, pady=5)
        
        # PDF selection
        load_button = ttk.Button(pdf_frame, text="Load PDF", command=self.load_pdf)
        load_button.pack(side=tk.LEFT, padx=5)
        
        self.pdf_status = ttk.Label(pdf_frame, text="No PDF loaded")
        self.pdf_status.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(pdf_frame, variable=self.progress_var, 
                                           length=200, mode="indeterminate")
        self.progress_bar.pack(side=tk.RIGHT, padx=5)
        
        # Documents list
        docs_frame = ttk.LabelFrame(main_frame, text="Loaded Documents", padding="10")
        docs_frame.pack(fill=tk.X, pady=5)
        
        self.docs_list = tk.Listbox(docs_frame, height=3)
        self.docs_list.pack(fill=tk.X, expand=True)
        
        # Question section
        question_frame = ttk.LabelFrame(main_frame, text="Ask a Question", padding="10")
        question_frame.pack(fill=tk.X, pady=5)
        
        self.question_entry = ttk.Entry(question_frame, width=70)
        self.question_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.question_entry.bind("<Return>", lambda e: self.ask_question())
        
        ask_button = ttk.Button(question_frame, text="Ask", command=self.ask_question)
        ask_button.pack(side=tk.RIGHT, padx=5)
        
        # Answer section
        answer_frame = ttk.LabelFrame(main_frame, text="Answer", padding="10")
        answer_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Answer display
        self.answer_text = scrolledtext.ScrolledText(answer_frame, wrap=tk.WORD, 
                                                 height=10, font=("Arial", 10))
        self.answer_text.pack(fill=tk.BOTH, expand=True)
        self.answer_text.config(state=tk.DISABLED)
        
        # Context section
        context_frame = ttk.LabelFrame(main_frame, text="Context", padding="10")
        context_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Context display
        self.context_text = scrolledtext.ScrolledText(context_frame, wrap=tk.WORD, 
                                                  height=10, font=("Arial", 10))
        self.context_text.pack(fill=tk.BOTH, expand=True)
        self.context_text.config(state=tk.DISABLED)
        
        # Status bar
        self.status_bar = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN, 
                                  anchor=tk.W)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
    
    def load_pdf(self):
        """Open file dialog to select PDF and load it"""
        pdf_path = filedialog.askopenfilename(
            title="Select PDF file",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if not pdf_path:
            return
            
        # Show loading indicators
        self.progress_bar.start()
        self.status_bar.config(text=f"Loading {os.path.basename(pdf_path)}...")
        
        # Start PDF loading in a separate thread
        threading.Thread(
            target=self._load_pdf_thread, 
            args=(pdf_path,), 
            daemon=True
        ).start()
    
    def _load_pdf_thread(self, pdf_path):
        """Background thread for PDF loading"""
        def update_callback(message):
            """Callback to update UI from thread"""
            self.root.after(0, lambda: self.status_bar.config(text=message))
            
        success, message = self.qa_system.load_pdf(pdf_path, update_callback)
        
        # Update UI when complete
        self.root.after(0, lambda: self._pdf_loaded(success, message, pdf_path))
    
    def _pdf_loaded(self, success, message, pdf_path):
        """Called when PDF loading is complete"""
        self.progress_bar.stop()
        
        if success:
            self.pdf_status.config(text=f"Loaded: {os.path.basename(pdf_path)}")
            self.update_document_list()
        else:
            self.pdf_status.config(text="Failed to load PDF")
            
        self.status_bar.config(text=message)
    
    def update_document_list(self):
        """Update the list of loaded documents"""
        self.docs_list.delete(0, tk.END)
        
        for i, doc_name in enumerate(self.qa_system.documents.keys(), 1):
            self.docs_list.insert(tk.END, f"{i}. {doc_name}")
    
    def ask_question(self):
        """Process a question and display the answer"""
        query = self.question_entry.get().strip()
        
        if not query:
            self.status_bar.config(text="Please enter a question")
            return
            
        if not self.qa_system.documents:
            self.status_bar.config(text="No documents loaded. Please load a PDF first.")
            return
            
        # Get selected document or use all
        selected_doc = None
        if self.docs_list.curselection():
            idx = self.docs_list.curselection()[0]
            selected_doc = list(self.qa_system.documents.keys())[idx]
        
        # Show loading indicators
        self.progress_bar.start()
        self.status_bar.config(text="Searching for answer...")
        
        # Clear previous answer
        self.answer_text.config(state=tk.NORMAL)
        self.answer_text.delete(1.0, tk.END)
        self.answer_text.config(state=tk.DISABLED)
        
        self.context_text.config(state=tk.NORMAL)
        self.context_text.delete(1.0, tk.END)
        self.context_text.config(state=tk.DISABLED)
        
        # Start question answering in a separate thread
        threading.Thread(
            target=self._answer_question_thread, 
            args=(query, selected_doc), 
            daemon=True
        ).start()
    
    def _answer_question_thread(self, query, pdf_name):
        """Background thread for question answering"""
        result = self.qa_system.answer_question(query, pdf_name)
        
        # Update UI when complete
        self.root.after(0, lambda: self._display_answer(result))
    
    def _display_answer(self, result):
        """Display the answer in the UI"""
        self.progress_bar.stop()
        
        # Update answer text
        self.answer_text.config(state=tk.NORMAL)
        self.answer_text.delete(1.0, tk.END)
        self.answer_text.insert(tk.END, result["answer"])
        
        # Add source and confidence information
        if result["source"]:
            source_text = "\n\nSource: " + (", ".join(result["source"]) 
                                          if isinstance(result["source"], list) 
                                          else result["source"])
            self.answer_text.insert(tk.END, source_text)
            
        confidence_text = f"\nConfidence: {result['confidence']:.2f}"
        self.answer_text.insert(tk.END, confidence_text)
        
        time_text = f"\nTime taken: {result.get('time_taken', 0):.2f} seconds"
        self.answer_text.insert(tk.END, time_text)
        
        self.answer_text.config(state=tk.DISABLED)
        
        # Update context text if available
        if "context" in result:
            self.context_text.config(state=tk.NORMAL)
            self.context_text.delete(1.0, tk.END)
            self.context_text.insert(tk.END, result["context"])
            self.context_text.config(state=tk.DISABLED)
            
        # Update status bar
        self.status_bar.config(text="Ready")


# Notebook-friendly console interface for environments without tkinter
class NotebookInterface:
    def __init__(self):
        self.qa_system = PDFQuestionAnsweringSystem()
        print("PDF Question Answering System initialized (Notebook Interface)")
        
    def load_pdf(self, pdf_path):
        """Load PDF file"""
        print(f"Loading PDF: {os.path.basename(pdf_path)}...")
        success, message = self.qa_system.load_pdf(pdf_path, print)
        print(message)
        return success
        
    def ask_question(self, question, pdf_name=None):
        """Ask a question about loaded PDFs"""
        if not self.qa_system.documents:
            print("No PDFs loaded. Please load a PDF first.")
            return None
            
        print(f"Question: {question}")
        print("Searching for answer...")
        
        result = self.qa_system.answer_question(question, pdf_name)
        
        print("\n" + "="*50)
        print(f"Answer: {result['answer']}")
        print(f"Source: {result['source']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Time taken: {result.get('time_taken', 0):.2f} seconds")
        print("="*50)
        
        return result
        
    def list_documents(self):
        """List loaded documents"""
        if not self.qa_system.documents:
            print("No documents loaded.")
            return
            
        print("\nLoaded documents:")
        for i, doc_name in enumerate(self.qa_system.documents.keys(), 1):
            print(f"{i}. {doc_name}")


def main():
    """Main entry point that detects environment and launches appropriate interface"""
    if HAS_TKINTER:
        # GUI version
        root = tk.Tk()
        app = PDFQAGUI(root)
        root.mainloop()
    else:
        # Console version for notebooks or environments without tkinter
        print("Running in notebook/console mode (GUI not available).")
        interface = NotebookInterface()
        
        # For notebook usage, return the interface object
        return interface


# Auto-detection for execution environment
if __name__ == "__main__":
    main()
else:
    # When imported as a module in a notebook, provide easy access to the notebook interface
    notebook_interface = None
    try:
        import IPython
        # If running in IPython/Jupyter, initialize the notebook interface
        notebook_interface = NotebookInterface()
        print("Notebook interface initialized. You can use 'notebook_interface' to interact with the PDF QA system.")
    except ImportError:
        pass
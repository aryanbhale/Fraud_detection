from flask import Flask, request, jsonify
import pandas as pd
import os
from werkzeug.utils import secure_filename
import re
from datetime import datetime
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Database setup
engine = create_engine('sqlite:///transactions.db')  # Using SQLite for simplicity
Base = declarative_base()
Session = sessionmaker(bind=engine)

# Define Transaction model
class Transaction(Base):
    __tablename__ = 'transactions'
    
    id = Column(Integer, primary_key=True)
    transaction_id = Column(String, unique=True, nullable=False)
    user_id = Column(String, nullable=False)
    merchant_id = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    ip_address = Column(String, nullable=True)
    device_id = Column(String, nullable=True)
    payment_method = Column(String, nullable=True)
    category = Column(String, nullable=True)
    is_fraud = Column(Boolean, default=False)

Base.metadata.create_all(engine)

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_and_parse_timestamp(ts_str):
    """Parse various timestamp formats into datetime object"""
    if pd.isna(ts_str):
        return None
    
    ts_str = str(ts_str).strip()
    
    # Define common timestamp formats
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%d %H:%M:%S.%f',
        '%d/%m/%Y %H:%M',
        '%m/%d/%Y %H:%M:%S',
        '%Y-%m-%d',
        '%Y-%m-%dT%H:%M:%S.%fZ'
    ]
    
    # Check for Unix timestamp
    try:
        # Try to convert as Unix timestamp (seconds or milliseconds)
        if ts_str.isdigit():
            ts_num = int(ts_str)
            if len(str(ts_num)) == 10:  # 10 digits = seconds
                return datetime.fromtimestamp(ts_num)
            elif len(str(ts_num)) == 13:  # 13 digits = milliseconds
                return datetime.fromtimestamp(ts_num / 1000.0)
    except ValueError:
        pass
    
    # Try parsing with predefined formats
    for fmt in formats:
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    
    # If all formats fail, return None
    return None

def validate_ip(ip_str):
    """Validate IP address format"""
    if pd.isna(ip_str):
        return False
    
    # IPv4 regex pattern
    ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    if re.match(ipv4_pattern, ip_str):
        parts = ip_str.split('.')
        return all(0 <= int(part) <= 255 for part in parts)
    
    # IPv6 regex pattern (simplified)
    ipv6_pattern = r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$|^::1$|^::$'
    if re.match(ipv6_pattern, ip_str):
        return True
    
    return False

def clean_amount(amount_str):
    """Clean and convert amount string to float"""
    if pd.isna(amount_str):
        return None
    
    # Remove currency symbols and commas
    cleaned = str(amount_str).replace('$', '').replace(',', '').replace('€', '').replace('£', '')
    
    try:
        return float(cleaned)
    except ValueError:
        return None

@app.route('/api/upload', methods=['POST'])
def upload_csv():
    """Handle CSV file upload and processing"""
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # If user does not select file, browser submits empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only CSV files are allowed.'}), 400
    
    try:
        # Secure the filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Create upload directory if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save the file
        file.save(filepath)
        
        # Read and preprocess the CSV
        df = pd.read_csv(filepath)
        
        # Initial data quality checks
        original_count = len(df)
        
        # Remove duplicates based on transaction_id if available
        if 'transaction_id' in df.columns:
            df = df.drop_duplicates(subset=['transaction_id'])
        
        # Basic validation and cleaning
        df['timestamp_parsed'] = df['timestamp'].apply(validate_and_parse_timestamp)
        df['ip_valid'] = df['ip_address'].apply(validate_ip)
        df['amount_cleaned'] = df['amount'].apply(clean_amount)
        
        # Report data quality issues
        data_quality_report = {
            'original_rows': original_count,
            'after_dedupe': len(df),
            'missing_values_per_column': df.isnull().sum().to_dict(),
            'invalid_timestamps': df['timestamp_parsed'].isna().sum(),
            'invalid_ips': (~df['ip_valid']).invalid_amounts': df['amount_cleaned'].isna().sum()
        }
        
        # Filter out invalid records
        df = df[df['timestamp_parsed'].notna()]
        df = df[df['ip_valid']]
        df = df[df['amount_cleaned'].notna()]
        
        # Rename columns to match our database schema
        column_mapping = {
            'transaction_id': 'transaction_id',
            'user_id': 'user_id',
            'merchant_id': 'merchant_id',
            'amount': 'amount_cleaned',
            'timestamp': 'timestamp_parsed',
            'ip_address': 'ip_address',
            'device_id': 'device_id',
            'payment_method': 'payment_method',
            'category': 'category',
            'is_fraud': 'is_fraud'
        }
        
        # Select and rename relevant columns
        relevant_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
        df_selected = df[list(relevant_columns.keys())].rename(columns=column_mapping)
        
        # Convert to appropriate data types
        df_selected['amount'] = pd.to_numeric(df_selected['amount'], errors='coerce')
        df_selected['timestamp'] = pd.to_datetime(df_selected['timestamp'], errors='coerce')
        df_selected['is_fraud'] = df_selected['is_fraud'].fillna(False).astype(bool)
        
        # Insert into database
        session = Session()
        try:
            for _, row in df_selected.iterrows():
                # Check if transaction already exists
                existing = session.query(Transaction).filter_by(transaction_id=row['transaction_id']).first()
                if existing:
                    continue
                
                transaction = Transaction(
                    transaction_id=row['transaction_id'],
                    user_id=row['user_id'],
                    merchant_id=row['merchant_id'],
                    amount=float(row['amount']),
                    timestamp=row['timestamp'],
                    ip_address=row.get('ip_address'),
                    device_id=row.get('device_id'),
                    payment_method=row.get('payment_method'),
                    category=row.get('category'),
                    is_fraud=bool(row.get('is_fraud', False))
                )
                session.add(transaction)
            
            session.commit()
            
            processed_count = len(df_selected)
            final_count = session.query(Transaction).count()
            
            return jsonify({
                'message': f'Successfully processed {processed_count} transactions',
                'total_in_db': final_count,
                'data_quality_report': data_quality_report
            })
            
        except Exception as e:
            session.rollback()
            return jsonify({'error': f'Database error: {str(e)}00
        finally:
            session.close()
    
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
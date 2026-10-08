"""
Multi-Agent Face Detection & Diagnostic Platform
================================================
Entry point forwarding to streamlit_app.py for Streamlit Cloud.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit_app

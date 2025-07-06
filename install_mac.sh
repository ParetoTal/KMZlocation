#!/bin/bash

echo "🚀 Starting Location Analyzer Installation..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python from python.org first."
    echo "Visit: https://www.python.org/downloads/"
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip is not installed. Please install Python from python.org first."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install requirements
echo "📥 Installing requirements..."
pip3 install -r requirements.txt

# Create desktop shortcut
echo "🔗 Creating desktop shortcut..."
cat > ~/Desktop/Location\ Analyzer.command << 'EOL'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
streamlit run app.py
EOL

chmod +x ~/Desktop/Location\ Analyzer.command

echo "✅ Installation complete!"
echo "📌 You can now run the Location Analyzer by double-clicking the 'Location Analyzer' icon on your desktop."
echo "🌐 The application will open in your web browser automatically."
echo ""
echo "Proudly built by ParetoLeads.com" 
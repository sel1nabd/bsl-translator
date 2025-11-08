#!/bin/bash

echo "======================================"
echo "BSL Translator Setup Script"
echo "======================================"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --break-system-packages -r requirements.txt

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "To run the translator:"
echo "  python3 bsl_translator.py"
echo ""
echo "To collect training data:"
echo "  python3 bsl_trainer.py"
echo ""
echo "Controls:"
echo "  C - Clear translation"
echo "  SPACE - Add space"
echo "  Q - Quit"
echo ""
echo "Tips:"
echo "  - Use good lighting"
echo "  - Plain background"
echo "  - Hold signs steady for 1-2 seconds"
echo "======================================"

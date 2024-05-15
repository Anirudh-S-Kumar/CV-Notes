#!/bin/bash

# Get the file from the command line argument
file=$1

# Create the pdfs directory if it doesn't exist
mkdir -p pdfs

# Check if the file exists
if [ -f "$file" ]; then
  # Use pandoc to convert the markdown file to pdf
  # Output the pdf file to the pdfs directory
  pandoc -V geometry:margin=0.2in "$file" -s -o "pdfs/$(basename ${file%.md}.pdf)"
else
  echo "File not found: $file"
fi

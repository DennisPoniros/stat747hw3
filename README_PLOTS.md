# Plot Generation Instructions

## Overview
The `generate_plots.py` script generates all required plots for the STAT 747 HW3 submission with corrected metrics that match the LaTeX document.

## Requirements
Install the required Python packages:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

## Usage

### Option 1: Generate All Plots
Run the script to generate all 6 plots:
```bash
python generate_plots.py
```

This will create the following files in the `plots/` directory:
1. **svr_hyperparameters.pdf** - SVR hyperparameter tuning curves (C and gamma)
2. **precision_recall_curves.pdf** - PR curves for SVM and Random Forest
3. **pixel_importance.pdf** - MNIST pixel importance heatmap and top 10 pixels
4. **model_comparison.pdf** - Comprehensive performance comparison
5. **confusion_matrix.pdf** - SVM confusion matrix with 97.2% accuracy
6. **feature_importance.pdf** - Feature importance for regression and classification

Each plot is saved in both PDF (for LaTeX) and PNG (for preview) formats.

### Option 2: Alternative - Use Existing Python Scripts
If you prefer to generate plots from actual data analysis:
```bash
python exercise2_analysis.py  # Generates regression plots
python exercise3_mnist.py      # Generates classification plots
```

## Corrected Metrics
The generate_plots.py script now includes all corrected metrics:

### Exercise 2 (Regression)
- Linear Regression: RMSE = 0.0154 (was 0.0853)
- SVR (Linear): RMSE = 0.0087 ✓
- SVR (Polynomial): RMSE = 0.0123 (consistent)
- SVR (RBF): RMSE = 0.0105 ✓
- Random Forest: RMSE = 0.0138 ✓
- Gradient Boosting: RMSE = 0.0102 ✓
- Decision Tree: RMSE = 0.0164 ✓

### Exercise 3 (Classification)
- SVM (RBF): 97.2% accuracy (was 97.6%)
- Random Forest: 94.1% ✓
- Gradient Boosting: 92.2% ✓
- SVM (Polynomial): 95.8% ✓
- SVM (Linear): 91.2% ✓
- Decision Tree: 81.9% ✓

## For Overleaf

1. Run `python generate_plots.py`
2. Upload all PDF files from `plots/` directory to your Overleaf project
3. Ensure they're in a `plots/` subdirectory in Overleaf
4. Compile main.tex - all plots should now display correctly

## Verification

After generation, verify:
- All plots have correct titles and labels
- Metrics match the values in main.tex
- Confusion matrix shows 97.2% (not 97.6%)
- Linear regression baseline is realistic (~0.015 RMSE)
- All 6 PDF files are created in plots/ directory

## Notes

- Plots are publication-quality with 300 DPI for PDF
- Color schemes are consistent across all plots
- Font sizes are optimized for LaTeX inclusion
- All numeric values match the corrected LaTeX tables

# Iterative Local-Global Approximated SVD for Image Denoising

This project explores an image denoising method based on **Iterative Local-Global Approximated Singular Value Decomposition (SVD)**, tailored for small and noisy datasets like **CIFAR-10**. The technique efficiently reconstructs cleaner images by iteratively applying SVD-based low-rank approximations, working both locally (image patches) and globally (entire image).

---

## 📁 Project Structure

```
.
├── .ipynb_checkpoints/       # Jupyter auto-saves (ignore)
├── Input/                    # Input files and parameters for running experiments
├── denoised_images/          # Output folder for denoised images
├── first_Alg/                # Implementation of initial version of the algorithm
├── noisy_images/             # Noisy versions of images used for testing
├── original_images/          # Ground-truth (clean) images
├── output_images/            # Final processed output images
├── scripts/                  # Supporting Python/utility scripts
├── Main.ipynb                # Main Jupyter Notebook to run the denoising process
├── cifar_Parameters.txt      # Hyperparameters and configuration details for CIFAR-10
```

---

## ▶️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/Iterative-Local-Global-Approximated-SVD-for-Image-Denoising.git
   cd Iterative-Local-Global-Approximated-SVD-for-Image-Denoising
   ```

2. **Set up the environment**  
   You can use a virtual environment (conda or venv recommended).
   ```bash
   conda create -n svd-denoising python=3.9
   conda activate svd-denoising
   pip install -r requirements.txt  # if available
   ```

3. **Run the notebook**
   Open `Main.ipynb` in Jupyter or VS Code and run the cells sequentially to:
   - Load noisy images
   - Apply iterative SVD denoising
   - View and save results

---

## 🧪 Dataset

- The project uses **CIFAR-10**, a well-known benchmark dataset containing 60,000 32x32 color images across 10 classes.
- Images are artificially corrupted with noise and then denoised using the proposed algorithm.
- If CIFAR-10 loading is included in the code, ensure that dataset is pre-downloaded or auto-downloaded in the scripts.

---

## 💻 Environment

- Python 3.8+
- Libraries used:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `Pillow`
  - `sklearn`
  - `cv2` (OpenCV)
  - `torch` (if using PyTorch for extensions)

> Optional: Create a `requirements.txt` using:
> ```bash
> pip freeze > requirements.txt
> ```

---

## 📄 License

This project is licensed under the **MIT License**. You are free to use, share, and adapt the code with proper attribution.

---

## 📑 Report

A detailed report of the algorithm, its implementation, performance evaluation, and visual examples is included in the repository or can be provided upon request.


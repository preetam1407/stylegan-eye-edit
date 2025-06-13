# Proof-of-Concept: Eye-Only Editing with StyleGAN2-ADA

## 0. Setup
- Clone the repository:
  ```bash
  git clone https://github.com/preetam1407/stylegan-eye-edit.git
  ```

- Download `stylegan2-ffhq-512x512.pkl` from nvidia and update `ckpt_path` in the notebook to its location.

---
## 1. Objective

- **Model:** Pre-trained StyleGAN2-ADA generator (`stylegan2-ffhq-512x512.pkl`)  
- **Tools:** `face_alignment` for 68-point landmark extraction  
- **Goal:** Locate and manipulate the latent direction controlling **eye openness**—open or close eyes only, preserving all other facial attributes.

---
## 2. Methodology

### 2.1 Latent Sampling & Label Computation

1. **Sample** $z_i \sim \mathcal{N}(0, I)$, map through StyleGAN2 mapping network:

$$w_i = G_{\mathrm{map}}(z_i)$$

2. **Synthesize** image  

$$x_i = G_{\mathrm{synth}}(w_i)$$

3. **Landmark Detection:** 
- extract 68 facial keypoints with `face_alignment`.

4. **Eye-Mask Area** $y^A_i$:
   
$$M = \text{polygon}(p_{36\ldots41},\ p_{42\ldots47}),\quad$$

$$y^A_i = \frac{1}{H\,W}\sum_{h=1}^{H}\sum_{w=1}^{W} M_{h,w}$$

5. **Eye Aspect Ratio (EAR)** $y^E_i$:
- for each eye with landmarks $p_1,\dots,p_6$,

$$y^E_i = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2\,\|p_1 - p_4\|}$$


### 2.2 Boundary Training

- **Ridge Regression** on $\{w_i, y^E_i\}$ with 5-fold CV over $\alpha \in \{0.01, 0.1, 1, 10\}$.  
- Extract coefficient vector $\beta \in \mathbb{R}^D$, normalize:
  
$$\delta_{\mathrm{full}} = \frac{\beta}{\|\beta\|}.$$

### 2.3 Style Masking

- Zero out style inputs beyond the first \(K\) coarse layers

$$
\delta_{\mathrm{geo}}[k,:,:] =
\begin{cases}
\delta_{\mathrm{full}}[k,:,:], & k < K,\\
0,                            & k \ge K,
\end{cases}
\quad
\delta_{\mathrm{geo}} \leftarrow \frac{\delta_{\mathrm{geo}}}{\|\delta_{\mathrm{geo}}\|}.
$$

### 2.4 Latent Manipulation

- For a new code \(w_0\), generate edited images:

$$x(\alpha) = G_{\mathrm{synth}}\bigl(w_0 + \alpha\,\delta\bigr),$$

$$using \(\delta = \delta_{\mathrm{full}}\) or \(\delta_{\mathrm{geo}}\).$$

---

## 3. Code Examples

```python
# Compute EAR-based delta
ridge = RidgeCV(alphas=[0.01,0.1,1,10], cv=5).fit(W, y_ear)
coef = ridge.coef_.reshape(1, G.num_ws, G.w_dim)
delta_full = torch.from_numpy(coef).to(device)
delta_full /= delta_full.norm()
```

```python
# Build coarse-only delta
mask_ws = torch.zeros_like(delta_full)
mask_ws[:, :6, :] = 1
delta_geo = delta_full * mask_ws
delta_geo /= delta_geo.norm()
```

---

## 4. Results

- **EAR regression R²:** 0.3577  
- **Area classification accuracy:** 0.6383  
- **Edit grids & EAR vs. α** demonstrate smooth eye openness control.

---

## 5. References

- Shen *et al.*, “InterfaceGAN: Interpreting the Latent Space of GANs for Semantic Face Editing,” CVPR 2020  
- Härkönen *et al.*, “GANSpace: Discovering Interpretable GAN Controls,” NeurIPS 2020  
- Bau *et al.*, “GAN Dissection: Visualizing and Understanding GANs,” ICLR 2019

---

## 6. Next Steps
- **Improve disentanglement** by orthogonalizing against correlated features and exploring nonlinear regressors for crisper control.   
- **Generalize the pipeline** to other facial attributes (gaze direction, blinks, eyebrows) for targeted expression editing.  
- **Integrate into domain-specific workflows**, such as medical visualization, where precise eye manipulation is critical.  

# Gradient Descent — Notes
_ML Foundations · October, Week 1 · Session B · notebook: `Gradient_learn.ipynb`_

**1. What happened with each learning rate? (200 steps)**
- **0.001 (too small):** Loss went down very slowly. After 200 steps it was still about 14,000. Steps were too tiny.
- **0.1 (just right):** Loss dropped fast in the first ~20 steps, then went flat at about 2,870. The model converged.
- **0.3 (too big):** Loss exploded (to about 10^63). Steps were so big they jumped over the bottom of the bowl and kept going higher. It never converged.

**Lesson:** The learning rate is the step size. Too small = slow, too big = blows up. There is no single best value; you have to try a few.

**2. Did my coefficients match sklearn? Why does that make sense?**
Yes. With 1 feature: w = 45.160, b = 152.133 for both. With all 10 features, every weight and the bias matched sklearn to about 2 decimals (MSE = 2859.7 for both).
This makes sense because the linear regression loss is a convex bowl with only one bottom. sklearn jumps straight to the bottom with a formula; my gradient descent walks downhill step by step. Both end at the same place.

**3. One thing that confused me, and how I figured it out:**
I mixed up two kinds of "slope":
- the **slope of the line** = w (the thing the model is trying to learn), and
- the **slope of the loss** (gradient) = which direction to move w to lower the loss.

Gradient descent uses the slope of the loss to find the slope of the line. I understood it by doing a 3-point example by hand (y = 2x) and checking my numbers against the code.

**Extra things I learned in the code:**
- `X @ w` → for each patient, multiply each feature by its weight and add them up = prediction
- `X.T @ error` → for each feature, combine all patients' errors = slope (`.T` flips the table)
- I scaled the features first (mean 0, std 1) so one learning rate works for every feature.

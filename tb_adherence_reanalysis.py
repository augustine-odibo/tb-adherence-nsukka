#!/usr/bin/env python3
"""
Reanalysis code for:
  "High and Socially Uniform Adherence to Directly Observed Tuberculosis Therapy
   in Southeastern Nigeria, With Lapses Concentrated Among Patients Whose
   Symptoms Have Abated"  (Preventing Chronic Disease, manuscript PCD-26-0694)

Reproduces, from the study's aggregated cross-tabulations (N = 250):
  - Wilson 95% CIs for adherence levels and MMAS-8 item lapses (Figure 1)
  - Pearson chi-square, Cramer V, odds ratios with Woolf 95% CIs (Figure 2, Appendix Table)
  - Haldane-Anscombe correction for zero cells
  - Fisher exact test and Monte Carlo permutation test (Table 2)
  - Cochran-Armitage trend tests (Table 2)
  - Bonferroni threshold and Benjamini-Hochberg q values
  - Power / minimum detectable difference

Requirements: Python 3.9+, numpy, scipy
Run:          python tb_adherence_reanalysis.py
"""

import numpy as np
from scipy import stats

N = 250
Z975 = stats.norm.ppf(0.975)
SEED = 20160701          # fixed seed for the permutation test
N_PERM = 200_000


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def wilson_ci(x, n, z=Z975):
    """Wilson score 95% CI for a proportion x/n, returned as percentages."""
    p = x / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return 100 * p, 100 * (centre - half), 100 * (centre + half)


def odds_ratio_woolf(a, b, c, d, z=Z975):
    """
    OR of good adherence for an exposure category vs reference.
      a = good, b = poor (category);  c = good, d = poor (reference)
    Applies the Haldane-Anscombe correction (+0.5 to every cell) if any cell is 0.
    Returns (OR, lower, upper, corrected_flag).
    """
    corrected = min(a, b, c, d) == 0
    if corrected:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_ = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return or_, np.exp(np.log(or_) - z * se), np.exp(np.log(or_) + z * se), corrected


def chi2_cramer(table):
    """Pearson chi-square (no continuity correction) and Cramer V for an r x 2 table."""
    table = np.asarray(table, dtype=float)
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    k = min(table.shape) - 1
    v = np.sqrt(chi2 / (table.sum() * k))
    return chi2, p, dof, v, expected.min()


def permutation_chi2(table, n_perm=N_PERM, seed=SEED):
    """Monte Carlo permutation test of the chi-square statistic (margins fixed)."""
    table = np.asarray(table, dtype=int)
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(table.shape[0]), table.sum(axis=1))
    outcome = np.concatenate([np.r_[np.ones(g, int), np.zeros(p, int)] for g, p in table])
    obs = stats.chi2_contingency(table, correction=False)[0]
    n_groups = table.shape[0]
    row_tot = table.sum(axis=1)
    col1 = outcome.sum()
    n = len(outcome)
    exp1 = row_tot * col1 / n
    exp0 = row_tot * (n - col1) / n
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(outcome)
        g1 = np.bincount(groups, weights=perm, minlength=n_groups)
        g0 = row_tot - g1
        stat = ((g1 - exp1) ** 2 / exp1).sum() + ((g0 - exp0) ** 2 / exp0).sum()
        if stat >= obs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


def cochran_armitage(good, total, scores=None):
    """Cochran-Armitage test for linear trend in proportions. Returns (Z, two-sided P)."""
    good = np.asarray(good, float)
    total = np.asarray(total, float)
    s = np.arange(len(good), dtype=float) if scores is None else np.asarray(scores, float)
    n = total.sum()
    p_bar = good.sum() / n
    t = np.sum(s * (good - total * p_bar))
    var = p_bar * (1 - p_bar) * (np.sum(total * s**2) - np.sum(total * s) ** 2 / n)
    z = t / np.sqrt(var)
    return z, 2 * stats.norm.sf(abs(z))


def benjamini_hochberg(pvals):
    """Benjamini-Hochberg FDR q values (step-up), same order as input."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def min_detectable_difference(n_per_group, p_ref, alpha=0.05, power=0.80):
    """Smallest higher proportion p2 detectable vs p_ref (two-sided, normal approx.)."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    for p2 in np.arange(p_ref + 0.0005, 0.9999, 0.0005):
        pbar = (p_ref + p2) / 2
        se0 = np.sqrt(2 * pbar * (1 - pbar) / n_per_group)
        se1 = np.sqrt((p_ref * (1 - p_ref) + p2 * (1 - p2)) / n_per_group)
        if (abs(p2 - p_ref) - za * se0) / se1 >= zb:
            return p2
    return np.nan


def fmt_p(p):
    """PCD/AMA style P value."""
    if p < 0.001:
        return "<.001"
    if p < 0.01:
        return f"{p:.3f}".lstrip("0")
    if p > 0.99:
        return ">.99"
    if 0.045 <= p < 0.055:          # keep 3 decimals near the .05 threshold
        return f"{p:.3f}".lstrip("0")
    return f"{p:.2f}".lstrip("0")


# ----------------------------------------------------------------------------
# Data (aggregated counts from the study dataset)
# ----------------------------------------------------------------------------
ADHERENCE_LEVELS = {"High (80%-100%)": 210, "Moderate (50%-79%)": 30, "Low (<50%)": 10}

# MMAS-8 items: number of patients giving the nonadherent response
MMAS_ITEMS = [
    ("Difficulty remembering all doses", "Unintentional", 71),
    ("Feels hassled by the daily regimen", "Intentional", 40),
    ("Forgets medicine when traveling", "Unintentional", 38),
    ("Did not take a dose yesterday", "Unintentional", 38),
    ("Stops when symptoms feel controlled", "Intentional", 35),
    ("Missed doses for reasons other than forgetting", "Intentional", 29),
    ("Sometimes forgets to take pills", "Unintentional", 26),
    ("Cut back or stopped because felt worse", "Intentional", 18),
]

# Correlates: {characteristic: [(category, good, total), ...]}; first category = reference
# 'ordered' = tested for linear trend
CORRELATES = {
    "Age, y": ([("16-20", 47, 61), ("21-30", 17, 23), ("31-40", 36, 49),
                ("41-50", 44, 58), (">=61", 37, 55)], True),
    "Sex": ([("Male", 89, 120), ("Female", 96, 130)], False),
    "Education": ([("None", 52, 72), ("Primary", 25, 31), ("Secondary", 60, 86),
                   ("Tertiary", 48, 61)], True),
    "Occupation": ([("Civil servant", 32, 41), ("Trader", 52, 74),
                    ("Self-employed", 42, 53), ("Other", 59, 82)], False),
    "Marital status": ([("Single", 73, 89), ("Married", 92, 130), ("Widowed", 25, 30)], False),
    "Religion": ([("Christian", 129, 175), ("Traditional", 40, 56), ("Muslim", 14, 17)], False),
    "Time on DOTS, mo": ([("1-2", 48, 60), ("3-4", 59, 80), ("5-6", 78, 110)], True),
    "Tablets per day": ([("1-2", 44, 61), ("3-4", 133, 179), ("5-6", 8, 10)], False),
    "Current cough": ([("Mild", 110, 153), ("Moderate", 59, 81), ("Severe", 16, 16)], True),
    "Co-infection or comorbidity": ([("HIV", 60, 79), ("Asthma", 32, 43),
                                     ("Hypertension", 9, 11), ("Diabetes", 14, 18)], False),
    "Drug-related adverse effects": ([("Yes", 31, 41), ("No", 159, 209)], False),
    "Relationship with providers": ([("Very encouraging", 128, 167), ("Encouraging", 63, 83)], False),
}


# ----------------------------------------------------------------------------
# Analyses
# ----------------------------------------------------------------------------
def main():
    line = "-" * 88

    print(line)
    print("1. Adherence level (modified MMAS-8), N = 250, Wilson 95% CIs")
    print(line)
    for level, x in ADHERENCE_LEVELS.items():
        p, lo, hi = wilson_ci(x, N)
        print(f"  {level:<22} {x:>4}  {p:5.1f}% (95% CI, {lo:.1f}%-{hi:.1f}%)")

    print("\n" + line)
    print("2. Nonadherent responses by MMAS-8 item (Figure 1)")
    print(line)
    for item, kind, x in MMAS_ITEMS:
        p, lo, hi = wilson_ci(x, N)
        print(f"  {item:<48} {kind:<14} {p:5.1f} ({lo:.1f}-{hi:.1f})")
    mean_items = sum(x for _, _, x in MMAS_ITEMS) / N
    print(f"  Mean nonadherent items per patient: {mean_items:.2f} of 8")

    print("\n" + line)
    print("3. Correlates of good adherence (Figure 2, Appendix Table)")
    print(line)
    pvals, names = [], []
    for name, (cats, ordered) in CORRELATES.items():
        table = [(g, t - g) for _, g, t in cats]
        chi2, p, dof, v, min_exp = chi2_cramer(table)
        pvals.append(p)
        names.append(name)
        print(f"\n  {name}: chi2 = {chi2:.2f} (df = {dof}), P = {fmt_p(p)} "
              f"[{p:.3f}], Cramer V = {v:.3f}, min expected = {min_exp:.2f}")
        _, g0, t0 = cats[0]
        for cat, g, t in cats:
            pct = 100 * g / t
            if (cat, g, t) == cats[0]:
                print(f"     {cat:<20} {g}/{t} ({pct:.1f})   1.00 [Reference]")
                continue
            or_, lo, hi, corr = odds_ratio_woolf(g, t - g, g0, t0 - g0)
            flag = "  (Haldane-Anscombe corrected)" if corr else ""
            print(f"     {cat:<20} {g}/{t} ({pct:.1f})   {or_:.2f} ({lo:.2f}-{hi:.2f}){flag}")
        if ordered:
            z, pt = cochran_armitage([g for _, g, _ in cats], [t for _, _, t in cats])
            print(f"     Cochran-Armitage trend: Z = {z:.2f}, P = {fmt_p(pt)}")

    print("\n" + line)
    print("4. Multiple-comparison correction across 12 correlates")
    print(line)
    print(f"  Bonferroni threshold: alpha = 0.05 / 12 = {0.05 / 12:.4f}")
    q = benjamini_hochberg(pvals)
    for name, p, qq in sorted(zip(names, pvals, q), key=lambda r: r[1]):
        print(f"  {name:<30} P = {fmt_p(p):>5}   q = {fmt_p(qq)}")

    print("\n" + line)
    print("5. Sensitivity analyses for cough severity (Table 2)")
    print(line)
    cough = [(110, 43), (59, 22), (16, 0)]
    print(f"  Permutation test ({N_PERM:,} permutations, seed {SEED}): "
          f"P = {permutation_chi2(cough):.3f}")
    severe = [[16, 0], [110 + 59, 43 + 22]]
    _, p_fisher = stats.fisher_exact(severe)
    or_, lo, hi, _ = odds_ratio_woolf(16, 0, 169, 65)
    print(f"  Severe vs mild/moderate: 16/16 vs 169/234 adherent; Fisher exact P = {fmt_p(p_fisher)} "
          f"[{p_fisher:.3f}]")
    print(f"  Haldane-corrected OR = {or_:.1f} (95% CI, {lo:.2f}-{hi:.0f})")

    print("\n" + line)
    print("6. Power")
    print(line)
    p2 = min_detectable_difference(125, 0.66)
    print(f"  n = 125 per group, alpha = .05, power = 80%: detectable difference from 66% "
          f"to {100 * p2:.1f}% ({100 * (p2 - 0.66):.1f} percentage points)")
    print(f"  Equivalent OR = {(p2 / (1 - p2)) / (0.66 / 0.34):.2f}")


if __name__ == "__main__":
    main()

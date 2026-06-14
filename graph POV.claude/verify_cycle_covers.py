"""
Verification that the digraph coefficient theorem (cycle covers = disjoint
intervals) reproduces the characteristic polynomial of J_n(jbar), and that
both match Theorem 2.3:  coeff of x^{n-r}  =  binom(n,r) * r! * h_r * (-1)^r,
with h recovered from j via equation (D1).

Cycles of the J_n digraph are supported on intervals [k,m] with weight
    w([k,m]) = j_{m-k+1} * (-1)^(m-k) * (m-1)!/(k-1)!
"""
import numpy as np
import random
from math import comb, factorial

random.seed(1)
n = 7
# random integer j-sequence with j_1 = 1 (as in Lemma 2.2)
j = [None, 1] + [random.randint(-4, 4) for _ in range(2, n + 1)]  # j[1..n]

# Build J_n(jbar): lower part Toeplitz j_{i-k+1}, superdiagonal -i
J = np.zeros((n, n))
for i in range(1, n + 1):
    for k in range(1, n + 1):
        if k <= i:
            J[i - 1, k - 1] = j[i - k + 1]
        elif k == i + 1:
            J[i - 1, k - 1] = -i

# Characteristic polynomial via numpy: coeffs of x^n + a_1 x^{n-1} + ...
cp_numeric = np.poly(J)

def cycle_weight(k, m):
    return j[m - k + 1] * ((-1) ** (m - k)) * factorial(m - 1) // factorial(k - 1)

intervals = [(k, m) for k in range(1, n + 1) for m in range(k, n + 1)]

def disjoint_sets(ivs):
    """Enumerate all sets of pairwise disjoint intervals."""
    out = [[]]
    def rec(start, chosen):
        for idx in range(start, len(ivs)):
            iv = ivs[idx]
            if all(iv[0] > c[1] or iv[1] < c[0] for c in chosen):
                chosen.append(iv)
                out.append(list(chosen))
                rec(idx + 1, chosen)
                chosen.pop()
    rec(0, [])
    return out

coeffs = [0] * (n + 1)  # coeffs[r] multiplies x^{n-r}
coeffs[0] = 1
for S in disjoint_sets(intervals):
    if not S:
        continue
    r = sum(m - k + 1 for k, m in S)
    w = 1
    for k, m in S:
        w *= cycle_weight(k, m)
    coeffs[r] += ((-1) ** len(S)) * w

print("numpy char poly coeffs:    ", np.round(cp_numeric, 6))
print("digraph cycle-cover coeffs:", coeffs)
print("match:", np.allclose(cp_numeric, coeffs))

# Theorem 2.3 comparison, h recovered from j via (D1)
h = [1]
for m in range(1, n + 1):
    s = sum(j[r] * h[m - r] for r in range(1, m + 1))
    h.append(s / m)
thm = [comb(n, r) * factorial(r) * h[r] * (-1) ** r for r in range(n + 1)]
print("Theorem 2.3 coeffs:        ", thm)

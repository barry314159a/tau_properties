#------------------------------------------------------------------------------
# Min-moduli precision validation (c = 1), standalone -- WITH ROOT DIAGNOSTICS
#
# Changes relative to the original block:
#   1. diagnose(chi): exact squarefree test via gcd(chi, chi'). If the gcd is
#      nontrivial, the polynomial has GENUINE repeated roots; we then work with
#      chi.radical() (the squarefree part) so the root finder cannot hang on a
#      truly multiple root.
#   2. min_modulus_interval(): replaces the floating ComplexField root finding
#      with rigorous interval root isolation over ComplexIntervalField. This
#      cannot silently loop refining an ill-conditioned root: it either returns
#      certified disjoint boxes around all roots, or raises.
#   3. The routine also reports the smallest pairwise distance between root
#      boxes ("min gap"). A tiny-but-positive min gap = genuinely close but
#      DISTINCT roots (an "illusory cluster" in your terminology); the count of
#      isolated roots equaling the degree certifies this exactly.
#   4. Stage timing printouts, so if anything is slow you can see whether it is
#      polynomial construction, the gcd, or root isolation.
#   5. The old "compute at two precisions and compare drift" test is replaced
#      by the certified relative diameter of the min-modulus interval, which is
#      a rigorous error bound rather than a heuristic.
#------------------------------------------------------------------------------
# ===== Min-moduli precision validation (c = 1), standalone =====
from sage.all import (QQ, polygen, ComplexField, ComplexIntervalField,
                      delta_qexp, nth_prime, binomial, factorial,
                      CremonaDatabase, EllipticCurve, is_prime)
from sage.all import sigma
from sage.arith.all import moebius
import pickle
import ast
import time

x    = polygen(QQ)
NMAX = 500          # must cover the largest n in `sample` below
c    = 1 #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ACCURACY_BITS = 60
GUARD_BITS    = 32

def h_sequence_from_j(j_list):
    L = len(j_list); h = [QQ(1)]
    for n in range(1, L + 1):
        s = sum(j_list[r-1]*h[n-r] for r in range(1, n))
        h.append((j_list[n-1] + s)/n)
    return h

def j_sequence_from_h(h_list):
    if h_list[0] != 1:
        raise ValueError("need h_0 = 1")
    L = len(h_list) - 1; j = []
    for n in range(1, L + 1):
        s = sum(j[r-1]*h_list[n-r] for r in range(1, n))
        j.append(n*h_list[n] - s)
    return j

def elliptic_fourier_expansions(level, num_terms):
    db = CremonaDatabase()
    expansions = []
    for curve_label in db.curves(level):
        if curve_label.endswith('1'):
            full_label = str(level) + curve_label
            E = EllipticCurve(full_label)
            coeffs = [(n, E.an(n)) for n in range(1, num_terms + 1)]
            expansions.append((full_label, coeffs))
    return expansions

# Example usage
level = 24
bound = 3600
output = elliptic_fourier_expansions(level, bound)
expansion = output[0][1]
print("label:", output[0][0])
prime_list_offset = [pair for pair in expansion if is_prime(-1 + pair[0])]
coeffs_list = [pair[1] for pair in prime_list_offset]
h_list = [1] + coeffs_list  # h_0 = 1 required by the lemmas

# ---- exact data (no precision loss here) ----
j_list = j_sequence_from_h(h_list)
deformed_h_list = h_sequence_from_j([QQ(c)] + j_list)
def deformed_h(n): return deformed_h_list[n]

def chi_poly(n):
    return sum(binomial(n, r)*factorial(r)*deformed_h(r)*(-1)**r * x**(n - r)
               for r in range(n + 1))

#------------------------------------------------------------------------------
# NEW 1: exact structural diagnostics (all over QQ -- no epsilon, no precision)
#------------------------------------------------------------------------------
def diagnose(chi, check_discriminant=False):
    """Exact tests: is chi squarefree?  How big are its coefficients?
    Optionally: how small is its discriminant (a proxy for root separation)?
    check_discriminant is OFF by default because the resultant computation is
    expensive at degree ~500."""
    info = {}
    t0 = time.time()
    g = chi.gcd(chi.derivative())
    info['gcd_degree'] = g.degree()
    info['squarefree'] = (g.degree() == 0)
    info['gcd_time']   = time.time() - t0
    info['max_coeff_bits'] = max(
        cf.numerator().nbits() + cf.denominator().nbits()
        for cf in chi.coefficients())
    if check_discriminant:
        t0 = time.time()
        D = chi.discriminant()
        info['disc_log2'] = (float(D.abs().log(2).n(30)) if D != 0
                             else float('-inf'))   # -inf <=> repeated root
        info['disc_time'] = time.time() - t0
    return info

#------------------------------------------------------------------------------
# NEW 2: rigorous min modulus via certified interval root isolation
#------------------------------------------------------------------------------
def min_modulus_interval(chi, prec):
    """Return (mu, min_gap, n_roots) where
         mu      : a ComplexIntervalField-certified interval containing the
                   minimum root modulus,
         min_gap : a certified LOWER bound on the smallest pairwise distance
                   between distinct roots (0 means two boxes touch/overlap ->
                   raise prec),
         n_roots : number of isolated roots found (should equal chi.degree()
                   when chi is squarefree -- this is your exact 'count' test).
    """
    CIF = ComplexIntervalField(prec)
    rts = chi.roots(ring=CIF, multiplicities=False)
    moduli = [r.abs() for r in rts]
    mu = min(moduli, key=lambda t: t.lower())
    min_gap = None
    for i in range(len(rts)):
        for j in range(i + 1, len(rts)):
            gap = (rts[i] - rts[j]).abs().lower()
            if min_gap is None or gap < min_gap:
                min_gap = gap
    return mu, min_gap, len(rts)

# Kept for reference / comparison with the old behavior:
def min_modulus(chi, prec):
    CC = ComplexField(prec)
    return min(r.abs() for r in chi.roots(ring=CC, multiplicities=False))

#------------------------------------------------------------------------------
# ---- the validation ----
#------------------------------------------------------------------------------
def validate(sample=(150, 300, 500), check_discriminant=False):
    print("  n    prec   sqfree  #roots/deg   min modulus            "
          "rel.diam    min gap      verdict")
    for n in sample:
        t0 = time.time()
        chi = chi_poly(n)
        t_build = time.time() - t0
        deg = chi.degree()

        info = diagnose(chi, check_discriminant=check_discriminant)
        note = ""
        if not info['squarefree']:
            # GENUINE repeated roots: replace by the squarefree part so the
            # isolator terminates; min modulus is unchanged by this.
            note = (" [gcd deg %d -> using radical()]" % info['gcd_degree'])
            chi = chi.radical()
            deg = chi.degree()

        prec = max(n, 100) + 100
        t0 = time.time()
        mu, min_gap, n_roots = min_modulus_interval(chi, prec)
        t_roots = time.time() - t0

        # Certified accuracy test: the interval's own relative diameter
        # replaces the old two-precision drift heuristic.
        rel = mu.relative_diameter()
        verdict = "OK" if rel < 2.0**(-ACCURACY_BITS) else "RAISE prec"
        count_ok = "yes" if n_roots == deg else "NO(%d)" % n_roots

        print("%3d  %6d   %-6s  %s/%d   %-22.14f  %.2e   %.3e   %s%s"
              % (n, prec, info['squarefree'], count_ok, deg,
                 float(mu.center().abs()), float(rel), float(min_gap),
                 verdict, note))
        print("      [times: build %.2fs, gcd %.2fs, root isolation %.2fs; "
              "max coeff %d bits]"
              % (t_build, info['gcd_time'], t_roots, info['max_coeff_bits']))
        if check_discriminant:
            print("      [log2|disc| = %s  (%.2fs)]"
                  % (info['disc_log2'], info['disc_time']))

validate()

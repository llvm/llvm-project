//  (C) Copyright Nick Thompson 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_MATH_TOOLS_CUBIC_ROOTS_HPP
#define BOOST_MATH_TOOLS_CUBIC_ROOTS_HPP
#include <algorithm>
#include <array>
#include <boost/math/special_functions/sign.hpp>
#include <boost/math/tools/roots.hpp>

namespace boost::math::tools {

// Solves ax^3 + bx^2 + cx + d = 0.
// Only returns the real roots, as types get weird for real coefficients and
// complex roots. Follows Numerical Recipes, Chapter 5, section 6. NB: A better
// algorithm apparently exists: Algorithm 954: An Accurate and Efficient Cubic
// and Quartic Equation Solver for Physical Applications However, I don't have
// access to that paper!
template <typename Real>
std::array<Real, 3> cubic_roots(Real a, Real b, Real c, Real d) {
    using std::abs;
    using std::acos;
    using std::cbrt;
    using std::cos;
    using std::fma;
    using std::sqrt;
    std::array<Real, 3> roots = {std::numeric_limits<Real>::quiet_NaN(),
                                 std::numeric_limits<Real>::quiet_NaN(),
                                 std::numeric_limits<Real>::quiet_NaN()};
    if (a == 0) {
        // bx^2 + cx + d = 0:
        if (b == 0) {
            // cx + d = 0:
            if (c == 0) {
                if (d != 0) {
                    // No solutions:
                    return roots;
                }
                roots[0] = 0;
                roots[1] = 0;
                roots[2] = 0;
                return roots;
            }
            roots[0] = -d / c;
            return roots;
        }
        auto [x0, x1] = quadratic_roots(b, c, d);
        roots[0] = x0;
        roots[1] = x1;
        return roots;
    }
    if (d == 0) {
        auto [x0, x1] = quadratic_roots(a, b, c);
        roots[0] = x0;
        roots[1] = x1;
        roots[2] = 0;
        std::sort(roots.begin(), roots.end());
        return roots;
    }
    Real p = b / a;
    Real q = c / a;
    Real r = d / a;
    Real Q = (p * p - 3 * q) / 9;
    Real R = (2 * p * p * p - 9 * p * q + 27 * r) / 54;
    if (R * R < Q * Q * Q) {
        Real rtQ = sqrt(Q);
        Real theta = acos(R / (Q * rtQ)) / 3;
        Real st = sin(theta);
        Real ct = cos(theta);
        roots[0] = -2 * rtQ * ct - p / 3;
        roots[1] = -rtQ * (-ct + sqrt(Real(3)) * st) - p / 3;
        roots[2] = rtQ * (ct + sqrt(Real(3)) * st) - p / 3;
    } else {
        // In Numerical Recipes, Chapter 5, Section 6, it is claimed that we
        // only have one real root if R^2 >= Q^3. But this isn't true; we can
        // even see this from equation 5.6.18. The condition for having three
        // real roots is that A = B. It *is* the case that if we're in this
        // branch, and we have 3 real roots, two are a double root. Take
        // (x+1)^2(x-2) = x^3 - 3x -2 as an example. This clearly has a double
        // root at x = -1, and it gets sent into this branch.
        Real arg = R * R - Q * Q * Q;
        Real A = (R >= 0 ? -1 : 1) * cbrt(abs(R) + sqrt(arg));
        Real B = 0;
        if (A != 0) {
            B = Q / A;
        }
        roots[0] = A + B - p / 3;
        // Yes, we're comparing floats for equality:
        // Any perturbation pushes the roots into the complex plane; out of the
        // bailiwick of this routine.
        if (A == B || arg == 0) {
            roots[1] = -A - p / 3;
            roots[2] = -A - p / 3;
        }
    }
    // Root polishing:
    for (auto &r : roots) {
        // Horner's method.
        // Here I'll take John Gustaffson's opinion that the fma is a *distinct*
        // operation from a*x +b: Make sure to compile these fmas into a single
        // instruction and not a function call! (I'm looking at you Windows.)
        Real f = fma(a, r, b);
        f = fma(f, r, c);
        f = fma(f, r, d);
        Real df = fma(3 * a, r, 2 * b);
        df = fma(df, r, c);
        if (df != 0) {
            Real d2f = fma(6 * a, r, 2 * b);
            Real denom = 2 * df * df - f * d2f;
            if (denom != 0) {
                r -= 2 * f * df / denom;
            } else {
                r -= f / df;
            }
        }
    }
    std::sort(roots.begin(), roots.end());
    return roots;
}

// Computes the empirical residual p(r) (first element) and expected residual
// eps*|rp'(r)| (second element) for a root. Recall that for a numerically
// computed root r satisfying r = r_0(1+eps) of a function p, |p(r)| <=
// eps|rp'(r)|.
template <typename Real>
std::array<Real, 2> cubic_root_residual(Real a, Real b, Real c, Real d,
                                        Real root) {
    using std::abs;
    using std::fma;
    std::array<Real, 2> out;
    Real residual = fma(a, root, b);
    residual = fma(residual, root, c);
    residual = fma(residual, root, d);

    out[0] = residual;

    // The expected residual is:
    // eps*[4|ar^3| + 3|br^2| + 2|cr| + |d|]
    // This can be demonstrated by assuming the coefficients and the root are
    // perturbed according to the rounding model of floating point arithmetic,
    // and then working through the inequalities.
    root = abs(root);
    Real expected_residual = fma(4 * abs(a), root, 3 * abs(b));
    expected_residual = fma(expected_residual, root, 2 * abs(c));
    expected_residual = fma(expected_residual, root, abs(d));
    out[1] = expected_residual * std::numeric_limits<Real>::epsilon();
    return out;
}

// Computes the condition number of rootfinding. This is defined in Corless, A
// Graduate Introduction to Numerical Methods, Section 3.2.1.
template <typename Real>
Real cubic_root_condition_number(Real a, Real b, Real c, Real d, Real root) {
    using std::abs;
    using std::fma;
    // There are *absolute* condition numbers that can be defined when r = 0;
    // but they basically reduce to the residual computed above.
    if (root == static_cast<Real>(0)) {
        return std::numeric_limits<Real>::infinity();
    }

    Real numerator = fma(abs(a), abs(root), abs(b));
    numerator = fma(numerator, abs(root), abs(c));
    numerator = fma(numerator, abs(root), abs(d));
    Real denominator = fma(3 * a, root, 2 * b);
    denominator = fma(denominator, root, c);
    if (denominator == static_cast<Real>(0)) {
        return std::numeric_limits<Real>::infinity();
    }
    denominator *= root;
    return numerator / abs(denominator);
}

} // namespace boost::math::tools
#endif

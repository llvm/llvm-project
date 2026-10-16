//  (C) Copyright Nick Thompson 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_MATH_TOOLS_QUARTIC_ROOTS_HPP
#define BOOST_MATH_TOOLS_QUARTIC_ROOTS_HPP
#include <array>
#include <cmath>
#include <boost/math/tools/cubic_roots.hpp>

namespace boost::math::tools {

namespace detail {

// Make sure the nans are always at the back of the array:
template<typename Real>
bool comparator(Real r1, Real r2) {
   using std::isnan;
   if (isnan(r1)) { return false; }
   if (isnan(r2)) { return true; }
   return r1 < r2;
}

template<typename Real>
std::array<Real, 4> polish_and_sort(Real a, Real b, Real c, Real d, Real e, std::array<Real, 4>& roots) {
    // Polish the roots with a Halley iterate.
    using std::fma;
    using std::abs;
    for (auto &r : roots) {
        Real df = fma(4*a, r, 3*b);
        df = fma(df, r, 2*c);
        df = fma(df, r, d);
        Real d2f = fma(12*a, r, 6*b);
        d2f = fma(d2f, r, 2*c);
        Real f = fma(a, r, b);
        f = fma(f,r,c);
        f = fma(f,r,d);
        f = fma(f,r,e);
        Real denom = 2*df*df - f*d2f;
        if (abs(denom) > (std::numeric_limits<Real>::min)())
        {
            r -= 2*f*df/denom;
        }
    }
    std::sort(roots.begin(), roots.end(), detail::comparator<Real>);
    return roots;
}

}
// Solves ax^4 + bx^3 + cx^2 + dx + e = 0.
// Only returns the real roots, as these are the only roots of interest in ray intersection problems.
// Follows Graphics Gems V: https://github.com/erich666/GraphicsGems/blob/master/gems/Roots3And4.c
template<typename Real>
std::array<Real, 4> quartic_roots(Real a, Real b, Real c, Real d, Real e) {
    using std::abs;
    using std::sqrt;
    auto nan = std::numeric_limits<Real>::quiet_NaN();
    std::array<Real, 4> roots{nan, nan, nan, nan};
    if (abs(a) <= (std::numeric_limits<Real>::min)()) {
        auto cbrts = cubic_roots(b, c, d, e);
        roots[0] = cbrts[0];
        roots[1] = cbrts[1];
        roots[2] = cbrts[2];
        if (b == 0 && c == 0 && d == 0 && e == 0) {
           roots[3] = 0;
        }
        return detail::polish_and_sort(a, b, c, d, e, roots);
    }
    if (abs(e) <= (std::numeric_limits<Real>::min)()) {
        auto v = cubic_roots(a, b, c, d);
        roots[0] = v[0];
        roots[1] = v[1];
        roots[2] = v[2];
        roots[3] = 0;
        return detail::polish_and_sort(a, b, c, d, e, roots);
    }
    // Now solve x^4 + Ax^3 + Bx^2 + Cx + D = 0.
    Real A = b/a;
    Real B = c/a;
    Real C = d/a;
    Real D = e/a;
    Real Asq = A*A;
    // Let x = y - A/4:
    // Mathematica: Expand[(y - A/4)^4 + A*(y - A/4)^3 + B*(y - A/4)^2 + C*(y - A/4) + D]
    // We now solve the depressed quartic y^4 + py^2 + qy + r = 0.
    Real p = B - 3*Asq/8;
    Real q = C - A*B/2 + Asq*A/8;
    Real r = D - A*C/4 + Asq*B/16 - 3*Asq*Asq/256;
    if (abs(r) <= (std::numeric_limits<Real>::min)()) {
        auto [r1, r2, r3] = cubic_roots(Real(1), Real(0), p, q);
        r1 -= A/4;
        r2 -= A/4;
        r3 -= A/4;
        roots[0] = r1;
        roots[1] = r2;
        roots[2] = r3;
        roots[3] = -A/4;
        return detail::polish_and_sort(a, b, c, d, e, roots);
    }
    // Biquadratic case:
    if (abs(q) <= (std::numeric_limits<Real>::min)()) {
        auto [r1, r2] = quadratic_roots(Real(1), p, r);
        if (r1 >= 0) {
           Real rtr = sqrt(r1);
           roots[0] = rtr - A/4;
           roots[1] = -rtr - A/4;
        }
        if (r2 >= 0) {
           Real rtr = sqrt(r2);
           roots[2] = rtr - A/4;
           roots[3] = -rtr - A/4;
        }
        return detail::polish_and_sort(a, b, c, d, e, roots);
    }

    // Now split the depressed quartic into two quadratics:
    // y^4 + py^2 + qy + r = (y^2 + sy + u)(y^2 - sy + v) = y^4 + (v+u-s^2)y^2 + s(v - u)y + uv
    // So p = v+u-s^2, q = s(v - u), r = uv.
    // Then (v+u)^2 - (v-u)^2 = 4uv = 4r = (p+s^2)^2 - q^2/s^2.
    // Multiply through by s^2 to get s^2(p+s^2)^2 - q^2 - 4rs^2 = 0, which is a cubic in s^2.
    // Then we let z = s^2, to get
    // z^3 + 2pz^2 + (p^2 - 4r)z - q^2 = 0.
    auto z_roots = cubic_roots(Real(1), 2*p, p*p - 4*r, -q*q);
    // z = s^2, so s = sqrt(z).
    // Hence we require a root > 0, and for the sake of sanity we should take the largest one:
    Real largest_root = std::numeric_limits<Real>::lowest();
    for (auto z : z_roots) {
        if (z > largest_root) {
            largest_root = z;
        }
    }
    // No real roots:
    if (largest_root <= 0) {
      return roots;
    }
    Real s = sqrt(largest_root);
    // s is nonzero, because we took care of the biquadratic case.
    Real v = (p + largest_root + q/s)/2;
    Real u = v - q/s;
    // Now solve y^2 + sy + u = 0:
    auto [root0, root1] = quadratic_roots(Real(1), s, u);

    // Now solve y^2 - sy + v = 0:
    auto [root2, root3] = quadratic_roots(Real(1), -s, v);
    roots[0] = root0;
    roots[1] = root1;
    roots[2] = root2;
    roots[3] = root3;

    for (auto& r : roots) {
        r -= A/4;
    }
    return detail::polish_and_sort(a, b, c, d, e, roots);
}

}
#endif

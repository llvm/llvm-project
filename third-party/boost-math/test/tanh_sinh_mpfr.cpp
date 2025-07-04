// Copyright Nick Thompson, 2020
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)
#include "math_unit_test.hpp"
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/mpfr.hpp>

using std::log;
using std::sin;
using std::abs;
using boost::math::quadrature::tanh_sinh;
using boost::multiprecision::mpfr_float;
using boost::math::constants::pi;
using boost::math::constants::zeta_three;

#ifdef BOOST_MATH_RUN_MP_TESTS

int main() {
    using Real = mpfr_float;
    int p = 100;
    mpfr_float::default_precision(p);
    auto f = [](Real x)->Real { return x*log(sin(x)); };
    auto integrator = tanh_sinh<mpfr_float>();
    Real Q = integrator.integrate(f, Real(0), pi<Real>()/2);
    // Sanity check: NIntegrate[x*Log[Sin[x]],{x,0,Pi/2}] = -0.329236
    Real expected = (7*zeta_three<Real>() - pi<Real>()*pi<Real>()*log(static_cast<Real>(4)))/16;
    CHECK_ULP_CLOSE(expected, Q, 3);
    return boost::math::test::report_errors();
}

#else

int main()
{
    return 0;
}

#endif

/*
 * Copyright Nick Thompson, 2020
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "math_unit_test.hpp"
#include <numeric>
#include <utility>
#include <iomanip>
#include <iostream>
#include <random>
#include <cmath>
#include <boost/assert.hpp>
#include <boost/core/demangle.hpp>
#include <boost/hana/for_each.hpp>
#include <boost/hana/ext/std/integer_sequence.hpp>
#include <boost/math/tools/condition_numbers.hpp>
#include <boost/math/special_functions/daubechies_wavelet.hpp>
#include <boost/math/special_functions/next.hpp>
#include <boost/math/quadrature/trapezoidal.hpp>

#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
using boost::multiprecision::float128;
#endif


using boost::math::constants::pi;
using boost::math::constants::root_two;

template<typename Real>
void test_exact_value()
{
    // The global phase of the wavelet is not constrained by anything other than convention.
    // Make sure that our conventions match the rest of the world:
    auto psi = boost::math::daubechies_wavelet<Real, 2>(2);
    Real computed = psi(1);
    Real expected = -1.366025403784439;
    CHECK_MOLLIFIED_CLOSE(expected, computed, 0.0001);
}

template<typename Real, int p>
void test_quadratures()
{
    std::cout << "Testing quadratures of " << p << " vanishing moment Daubechies wavelet on type " << boost::core::demangle(typeid(Real).name()) << "\n";
    using boost::math::quadrature::trapezoidal;
    auto psi = boost::math::daubechies_wavelet<Real, p>();
    std::cout << "Wavelet functor size is " << psi.bytes() << " bytes" << std::endl;
        
    Real tol = std::numeric_limits<Real>::epsilon();
    Real error_estimate = std::numeric_limits<Real>::quiet_NaN();
    Real L1 = std::numeric_limits<Real>::quiet_NaN();
    auto [a, b] = psi.support();
    CHECK_ULP_CLOSE(Real(-p+1), a, 0);
    CHECK_ULP_CLOSE(Real(p), b, 0);
    // A wavelet is a function of zero average; ensure the quadrature over its support is zero.
    Real Q = trapezoidal(psi, a, b, tol, 15, &error_estimate, &L1);
    if (!CHECK_MOLLIFIED_CLOSE(Real(0), Q, Real(0.0001)))
    {
        std::cerr << "  Quadrature of " << p << " vanishing moment wavelet does not vanish.\n";
        std::cerr << "  Error estimate: " << error_estimate << ", L1 norm: " << L1 << "\n";
    }
    auto psi_sq = [psi](Real x) {
        Real t = psi(x);
        return t*t;
    };
    Q = trapezoidal(psi_sq, a, b, tol, 15, &error_estimate, &L1);
    Real quad_tol = 2000*std::sqrt(std::numeric_limits<Real>::epsilon())/(p*p*p);
    if (!CHECK_MOLLIFIED_CLOSE(Real(1), Q, quad_tol))
    {
        std::cerr << "  L2 norm of " << p << " vanishing moment wavelet does not vanish.\n";
        std::cerr << "  Error estimate: " << error_estimate << ", L1 norm: " << L1 << "\n";
    }
    // psi is orthogonal to its integer translates: \int \psi(x-k) \psi(x) \, \mathrm{d}x = 0
    // g_n = 1/sqrt(2) <psi(t/2), phi(t-n)> (Mallat, 7.55)

    // Now hit the boundary. Much can go wrong here; this just tests for segfaults:
    int samples = 500;
    Real xlo = a;
    Real xhi = b;
    for (int i = 0; i < samples; ++i)
    {
        CHECK_ULP_CLOSE(Real(0), psi(xlo), 0);
        CHECK_ULP_CLOSE(Real(0), psi(xhi), 0);
        if constexpr (p > 2)
        {
            CHECK_ULP_CLOSE(Real(0), psi.prime(xlo), 0);
            CHECK_ULP_CLOSE(Real(0), psi.prime(xhi), 0);
            if constexpr (p >= 6) {
                CHECK_ULP_CLOSE(Real(0), psi.double_prime(xlo), 0);
                CHECK_ULP_CLOSE(Real(0), psi.double_prime(xhi), 0);
            }
        }
        xlo = std::nextafter(xlo, std::numeric_limits<Real>::lowest());
        xhi = std::nextafter(xhi, (std::numeric_limits<Real>::max)());
    }

    xlo = a;
    xhi = b;
    for (int i = 0; i < samples; ++i) {
        std::cout << std::setprecision(std::numeric_limits<Real>::max_digits10);
        BOOST_ASSERT(abs(psi(xlo)) <= 5);
        BOOST_ASSERT(abs(psi(xhi)) <= 5);
        if constexpr (p > 2)
        {
            BOOST_ASSERT(abs(psi.prime(xlo)) <= 5);
            BOOST_ASSERT(abs(psi.prime(xhi)) <= 5);
            if constexpr (p >= 6)
            {
                BOOST_ASSERT(abs(psi.double_prime(xlo)) <= 5);
                BOOST_ASSERT(abs(psi.double_prime(xhi)) <= 5);
            }
        }
        xlo = std::nextafter(xlo, (std::numeric_limits<Real>::max)());
        xhi = std::nextafter(xhi, std::numeric_limits<Real>::lowest());
    }
}

int main()
{
    #ifndef __MINGW32__
    try
    {
      test_exact_value<double>();

      boost::hana::for_each(std::make_index_sequence<17>(), [&](auto i) {
         test_quadratures<float, i + 3>();
         test_quadratures<double, i + 3>();
         });
    }
    catch (std::bad_alloc)
    {
       // not much we can do about this, this test uses lots of memory!
    }
    #endif
    return boost::math::test::report_errors();
}

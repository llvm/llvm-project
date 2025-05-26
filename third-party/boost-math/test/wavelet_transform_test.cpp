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
#include <cmath>
#include <cfloat>
#include <boost/core/demangle.hpp>
#include <boost/hana/for_each.hpp>
#include <boost/hana/ext/std/integer_sequence.hpp>
#include <boost/math/quadrature/wavelet_transforms.hpp>
#include <boost/math/tools/minima.hpp>
#include <boost/math/quadrature/trapezoidal.hpp>

#if __has_include(<stdfloat>)
#  include <stdfloat>
#endif

#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
using boost::multiprecision::float128;
#endif


using boost::math::constants::pi;
using boost::math::constants::root_two;
using boost::math::quadrature::daubechies_wavelet_transform;
using boost::math::quadrature::trapezoidal;

template<typename Real, int p>
void test_wavelet_transform()
{
    using std::abs;
    std::cout << "Testing wavelet transform of " << p << " vanishing moment Daubechies wavelet on type " << boost::core::demangle(typeid(Real).name()) << "\n";
    auto psi = boost::math::daubechies_wavelet<Real, p>();

    auto abs_psi = [&psi](Real x) {
        return abs(psi(x));
    };
    auto [a, b] = psi.support();
    auto psil1 = trapezoidal(abs_psi, a, b, 100*std::numeric_limits<Real>::epsilon());
    Real psi_sup_norm = 0;
    for (double x = a; x < b; x += 0.00001)
    {
        Real y = psi(x);
        if (std::abs(y) > psi_sup_norm)
        {
            psi_sup_norm = std::abs(y);
        }
    }
    // An even function:
    auto f = [](Real x) {
        return std::exp(-abs(x));
    };
    Real fmax = 1;
    Real fl2 = 1;
    Real fl1 = 2;

    auto Wf = daubechies_wavelet_transform(f, psi);
    for (double s = 0; s < 10; s += 0.01)
    {
        Real w1 = Wf(s, 0.0);
        Real w2 = Wf(-s, 0.0);
        // Since f is an even function, we get w1 = w2:
        CHECK_ULP_CLOSE(w1, w2, 12);
    }

    // The wavelet transform with respect to Daubechies wavelets 
    for (double s = -10; s < 10; s += 0.1)
    {
        for (double t = -10; t < 10; t+= 0.1)
        {
            Real w = Wf(s, t);
            // Integral inequality:
            Real r1 = sqrt(abs(s))*fmax*psil1;
            if (!CHECK_LE(abs(w), r1))
            {
                std::cerr << "  Integral inequality |W[f](s,t)| <= ||f||_infty ||psi||_1 is violated.\n";
            }
            if (!CHECK_LE(abs(w), fl2))
            {
                std::cerr << "  Integral inequality | int f psi_s,t| <= ||f||_2 ||psi||_2 violated.\n";
            }
            Real r4 = sqrt(abs(s))*fl1*psi_sup_norm;
            if (!CHECK_LE(abs(w), r4))
            {
                std::cerr << "  Integral inequality |W[f](s,t)| <= sqrt(|s|)||f||_1 ||psi||_infty is violated.\n";
            }
            Real r5 = sqrt(abs(s))*fmax*psil1;
            if (!CHECK_LE(abs(w), r5))
            {
                std::cerr << "  Integral inequality |W[f](s,t)| <= sqrt(|s|)||f||_infty ||psi||_1 is violated.\n";
            }
            if (s != 0)
            {
                Real r2 = fl1*psi_sup_norm/sqrt(abs(s));
                if(!CHECK_LE(abs(w), r2))
                {
                    std::cerr << "  Integral inequality |W[f](s,t)| <= ||f||_1 ||psi||_infty/sqrt(|s|) is violated.\n";
                }
            }

        }
    }

    if (p > 5)
    {
        // Wavelet transform of a constant is zero.
        // The quadrature sum is horribly ill-conditioned (technically infinite),
        // so we'll only test on the more rapidly converging sums.
        auto g = [](Real ) { return Real(7); };
        auto Wg = daubechies_wavelet_transform(g, psi);
        for (double s = -10; s < 10; s += 0.1)
        {
            for (double t = -10; t < 10; t+= 0.1)
            {
                Real w = Wg(s, t);
                if (!CHECK_LE(abs(w), Real(10*sqrt(std::numeric_limits<Real>::epsilon()))))
                {
                    std::cerr << "  Wavelet transform of constant with respect to " << p << " vanishing moment Daubechies wavelet is insufficiently small\n";
                }

            }
        }
        // Wavelet transform of psi evaluated at s = 1, t = 0 is L2 norm of psi:
        auto Wpsi = daubechies_wavelet_transform(psi, psi);
        CHECK_MOLLIFIED_CLOSE(Real(1), Wpsi(1,0), Real(2*sqrt(std::numeric_limits<Real>::epsilon())));
    }

}

int main()
{
    try{
       #ifdef __STDCPP_FLOAT64_T__
       test_wavelet_transform<std::float64_t, 2>();
       test_wavelet_transform<std::float64_t, 8>();
       test_wavelet_transform<std::float64_t, 16>();
       #else
       test_wavelet_transform<double, 2>();
       test_wavelet_transform<double, 8>();
       test_wavelet_transform<double, 16>();
       #endif
       // All these tests pass, but the compilation takes too long on CI:
       //boost::hana::for_each(std::make_index_sequence<17>(), [&](auto i) {
       //    test_wavelet_transform<double, i+3>();
       //});
    }
    catch (std::bad_alloc const & e)
    {
        std::cerr << "Ran out of memory in wavelet transform test: " << e.what() << "\n";
       // not much we can do about this, this test uses lots of memory!
    }

    return boost::math::test::report_errors();
}

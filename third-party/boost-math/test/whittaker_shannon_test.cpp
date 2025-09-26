/*
 * Copyright Nick Thompson, 2019
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "math_unit_test.hpp"
#include <numeric>
#include <utility>
#include <random>
#include <boost/core/demangle.hpp>
#include <boost/math/interpolators/whittaker_shannon.hpp>

#if __has_include(<stdfloat>)
#  include <stdfloat>
#endif

using boost::math::interpolators::whittaker_shannon;

template<class Real>
void test_trivial()
{
    Real t0 = 0;
    Real h = Real(1)/Real(16);
    std::vector<Real> v{Real(1.5)};
    std::vector<Real> v_copy = v;
    auto ws = whittaker_shannon<decltype(v)>(std::move(v), t0, h);


    Real expected = 0;
    if(!CHECK_MOLLIFIED_CLOSE(expected, ws.prime(0), 10*std::numeric_limits<Real>::epsilon())) {
        std::cerr << "  Problem occurred at abscissa " << 0 << "\n";
    }

    expected = -v_copy[0]/h;
    if(!CHECK_MOLLIFIED_CLOSE(expected, ws.prime(h), 10*std::numeric_limits<Real>::epsilon())) {
        std::cerr << "  Problem occurred at abscissa " << 0 << "\n";
    }
}

template<class Real>
void test_knots()
{
    Real t0 = 0;
    Real h = Real(1)/Real(16);
    size_t n = 512;
    std::vector<Real> v(n);
    std::mt19937 gen(323723);
    std::uniform_real_distribution<Real> dis(Real(1.0), Real(2.0));

    for(size_t i = 0;  i < n; ++i) {
      v[i] = static_cast<Real>(dis(gen));
    }
    auto ws = whittaker_shannon<decltype(v)>(std::move(v), t0, h);

    size_t i = 0;
    while (i < n) {
      Real t = t0 + i*h;
      Real expected = ws[i];
      Real computed = ws(t);
      CHECK_ULP_CLOSE(expected, computed, 16);
      ++i;
    }
}

template<class Real>
void test_bump()
{
    using std::exp;
    using std::abs;
    using std::sqrt;
    auto bump = [](Real x) { using std::exp; using std::abs; if (abs(x) >= 1) { return Real(0); } return exp(-Real(1)/(Real(1)-x*x)); };

    auto bump_prime = [&bump](Real x) { Real z = 1-x*x; return -2*x*bump(x)/(z*z); };

    Real t0 = -1;
    size_t n = 2049;
    Real h = Real(2)/Real(n-1);

    std::vector<Real> v(n);
    for(size_t i = 0; i < n; ++i) {
        Real t = t0 + i*h;
        v[i] = bump(t);
    }


    std::vector<Real> v_copy = v;
    auto ws = whittaker_shannon<decltype(v)>(std::move(v), t0, h);

    // Test the knots:
    for(size_t i = v_copy.size()/4; i < 3*v_copy.size()/4; ++i) {
        Real t = t0 + i*h;
        Real expected = v_copy[i];
        Real computed = ws(t);
        if(!CHECK_MOLLIFIED_CLOSE(expected, computed, 10*std::numeric_limits<Real>::epsilon())) {
            std::cerr << "  Problem occurred at abscissa " << t << "\n";
        }

        Real expected_prime = bump_prime(t);
        Real computed_prime = ws.prime(t);
        if(!CHECK_MOLLIFIED_CLOSE(expected_prime, computed_prime, 1000*std::numeric_limits<Real>::epsilon())) {
            std::cerr << "  Problem occurred at abscissa " << t << "\n";
        }

    }

    std::mt19937 gen(323723);
    std::uniform_real_distribution<long double> dis(-0.85, 0.85);

    size_t i = 0;
    while (i++ < 1000)
    {
        Real t = static_cast<Real>(dis(gen));
        Real expected = bump(t);
        Real computed = ws(t);
        if(!CHECK_MOLLIFIED_CLOSE(expected, computed, 10*std::numeric_limits<Real>::epsilon())) {
            std::cerr << "  Problem occurred at abscissa " << t << "\n";
        }

        Real expected_prime = bump_prime(t);
        Real computed_prime = ws.prime(t);
        if(!CHECK_MOLLIFIED_CLOSE(expected_prime, computed_prime, sqrt(std::numeric_limits<Real>::epsilon()))) {
            std::cerr << "  Problem occurred at abscissa " << t << "\n";
        }
    }
}


int main()
{
    #ifdef __STDCPP_FLOAT32_T__
    test_trivial<std::float32_t>();
    test_knots<std::float32_t>();
    #else
    test_trivial<float>();
    test_knots<float>();
    #endif
    
    #ifdef __STDCPP_FLOAT64_T__
    test_trivial<std::float64_t>();
    test_knots<std::float64_t>();
    test_bump<std::float64_t>();
    #else
    test_trivial<double>();
    test_knots<double>();
    test_bump<double>();
    #endif


#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test_knots<long double>();
#if LDBL_MANT_DIG <= 64
    // Anything more precise than this fails for unknown reasons
    test_bump<long double>();
#endif
#endif

    return boost::math::test::report_errors();
}

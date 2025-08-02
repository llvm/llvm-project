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
#include <cmath>
#include <boost/core/demangle.hpp>
#include <boost/math/special_functions/gegenbauer.hpp>
#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
using boost::multiprecision::float128;
#endif

#if __has_include(<stdfloat>)
#  include <stdfloat>
#endif

using std::abs;
using boost::math::gegenbauer;
using boost::math::gegenbauer_derivative;

template<class Real>
void test_parity()
{
    std::mt19937 gen(323723);
    std::uniform_real_distribution<Real> xdis(-1, +1);
    std::uniform_real_distribution<Real> lambdadis(Real(-0.5), 1);

    for(unsigned n = 0; n < 50; ++n) {
        unsigned calls = 50;
        unsigned i = 0;
        while(i++ < calls) {
            Real x = xdis(gen);
            Real lambda = lambdadis(gen);
            if (n & 1) {
                CHECK_ULP_CLOSE(gegenbauer(n, lambda, -x), -gegenbauer(n, lambda, x), 0);
            }
            else {
                CHECK_ULP_CLOSE(gegenbauer(n, lambda, -x), gegenbauer(n, lambda, x), 0);
            }
        }
    }
}

template<class Real>
void test_quadratic()
{
    Real lambda = 1/Real(4);
    auto c2 = [&](Real x) { return -lambda + 2*lambda*(1+lambda)*x*x; };

    Real x = -1;
    Real h = 1/Real(256);
    while (x < 1) {
        Real expected = c2(x);
        Real computed = gegenbauer(2, lambda, x);
        CHECK_ULP_CLOSE(expected, computed, 0);
        x += h;
    }
}

template<class Real>
void test_cubic()
{
    Real lambda = 1/Real(4);
    auto c3 = [&](Real x) { return lambda*(1+lambda)*x*(-2 + 4*(2+lambda)*x*x/3); };

    Real x = -1;
    Real h = 1/Real(256);
    while (x < 1) {
        Real expected = c3(x);
        Real computed = gegenbauer(3, lambda, x);
        CHECK_ULP_CLOSE(expected, computed, 4);
        x += h;
    }
}

template<class Real>
void test_derivative()
{
    Real lambda = Real(0.5);
    auto c3_prime = [&](Real x) { return 2*lambda*(lambda+1)*(-1 + 2*(lambda+2)*x*x); };
    auto c3_double_prime = [&](Real x) { return 8*lambda*(lambda+1)*(lambda+2)*x; };
    Real x = -1;
    Real h = 1/Real(256);
    while (x < 1) {
        Real expected = c3_prime(x);
        Real computed = gegenbauer_derivative(3, lambda, x, 1);
        CHECK_ULP_CLOSE(expected, computed, 1);

        expected = c3_double_prime(x);
        computed = gegenbauer_derivative(3, lambda, x, 2);
        CHECK_ULP_CLOSE(expected, computed, 1);

        x += h;
    }

}



int main()
{
    #ifdef __STDCPP_FLOAT32_T__

    test_parity<std::float32_t>();
    test_quadratic<std::float32_t>();
    test_derivative<std::float32_t>();

    #else
    
    test_parity<float>();
    test_quadratic<float>();
    test_derivative<float>();

    #endif

    #ifdef __STDCPP_FLOAT64_T__

    test_parity<std::float64_t>();
    test_quadratic<std::float64_t>();
    test_cubic<std::float64_t>();
    test_derivative<std::float64_t>();

    #else

    test_parity<double>();
    test_quadratic<double>();
    test_cubic<double>();
    test_derivative<double>();

    #endif

    #ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    
    test_parity<long double>();
    test_quadratic<long double>();
    test_cubic<long double>();
    test_derivative<long double>();
    
    #endif

    #ifdef BOOST_HAS_FLOAT128
    
    test_quadratic<boost::multiprecision::float128>();
    test_cubic<boost::multiprecision::float128>();
    
    #endif

    return boost::math::test::report_errors();
}

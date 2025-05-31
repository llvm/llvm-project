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
#include <boost/math/special_functions/jacobi.hpp>
#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
using boost::multiprecision::float128;
#endif

#if __has_include(<stdfloat>)
#  include <stdfloat>
#endif

using std::abs;
using boost::math::jacobi;
using boost::math::jacobi_derivative;

template<typename Real>
void test_to_quadratic()
{
    Real h = 1/Real(8);
    for (Real alpha = -1 + h; alpha < 2; alpha += h) {
        for (Real beta = -1 + h; beta < 2; beta += h) {
            for (Real x = -1; x < 1; x += h) {
                Real expected = 1;
                Real computed = jacobi(0, alpha, beta, x);
                CHECK_ULP_CLOSE(expected, computed, 0);

                expected = (alpha + 1) + (alpha + beta +2)*(x-1)/2;
                computed = jacobi(1, alpha, beta, x);
                CHECK_ULP_CLOSE(expected, computed, 0);

                expected = (alpha + 1)*(alpha+2)/2 + (alpha + 2)*(alpha + beta + 3)*(x-1)/2 + (alpha + beta + 3)*(alpha + beta + 4)*(x-1)*(x-1)/8;
                computed = jacobi(2, alpha, beta, x);
                CHECK_ULP_CLOSE(expected, computed, 1);

            }
        }
    }
}

template<typename Real>
void test_symmetry()
{
    Real h = 1/Real(4);
    for (Real alpha = -1 + h; alpha < 2; alpha += h) {
        for (Real beta = -1 + h; beta < 2; beta += h) {
            for (Real x = -1; x < 1; x += h) {
                for (size_t n = 0; n < 20; n += 2)
                {
                    Real expected = jacobi(n, beta, alpha , -x);
                    Real computed = jacobi(n, alpha, beta, x);
                    CHECK_ULP_CLOSE(expected, computed, 0);

                    expected = jacobi(n+1, beta, alpha, -x);
                    computed = -jacobi(n+1, alpha, beta, x);
                    CHECK_ULP_CLOSE(expected, computed, 0);
                }
            }
        }
    }
}

template<typename Real>
void test_derivative()
{
    Real h = 1/Real(4);
    for (Real alpha = -1 + h; alpha < 2; alpha += h) {
        for (Real beta = -1 + h; beta < 2; beta += h) {
            for (Real x = -1; x < 1; x += h) {
                Real expected = 0;
                Real computed = jacobi_derivative(0, alpha, beta, x, 1);
                CHECK_ULP_CLOSE(expected, computed, 0);

                expected = (alpha + beta + 2)/2;
                computed = jacobi_derivative(1, alpha, beta, x, 1);
                CHECK_ULP_CLOSE(expected, computed, 0);

                expected = (alpha + 2)*(alpha + beta + 3)/2 + (alpha + beta + 3)*(alpha + beta + 4)*(x-1)/4;
                computed = jacobi_derivative(2, alpha, beta, x, 1);
                CHECK_ULP_CLOSE(expected, computed, 0);

                expected = (alpha + beta + 3)*(alpha + beta + 4)/4;
                computed = jacobi_derivative(2, alpha, beta, x, 2);
                CHECK_ULP_CLOSE(expected, computed, 0);
            }
        }
    }
}

int main()
{
    #ifdef __STDCPP_FLOAT32_T__
    
    test_symmetry<std::float32_t>();
    test_derivative<std::float32_t>();

    #else

    test_symmetry<float>();
    test_derivative<float>();

    #endif
    
    #ifdef __STDCPP_FLOAT64_T__

    test_to_quadratic<std::float64_t>();
    test_symmetry<std::float64_t>();
    test_derivative<std::float64_t>();

    #else

    test_to_quadratic<double>();
    test_symmetry<double>();
    test_derivative<double>();

    #endif

    #ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    
    test_to_quadratic<long double>();
    test_symmetry<long double>();
    test_derivative<long double>();
    
    #endif

    #ifdef BOOST_HAS_FLOAT128
    
    test_to_quadratic<boost::multiprecision::float128>();
    test_symmetry<boost::multiprecision::float128>();
    test_derivative<boost::multiprecision::float128>();
    
    #endif

    return boost::math::test::report_errors();
}

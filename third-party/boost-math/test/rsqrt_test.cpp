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
#include <boost/math/special_functions/rsqrt.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>

#if __has_include(<stdfloat>)
#  include <stdfloat>
#endif

#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
using boost::multiprecision::float128;
#endif

using boost::math::rsqrt;

template<typename Real>
void test_rsqrt()
{
    std::cout << "Testing rsqrt on type " << boost::core::demangle(typeid(Real).name()) << "\n";
    using std::sqrt;
    Real x = (std::numeric_limits<Real>::min)();
    while (x < 10000*std::numeric_limits<Real>::epsilon()) {
        Real expected = Real(1)/sqrt(x);
        Real computed = rsqrt(x);
        if(!CHECK_ULP_CLOSE(expected, computed, 2)) {
            std::cerr << "  1/sqrt(" << x << ") is computed incorrectly.\n";
        }
        x += std::numeric_limits<Real>::epsilon();
    }

    // x ~ 1:
    x = 1;
    while (x < 1 + 1000*std::numeric_limits<Real>::epsilon()) {
        Real expected = Real(1)/sqrt(x);
        Real computed = rsqrt(x);
        if(!CHECK_ULP_CLOSE(expected, computed, 2)) {
            std::cerr << "  1/sqrt(" << x << ") is computed incorrectly.\n";
        }
        x += std::numeric_limits<Real>::epsilon();
    }

    // x ~ 1000:
    x = 1000;
    while (x < 1000 + 1000*1000*std::numeric_limits<Real>::epsilon()) {
        Real expected = Real(1)/sqrt(x);
        Real computed = rsqrt(x);
        if(!CHECK_ULP_CLOSE(expected, computed, 2)) {
            std::cerr << "  1/sqrt(" << x << ") is computed incorrectly.\n";
        }
        x += 1000*std::numeric_limits<Real>::epsilon();
    }

    x = std::numeric_limits<Real>::infinity();
    Real expected = Real(1)/sqrt(x);
    Real computed = rsqrt(x);
    if (!CHECK_ULP_CLOSE(expected, computed, 0)) {
        std::cerr << "Reciprocal square root of infinity not correctly computed.\n";
    }

    x = (std::numeric_limits<Real>::max)();
    expected = Real(1)/sqrt(x);
    computed = rsqrt(x);
    if (!CHECK_EQUAL(expected, computed)) {
        std::cerr << "Reciprocal square root of std::numeric_limits<Real>::max() not correctly computed.\n";
    }

    if (!CHECK_NAN(rsqrt(std::numeric_limits<Real>::quiet_NaN()))) {
        std::cerr << "Reciprocal square root of std::numeric_limits<Real>::quiet_NaN() is not a NaN.\n";
    }
}


int main()
{
    #ifdef __STDCPP_FLOAT32_T__
    test_rsqrt<std::float32_t>();
    #else
    test_rsqrt<float>();
    #endif
    
    #ifdef __STDCPP_FLOAT64_T__
    test_rsqrt<std::float64_t>();
    #else
    test_rsqrt<double>();
    #endif
    
    test_rsqrt<long double>();
    test_rsqrt<boost::multiprecision::cpp_bin_float_50>();
    test_rsqrt<boost::multiprecision::cpp_bin_float_100>();

    #ifdef BOOST_HAS_FLOAT128
    test_rsqrt<float128>();
    #endif
    return boost::math::test::report_errors();
}

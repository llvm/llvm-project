/*
 * Copyright Nick Thompson, 2020
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "math_unit_test.hpp"
#include <numeric>
#include <utility>
#include <random>
#include <boost/math/interpolators/makima.hpp>
#include <boost/circular_buffer.hpp>
#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
using boost::multiprecision::float128;
#endif

#if __has_include(<stdfloat>)
#  include <stdfloat>
#endif

using boost::math::interpolators::makima;

template<typename Real>
void test_constant()
{

    std::vector<Real> x{0,1,2,3, 9, 22, 81};
    std::vector<Real> y(x.size());
    for (auto & t : y) {
        t = 7;
    }

    auto x_copy = x;
    auto y_copy = y;
    auto akima = makima(std::move(x_copy), std::move(y_copy));

    for (Real t = x[0]; t <= x.back(); t += Real(0.25)) {
        CHECK_ULP_CLOSE(Real(7), akima(t), 2);
        CHECK_ULP_CLOSE(Real(0), akima.prime(t), 2);
    }

    boost::circular_buffer<Real> x_buf(x.size());
    for (auto & t : x) {
        x_buf.push_back(t);
    }

    boost::circular_buffer<Real> y_buf(x.size());
    for (auto & t : y) {
        y_buf.push_back(t);
    }

    auto circular_akima = makima(std::move(x_buf), std::move(y_buf));

    for (Real t = x[0]; t <= x.back(); t += Real(0.25)) {
        CHECK_ULP_CLOSE(Real(7), circular_akima(t), 2);
        CHECK_ULP_CLOSE(Real(0), akima.prime(t), 2);
    }

    circular_akima.push_back(x.back() + 1, 7);
    CHECK_ULP_CLOSE(Real(0), circular_akima.prime(x.back()+1), 2);

}

template<typename Real>
void test_linear()
{
    std::vector<Real> x{0,1,2,3};
    std::vector<Real> y{0,1,2,3};

    auto x_copy = x;
    auto y_copy = y;
    auto akima = makima(std::move(x_copy), std::move(y_copy));

    CHECK_ULP_CLOSE(y[0], akima(x[0]), 0);
    CHECK_ULP_CLOSE(Real(1)/Real(2), akima(Real(1)/Real(2)), 10);
    CHECK_ULP_CLOSE(y[1], akima(x[1]), 0);
    CHECK_ULP_CLOSE(Real(3)/Real(2), akima(Real(3)/Real(2)), 10);
    CHECK_ULP_CLOSE(y[2], akima(x[2]), 0);
    CHECK_ULP_CLOSE(Real(5)/Real(2), akima(Real(5)/Real(2)), 10);
    CHECK_ULP_CLOSE(y[3], akima(x[3]), 0);

    x.resize(45);
    y.resize(45);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = i;
        y[i] = i;
    }

    x_copy = x;
    y_copy = y;
    akima = makima(std::move(x_copy), std::move(y_copy));
    for (Real t = 0; t < x.back(); t += Real(0.5)) {
        CHECK_ULP_CLOSE(t, akima(t), 0);
        CHECK_ULP_CLOSE(Real(1), akima.prime(t), 0);
    }

    x_copy = x;
    y_copy = y;
    // Test endpoint derivatives:
    akima = makima(std::move(x_copy), std::move(y_copy), Real(1), Real(1));
    for (Real t = 0; t < x.back(); t += Real(0.5)) {
        CHECK_ULP_CLOSE(t, akima(t), 0);
        CHECK_ULP_CLOSE(Real(1), akima.prime(t), 0);
    }


    boost::circular_buffer<Real> x_buf(x.size());
    for (auto & t : x) {
        x_buf.push_back(t);
    }

    boost::circular_buffer<Real> y_buf(x.size());
    for (auto & t : y) {
        y_buf.push_back(t);
    }

    auto circular_akima = makima(std::move(x_buf), std::move(y_buf));

    for (Real t = x[0]; t <= x.back(); t += Real(0.25)) {
        CHECK_ULP_CLOSE(t, circular_akima(t), 2);
        CHECK_ULP_CLOSE(Real(1), circular_akima.prime(t), 2);
    }

    circular_akima.push_back(x.back() + 1, y.back()+1);

    CHECK_ULP_CLOSE(Real(y.back() + 1), circular_akima(Real(x.back()+1)), 2);
    CHECK_ULP_CLOSE(Real(1), circular_akima.prime(Real(x.back()+1)), 2);

}

template<typename Real>
void test_interpolation_condition()
{
    for (size_t n = 4; n < 50; ++n) {
        std::vector<Real> x(n);
        std::vector<Real> y(n);
        std::default_random_engine rd;
        std::uniform_real_distribution<Real> dis(0,1);
        Real x0 = dis(rd);
        x[0] = x0;
        y[0] = dis(rd);
        for (size_t i = 1; i < n; ++i) {
            x[i] = x[i-1] + dis(rd);
            y[i] = dis(rd);
        }

        auto x_copy = x;
        auto y_copy = y;
        auto s = makima(std::move(x_copy), std::move(y_copy));
        //std::cout << "s = " << s << "\n";
        for (size_t i = 0; i < x.size(); ++i) {
            CHECK_ULP_CLOSE(y[i], s(x[i]), 2);
        }

        x_copy = x;
        y_copy = y;
        // The interpolation condition is not affected by the endpoint derivatives, even though these derivatives might be super weird:
        s = makima(std::move(x_copy), std::move(y_copy), Real(0), Real(0));
        //std::cout << "s = " << s << "\n";
        for (size_t i = 0; i < x.size(); ++i) {
            CHECK_ULP_CLOSE(y[i], s(x[i]), 2);
        }

    }
}

int main()
{
#if (__GNUC__ > 7) || defined(_MSC_VER) || defined(__clang__)
    
    #ifdef __STDCPP_FLOAT32_T__
    test_constant<std::float32_t>();
    test_linear<std::float32_t>();
    test_interpolation_condition<std::float32_t>();
    #else
    test_constant<float>();
    test_linear<float>();
    test_interpolation_condition<float>();
    #endif

    #ifdef __STDCPP_FLOAT64_T__
    test_constant<std::float64_t>();
    test_linear<std::float64_t>();
    test_interpolation_condition<std::float64_t>();
    #else
    test_constant<double>();
    test_linear<double>();
    test_interpolation_condition<double>();
    #endif

    test_constant<long double>();
    test_linear<long double>();
    test_interpolation_condition<long double>();

#ifdef BOOST_HAS_FLOAT128
    test_constant<float128>();
    test_linear<float128>();
#endif
#endif
    return boost::math::test::report_errors();
}

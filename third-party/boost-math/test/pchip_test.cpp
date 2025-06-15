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
#include <boost/math/interpolators/pchip.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/assert.hpp>
#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
using boost::multiprecision::float128;
#endif

#if __has_include(<stdfloat>)
#  include <stdfloat>
#endif

using boost::math::interpolators::pchip;

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
    auto pchip_spline = pchip(std::move(x_copy), std::move(y_copy));
    //std::cout << "Constant value pchip spline = " << pchip_spline << "\n";

    for (Real t = x[0]; t <= x.back(); t += Real(0.25)) {
        CHECK_ULP_CLOSE(Real(7), pchip_spline(t), 2);
        CHECK_ULP_CLOSE(Real(0), pchip_spline.prime(t), 2);
    }

    boost::circular_buffer<Real> x_buf(x.size());
    for (auto & t : x) {
        x_buf.push_back(t);
    }

    boost::circular_buffer<Real> y_buf(x.size());
    for (auto & t : y) {
        y_buf.push_back(t);
    }

    auto circular_pchip_spline = pchip(std::move(x_buf), std::move(y_buf));

    for (Real t = x[0]; t <= x.back(); t += Real(0.25)) {
        CHECK_ULP_CLOSE(Real(7), circular_pchip_spline(t), 2);
        CHECK_ULP_CLOSE(Real(0), pchip_spline.prime(t), 2);
    }

    circular_pchip_spline.push_back(x.back() + 1, 7);
    CHECK_ULP_CLOSE(Real(0), circular_pchip_spline.prime(x.back()+1), 2);

}

template<typename Real>
void test_linear()
{
    std::vector<Real> x{0,1,2,3};
    std::vector<Real> y{0,1,2,3};

    auto x_copy = x;
    auto y_copy = y;
    auto pchip_spline = pchip(std::move(x_copy), std::move(y_copy));

    CHECK_ULP_CLOSE(y[0], pchip_spline(x[0]), 0);
    CHECK_ULP_CLOSE(Real(1)/Real(2), pchip_spline(Real(1)/Real(2)), 10);
    CHECK_ULP_CLOSE(y[1], pchip_spline(x[1]), 0);
    CHECK_ULP_CLOSE(Real(3)/Real(2), pchip_spline(Real(3)/Real(2)), 10);
    CHECK_ULP_CLOSE(y[2], pchip_spline(x[2]), 0);
    CHECK_ULP_CLOSE(Real(5)/Real(2), pchip_spline(Real(5)/Real(2)), 10);
    CHECK_ULP_CLOSE(y[3], pchip_spline(x[3]), 0);

    x.resize(45);
    y.resize(45);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = i;
        y[i] = i;
    }

    x_copy = x;
    y_copy = y;
    pchip_spline = pchip(std::move(x_copy), std::move(y_copy));
    for (Real t = 0; t < x.back(); t += Real(0.5)) {
        CHECK_ULP_CLOSE(t, pchip_spline(t), 0);
        CHECK_ULP_CLOSE(Real(1), pchip_spline.prime(t), 0);
    }

    x_copy = x;
    y_copy = y;
    // Test endpoint derivatives:
    pchip_spline = pchip(std::move(x_copy), std::move(y_copy), Real(1), Real(1));
    for (Real t = 0; t < x.back(); t += Real(0.5)) {
        CHECK_ULP_CLOSE(t, pchip_spline(t), 0);
        CHECK_ULP_CLOSE(Real(1), pchip_spline.prime(t), 0);
    }


    boost::circular_buffer<Real> x_buf(x.size());
    for (auto & t : x) {
        x_buf.push_back(t);
    }

    boost::circular_buffer<Real> y_buf(x.size());
    for (auto & t : y) {
        y_buf.push_back(t);
    }

    auto circular_pchip_spline = pchip(std::move(x_buf), std::move(y_buf));

    for (Real t = x[0]; t <= x.back(); t += Real(0.25)) {
        CHECK_ULP_CLOSE(t, circular_pchip_spline(t), 2);
        CHECK_ULP_CLOSE(Real(1), circular_pchip_spline.prime(t), 2);
    }

    circular_pchip_spline.push_back(x.back() + 1, y.back()+1);

    CHECK_ULP_CLOSE(Real(y.back() + 1), circular_pchip_spline(Real(x.back()+1)), 2);
    CHECK_ULP_CLOSE(Real(1), circular_pchip_spline.prime(Real(x.back()+1)), 2);

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
        auto s = pchip(std::move(x_copy), std::move(y_copy));
        //std::cout << "s = " << s << "\n";
        for (size_t i = 0; i < x.size(); ++i) {
            CHECK_ULP_CLOSE(y[i], s(x[i]), 2);
        }

        x_copy = x;
        y_copy = y;
        // The interpolation condition is not affected by the endpoint derivatives, even though these derivatives might be super weird:
        s = pchip(std::move(x_copy), std::move(y_copy), Real(0), Real(0));
        //std::cout << "s = " << s << "\n";
        for (size_t i = 0; i < x.size(); ++i) {
            CHECK_ULP_CLOSE(y[i], s(x[i]), 2);
        }

    }
}

template<typename Real>
void test_monotonicity()
{
    for (size_t n = 4; n < 50; ++n) {
        std::vector<Real> x(n);
        std::vector<Real> y(n);
        std::default_random_engine rd;
        std::uniform_real_distribution<Real> dis(0,1);
        Real x0 = dis(rd);
        x[0] = x0;
        y[0] = dis(rd);
        // Monotone increasing:
        for (size_t i = 1; i < n; ++i) {
            x[i] = x[i-1] + dis(rd);
            y[i] = y[i-1] + dis(rd);
        }

        auto x_copy = x;
        auto y_copy = y;
        auto s = pchip(std::move(x_copy), std::move(y_copy));
        //std::cout << "s = " << s << "\n";
        for (size_t i = 0; i < x.size() - 1; ++i) {
            Real tmin = x[i];
            Real tmax = x[i+1];
            Real val = y[i];
            CHECK_ULP_CLOSE(y[i], s(x[i]), 2);
            for (Real t = tmin; t < tmax; t += (tmax-tmin)/16) {
                Real greater_val = s(t);
                BOOST_ASSERT(val <= greater_val);
                val = greater_val;
            }
        }


        x[0] = dis(rd);
        y[0] = dis(rd);
        // Monotone decreasing:
        for (size_t i = 1; i < n; ++i) {
            x[i] = x[i-1] + dis(rd);
            y[i] = y[i-1] - dis(rd);
        }

        x_copy = x;
        y_copy = y;
        s = pchip(std::move(x_copy), std::move(y_copy));
        //std::cout << "s = " << s << "\n";
        for (size_t i = 0; i < x.size() - 1; ++i) {
            Real tmin = x[i];
            Real tmax = x[i+1];
            Real val = y[i];
            CHECK_ULP_CLOSE(y[i], s(x[i]), 2);
            for (Real t = tmin; t < tmax; t += (tmax-tmin)/16) {
                Real lesser_val = s(t);
                BOOST_ASSERT(val >= lesser_val);
                val = lesser_val;
            }
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
    test_monotonicity<std::float32_t>();
    #else
    test_constant<float>();
    test_linear<float>();
    test_interpolation_condition<float>();
    test_monotonicity<float>();
    #endif

    #ifdef __STDCPP_FLOAT64_T__
    test_constant<std::float64_t>();
    test_linear<std::float64_t>();
    test_interpolation_condition<std::float64_t>();
    test_monotonicity<std::float64_t>();
    #else
    test_constant<double>();
    test_linear<double>();
    test_interpolation_condition<double>();
    test_monotonicity<double>();
    #endif

    test_constant<long double>();
    test_linear<long double>();
    test_interpolation_condition<long double>();
    test_monotonicity<long double>();

    #ifdef BOOST_HAS_FLOAT128
    test_constant<float128>();
    test_linear<float128>();
    #endif
#endif
    return boost::math::test::report_errors();
}

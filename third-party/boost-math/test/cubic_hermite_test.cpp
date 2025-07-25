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
#include <array>
#include <vector>
#include <boost/math/interpolators/cubic_hermite.hpp>
#include <boost/math/special_functions/next.hpp>
#include <boost/circular_buffer.hpp>
#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
using boost::multiprecision::float128;
#endif

#if __has_include(<stdfloat>)
#  include <stdfloat>
#endif

using boost::math::interpolators::cubic_hermite;
using boost::math::interpolators::cardinal_cubic_hermite;
using boost::math::interpolators::cardinal_cubic_hermite_aos;


template<typename Real>
void test_constant()
{
    Real x0 = 0;
    std::vector<Real> x{x0,1,2,3, 9, 22, 81};
    std::vector<Real> y(x.size());
    for (auto & t : y)
    {
        t = 7;
    }

    std::vector<Real> dydx(x.size(), Real(0));
    auto x_copy = x;
    auto y_copy = y;
    auto dydx_copy = dydx;
    auto hermite_spline = cubic_hermite(std::move(x_copy), std::move(y_copy), std::move(dydx_copy));

    // Now check the boundaries:
    Real tlo = x.front();
    Real thi = x.back();
    int samples = 5000;
    int i = 0;
    while (i++ < samples)
    {
        CHECK_ULP_CLOSE(Real(7), hermite_spline(tlo), 2);
        CHECK_ULP_CLOSE(Real(7), hermite_spline(thi), 2);
        CHECK_ULP_CLOSE(Real(0), hermite_spline.prime(tlo), 2);
        CHECK_ULP_CLOSE(Real(0), hermite_spline.prime(thi), 2);
        tlo = boost::math::nextafter(tlo, (std::numeric_limits<Real>::max)());
        thi = boost::math::nextafter(thi, std::numeric_limits<Real>::lowest());
    }

    boost::circular_buffer<Real> x_buf(x.size());
    for (auto & t : x) {
        x_buf.push_back(t);
    }

    boost::circular_buffer<Real> y_buf(x.size());
    for (auto & t : y) {
        y_buf.push_back(t);
    }

    boost::circular_buffer<Real> dydx_buf(x.size());
    for (auto & t : dydx) {
        dydx_buf.push_back(t);
    }

    auto circular_hermite_spline = cubic_hermite(std::move(x_buf), std::move(y_buf), std::move(dydx_buf));

    for (Real t = x[0]; t <= x.back(); t += Real(0.25)) {
        CHECK_ULP_CLOSE(Real(7), circular_hermite_spline(t), 2);
        CHECK_ULP_CLOSE(Real(0), circular_hermite_spline.prime(t), 2);
    }

    circular_hermite_spline.push_back(x.back() + 1, 7, 0);
    CHECK_ULP_CLOSE(Real(0), circular_hermite_spline.prime(x.back()+1), 2);

}

template<typename Real>
void test_linear()
{
    std::vector<Real> x{0,1,2,3};
    std::vector<Real> y{0,1,2,3};
    std::vector<Real> dydx{1,1,1,1};

    auto x_copy = x;
    auto y_copy = y;
    auto dydx_copy = dydx;
    auto hermite_spline = cubic_hermite(std::move(x_copy), std::move(y_copy), std::move(dydx_copy));

    CHECK_ULP_CLOSE(y[0], hermite_spline(x[0]), 0);
    CHECK_ULP_CLOSE(Real(1)/Real(2), hermite_spline(Real(1)/Real(2)), 10);
    CHECK_ULP_CLOSE(y[1], hermite_spline(x[1]), 0);
    CHECK_ULP_CLOSE(Real(3)/Real(2), hermite_spline(Real(3)/Real(2)), 10);
    CHECK_ULP_CLOSE(y[2], hermite_spline(x[2]), 0);
    CHECK_ULP_CLOSE(Real(5)/Real(2), hermite_spline(Real(5)/Real(2)), 10);
    CHECK_ULP_CLOSE(y[3], hermite_spline(x[3]), 0);

    x.resize(45);
    y.resize(45);
    dydx.resize(45);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = i;
        y[i] = i;
        dydx[i] = 1;
    }

    x_copy = x;
    y_copy = y;
    dydx_copy = dydx;
    hermite_spline = cubic_hermite(std::move(x_copy), std::move(y_copy), std::move(dydx_copy));
    for (Real t = 0; t < x.back(); t += Real(0.5)) {
        CHECK_ULP_CLOSE(t, hermite_spline(t), 0);
        CHECK_ULP_CLOSE(Real(1), hermite_spline.prime(t), 0);
    }

    boost::circular_buffer<Real> x_buf(x.size());
    for (auto & t : x) {
        x_buf.push_back(t);
    }

    boost::circular_buffer<Real> y_buf(x.size());
    for (auto & t : y) {
        y_buf.push_back(t);
    }

    boost::circular_buffer<Real> dydx_buf(x.size());
    for (auto & t : dydx) {
        dydx_buf.push_back(t);
    }

    auto circular_hermite_spline = cubic_hermite(std::move(x_buf), std::move(y_buf), std::move(dydx_buf));

    for (Real t = x[0]; t <= x.back(); t += Real(0.25)) {
        CHECK_ULP_CLOSE(t, circular_hermite_spline(t), 2);
        CHECK_ULP_CLOSE(Real(1), circular_hermite_spline.prime(t), 2);
    }

    circular_hermite_spline.push_back(x.back() + 1, y.back()+1, 1);

    CHECK_ULP_CLOSE(Real(y.back() + 1), circular_hermite_spline(Real(x.back()+1)), 2);
    CHECK_ULP_CLOSE(Real(1), circular_hermite_spline.prime(Real(x.back()+1)), 2);

}

template<typename Real>
void test_quadratic()
{
    std::vector<Real> x(50);
    std::default_random_engine rd;
    std::uniform_real_distribution<Real> dis(Real(0.1), Real(1));
    Real x0 = dis(rd);
    x[0] = x0;
    for (size_t i = 1; i < x.size(); ++i) {
        x[i] = x[i-1] + dis(rd);
    }
    Real xmax = x.back();

    std::vector<Real> y(x.size());
    std::vector<Real> dydx(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        y[i] = x[i]*x[i]/2;
        dydx[i] = x[i];
    }

    auto s = cubic_hermite(std::move(x), std::move(y), std::move(dydx));
    for (Real t = x0; t <= xmax; t+= Real(0.0125))
    {
        CHECK_ULP_CLOSE(t*t/2, s(t), 5);
        CHECK_ULP_CLOSE(t, s.prime(t), 138);
    }
}

template<typename Real>
void test_interpolation_condition()
{
    for (size_t n = 4; n < 50; ++n) {
        std::vector<Real> x(n);
        std::vector<Real> y(n);
        std::vector<Real> dydx(n);
        std::default_random_engine rd;
        std::uniform_real_distribution<Real> dis(0,1);
        Real x0 = dis(rd);
        x[0] = x0;
        y[0] = dis(rd);
        for (size_t i = 1; i < n; ++i) {
            x[i] = x[i-1] + dis(rd);
            y[i] = dis(rd);
            dydx[i] = dis(rd);
        }

        auto x_copy = x;
        auto y_copy = y;
        auto dydx_copy = dydx;
        auto s = cubic_hermite(std::move(x_copy), std::move(y_copy), std::move(dydx_copy));
        //std::cout << "s = " << s << "\n";
        for (size_t i = 0; i < x.size(); ++i) {
            CHECK_ULP_CLOSE(y[i], s(x[i]), 2);
            CHECK_ULP_CLOSE(dydx[i], s.prime(x[i]), 2);
        }
    }
}

template<typename Real>
void test_cardinal_constant()
{
    Real x0 = 0;
    Real dx = 2;
    std::vector<Real> y(25);
    for (auto & t : y) {
        t = 7;
    }

    std::vector<Real> dydx(y.size(), Real(0));

    auto hermite_spline = cardinal_cubic_hermite(std::move(y), std::move(dydx), x0, dx);

    for (Real t = x0; t <= x0 + 24*dx; t += Real(0.25)) {
        CHECK_ULP_CLOSE(Real(7), hermite_spline(t), 2);
        CHECK_ULP_CLOSE(Real(0), hermite_spline.prime(t), 2);
    }

    // Array of structs:

    std::vector<std::array<Real, 2>> data(25);
    for (auto & t : data) {
        t[0] = 7;
        t[1] = 0;
    }
    auto hermite_spline_aos = cardinal_cubic_hermite_aos(std::move(data), x0, dx);

    for (Real t = x0; t <= x0 + 24*dx; t += Real(0.25)) {
        if (!CHECK_ULP_CLOSE(Real(7), hermite_spline_aos(t), 2)) {
            std::cerr << "  Wrong evaluation at t = " << t << "\n";
        }
        if (!CHECK_ULP_CLOSE(Real(0), hermite_spline_aos.prime(t), 2)) {
            std::cerr << "  Wrong evaluation at t = " << t << "\n";
        }
    }

    // Now check the boundaries:
    Real tlo = x0;
    Real thi = x0 + (25-1)*dx;
    int samples = 5000;
    int i = 0;
    while (i++ < samples)
    {
        CHECK_ULP_CLOSE(Real(7), hermite_spline(tlo), 2);
        CHECK_ULP_CLOSE(Real(7), hermite_spline(thi), 2);
        CHECK_ULP_CLOSE(Real(7), hermite_spline_aos(tlo), 2);
        CHECK_ULP_CLOSE(Real(7), hermite_spline_aos(thi), 2);
        CHECK_ULP_CLOSE(Real(0), hermite_spline.prime(tlo), 2);
        CHECK_ULP_CLOSE(Real(0), hermite_spline.prime(thi), 2);
        CHECK_ULP_CLOSE(Real(0), hermite_spline_aos.prime(tlo), 2);
        CHECK_ULP_CLOSE(Real(0), hermite_spline_aos.prime(thi), 2);

        tlo = boost::math::nextafter(tlo, (std::numeric_limits<Real>::max)());
        thi = boost::math::nextafter(thi, std::numeric_limits<Real>::lowest());
    }

}


template<typename Real>
void test_cardinal_linear()
{
    Real x0 = 0;
    Real dx = 1;
    std::vector<Real> y{0,1,2,3};
    std::vector<Real> dydx{1,1,1,1};
    auto y_copy = y;
    auto dydx_copy = dydx;
    auto hermite_spline = cardinal_cubic_hermite(std::move(y_copy), std::move(dydx_copy), x0, dx);

    CHECK_ULP_CLOSE(y[0], hermite_spline(0), 0);
    CHECK_ULP_CLOSE(Real(1)/Real(2), hermite_spline(Real(1)/Real(2)), 10);
    CHECK_ULP_CLOSE(y[1], hermite_spline(1), 0);
    CHECK_ULP_CLOSE(Real(3)/Real(2), hermite_spline(Real(3)/Real(2)), 10);
    CHECK_ULP_CLOSE(y[2], hermite_spline(2), 0);
    CHECK_ULP_CLOSE(Real(5)/Real(2), hermite_spline(Real(5)/Real(2)), 10);
    CHECK_ULP_CLOSE(y[3], hermite_spline(3), 0);


    y.resize(45);
    dydx.resize(45);
    for (size_t i = 0; i < y.size(); ++i) {
        y[i] = i;
        dydx[i] = 1;
    }

    hermite_spline = cardinal_cubic_hermite(std::move(y), std::move(dydx), x0, dx);
    for (Real t = 0; t < 44; t += Real(0.5)) {
        CHECK_ULP_CLOSE(t, hermite_spline(t), 0);
        CHECK_ULP_CLOSE(Real(1), hermite_spline.prime(t), 0);
    }

    std::vector<std::array<Real, 2>> data(45);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i][0] = i;
        data[i][1] = 1;
    }

    auto hermite_spline_aos = cardinal_cubic_hermite_aos(std::move(data), x0, dx);
    for (Real t = 0; t < 44; t += Real(0.5)) {
        CHECK_ULP_CLOSE(t, hermite_spline_aos(t), 0);
        CHECK_ULP_CLOSE(Real(1), hermite_spline_aos.prime(t), 0);
    }

    Real tlo = x0;
    Real thi = x0 + (45-1)*dx;
    int samples = 5000;
    int i = 0;
    while (i++ < samples)
    {
        CHECK_ULP_CLOSE(Real(tlo), hermite_spline(tlo), 2);
        CHECK_ULP_CLOSE(Real(thi), hermite_spline(thi), 2);
        CHECK_ULP_CLOSE(Real(1), hermite_spline.prime(tlo), 2);
        CHECK_ULP_CLOSE(Real(1), hermite_spline.prime(thi), 2);
        CHECK_ULP_CLOSE(Real(tlo), hermite_spline_aos(tlo), 2);
        CHECK_ULP_CLOSE(Real(thi), hermite_spline_aos(thi), 2);
        CHECK_ULP_CLOSE(Real(1), hermite_spline_aos.prime(tlo), 2);
        CHECK_ULP_CLOSE(Real(1), hermite_spline_aos.prime(thi), 2);

        tlo = boost::math::nextafter(tlo, (std::numeric_limits<Real>::max)());
        thi = boost::math::nextafter(thi, std::numeric_limits<Real>::lowest());
    }


}


template<typename Real>
void test_cardinal_quadratic()
{
    Real x0 = -1;
    Real dx = Real(1)/Real(256);

    std::vector<Real> y(50);
    std::vector<Real> dydx(y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        Real x = x0 + i*dx;
        y[i] = x*x/2;
        dydx[i] = x;
    }

    auto s = cardinal_cubic_hermite(std::move(y), std::move(dydx), x0, dx);
    for (Real t = x0; t <= x0 + 49*dx; t+= Real(0.0125))
    {
        CHECK_ULP_CLOSE(t*t/2, s(t), 12);
        CHECK_ULP_CLOSE(t, s.prime(t), 70);
    }

    std::vector<std::array<Real, 2>> data(50);
    for (size_t i = 0; i < data.size(); ++i) {
        Real x = x0 + i*dx;
        data[i][0] = x*x/2;
        data[i][1] = x;
    }


    auto saos = cardinal_cubic_hermite_aos(std::move(data), x0, dx);
    for (Real t = x0; t <= x0 + 49*dx; t+= Real(0.0125))
    {
        CHECK_ULP_CLOSE(t*t/2, saos(t), 12);
        CHECK_ULP_CLOSE(t, saos.prime(t), 70);
    }

    auto [tlo, thi] = s.domain();
    int samples = 5000;
    int i = 0;
    while (i++ < samples)
    {
        CHECK_ULP_CLOSE(Real(tlo*tlo/2), s(tlo), 3);
        CHECK_ULP_CLOSE(Real(thi*thi/2), s(thi), 3);
        CHECK_ULP_CLOSE(Real(tlo), s.prime(tlo), 3);
        CHECK_ULP_CLOSE(Real(thi), s.prime(thi), 3);
        CHECK_ULP_CLOSE(Real(tlo*tlo/2), saos(tlo), 3);
        CHECK_ULP_CLOSE(Real(thi*thi/2), saos(thi), 3);
        CHECK_ULP_CLOSE(Real(tlo), saos.prime(tlo), 3);
        CHECK_ULP_CLOSE(Real(thi), saos.prime(thi), 3);

        tlo = boost::math::nextafter(tlo, (std::numeric_limits<Real>::max)());
        thi = boost::math::nextafter(thi, std::numeric_limits<Real>::lowest());
    }
}


template<typename Real>
void test_cardinal_interpolation_condition()
{
    for (size_t n = 4; n < 50; ++n) {
        std::vector<Real> y(n);
        std::vector<Real> dydx(n);
        std::default_random_engine rd;
        std::uniform_real_distribution<Real> dis(Real(0.1), Real(1));
        Real x0 = Real(2);
        Real dx = Real(1)/Real(128);
        for (size_t i = 0; i < n; ++i) {
            y[i] = dis(rd);
            dydx[i] = dis(rd);
        }

        auto y_copy = y;
        auto dydx_copy = dydx;
        auto s = cardinal_cubic_hermite(std::move(y_copy), std::move(dydx_copy), x0, dx);
        for (size_t i = 0; i < y.size(); ++i) {
            CHECK_ULP_CLOSE(y[i], s(x0 + i*dx), 2);
            CHECK_ULP_CLOSE(dydx[i], s.prime(x0 + i*dx), 2);
        }
    }
}



int main()
{
    #ifdef __STDCPP_FLOAT32_T__
    test_constant<std::float32_t>();
    test_linear<std::float32_t>();
    test_quadratic<std::float32_t>();
    test_interpolation_condition<std::float32_t>();
    test_cardinal_constant<std::float32_t>();
    test_cardinal_linear<std::float32_t>();
    test_cardinal_quadratic<std::float32_t>();
    test_cardinal_interpolation_condition<std::float32_t>();
    #else
    test_constant<float>();
    test_linear<float>();
    test_quadratic<float>();
    test_interpolation_condition<float>();
    test_cardinal_constant<float>();
    test_cardinal_linear<float>();
    test_cardinal_quadratic<float>();
    test_cardinal_interpolation_condition<float>();
    #endif

    #ifdef __STDCPP_FLOAT64_T__
    test_constant<std::float64_t>();
    test_linear<std::float64_t>();
    test_quadratic<std::float64_t>();
    test_interpolation_condition<std::float64_t>();
    test_cardinal_constant<std::float64_t>();
    test_cardinal_linear<std::float64_t>();
    test_cardinal_quadratic<std::float64_t>();
    test_cardinal_interpolation_condition<std::float64_t>();
    #else
    test_constant<double>();
    test_linear<double>();
    test_quadratic<double>();
    test_interpolation_condition<double>();
    test_cardinal_constant<double>();
    test_cardinal_linear<double>();
    test_cardinal_quadratic<double>();
    test_cardinal_interpolation_condition<double>();
    #endif

    test_constant<long double>();
    test_linear<long double>();
    test_quadratic<long double>();
    test_interpolation_condition<long double>();
    test_cardinal_constant<long double>();
    test_cardinal_linear<long double>();
    test_cardinal_quadratic<long double>();
    test_cardinal_interpolation_condition<long double>();


    #ifdef BOOST_HAS_FLOAT128
    test_constant<float128>();
    test_linear<float128>();
    test_cardinal_constant<float128>();
    test_cardinal_linear<float128>();
    #endif

    return boost::math::test::report_errors();
}

/*
 * Copyright Nick Thompson, 2020
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "math_unit_test.hpp"
#include <numeric>
#include <utility>
#include <array>
#include <boost/random/uniform_real.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/interpolators/septic_hermite.hpp>
#include <boost/math/special_functions/next.hpp>
#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
using boost::multiprecision::float128;
#endif

#if __has_include(<stdfloat>)
#  include <stdfloat>
#endif

using boost::math::interpolators::septic_hermite;
using boost::math::interpolators::cardinal_septic_hermite;
using boost::math::interpolators::cardinal_septic_hermite_aos;

template<typename Real>
void test_constant()
{

    std::vector<Real> x{0,1,2,3, 9, 22, 81};
    std::vector<Real> y(x.size());
    std::vector<Real> dydx(x.size(), 0);
    std::vector<Real> d2ydx2(x.size(), 0);
    std::vector<Real> d3ydx3(x.size(), 0);
    for (auto & t : y)
    {
        t = 7;
    }

    auto sh = septic_hermite(std::move(x), std::move(y), std::move(dydx), std::move(d2ydx2), std::move(d3ydx3));

    for (Real t = 0; t <= 81; t += Real(0.25))
    {
        CHECK_ULP_CLOSE(Real(7), sh(t), 24);
        CHECK_ULP_CLOSE(Real(0), sh.prime(t), 24);
    }

    Real x0 = 0;
    Real dx = 1;
    y.resize(128, 7);
    dydx.resize(128, 0);
    d2ydx2.resize(128, 0);
    d3ydx3.resize(128, 0);
    auto csh = cardinal_septic_hermite(std::move(y), std::move(dydx), std::move(d2ydx2), std::move(d3ydx3), x0, dx);
    for (Real t = x0; t <= 127; t += Real(0.25))
    {
        CHECK_ULP_CLOSE(Real(7), csh(t), 24);
        CHECK_ULP_CLOSE(Real(0), csh.prime(t), 24);
        CHECK_ULP_CLOSE(Real(0), csh.double_prime(t), 24);
    }

    std::vector<std::array<Real, 4>> data(128);
    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i][0] = 7;
        data[i][1] = 0;
        data[i][2] = 0;
        data[i][3] = 0;
    }
    auto csh_aos = cardinal_septic_hermite_aos(std::move(data), x0, dx);
    for (Real t = x0; t <= 127; t += Real(0.25))
    {
        CHECK_ULP_CLOSE(Real(7), csh_aos(t), 24);
        CHECK_ULP_CLOSE(Real(0), csh_aos.prime(t), 24);
        CHECK_ULP_CLOSE(Real(0), csh_aos.double_prime(t), 24);
    }

    // Now check the boundaries:
    auto [tlo, thi] = csh.domain();
    int samples = 5000;
    int i = 0;
    while (i++ < samples)
    {
        CHECK_ULP_CLOSE(Real(7), csh(tlo), 2);
        CHECK_ULP_CLOSE(Real(7), csh(thi), 2);
        CHECK_ULP_CLOSE(Real(7), csh_aos(tlo), 2);
        CHECK_ULP_CLOSE(Real(7), csh_aos(thi), 2);
        CHECK_ULP_CLOSE(Real(0), csh.prime(tlo), 2);
        CHECK_ULP_CLOSE(Real(0), csh.prime(thi), 2);
        CHECK_ULP_CLOSE(Real(0), csh_aos.prime(tlo), 2);
        CHECK_ULP_CLOSE(Real(0), csh_aos.prime(thi), 2);
        CHECK_ULP_CLOSE(Real(0), csh.double_prime(tlo), 2);
        CHECK_ULP_CLOSE(Real(0), csh.double_prime(thi), 2);
        CHECK_ULP_CLOSE(Real(0), csh_aos.double_prime(tlo), 2);
        CHECK_ULP_CLOSE(Real(0), csh_aos.double_prime(thi), 2);

        tlo = boost::math::nextafter(tlo, (std::numeric_limits<Real>::max)());
        thi = boost::math::nextafter(thi, std::numeric_limits<Real>::lowest());
    }

}


template<typename Real>
void test_linear()
{
    std::vector<Real> x{0,1,2,3,4,5,6,7,8,9};
    std::vector<Real> y = x;
    std::vector<Real> dydx(x.size(), 1);
    std::vector<Real> d2ydx2(x.size(), 0);
    std::vector<Real> d3ydx3(x.size(), 0);

    auto sh = septic_hermite(std::move(x), std::move(y), std::move(dydx), std::move(d2ydx2), std::move(d3ydx3));

    for (Real t = 0; t <= 9; t += Real(0.25))
    {
        CHECK_ULP_CLOSE(Real(t), sh(t), 2);
        CHECK_ULP_CLOSE(Real(1), sh.prime(t), 2);
    }

    boost::random::mt19937 rng;
    boost::random::uniform_real_distribution<Real> dis(Real(0.5), Real(1));
    x.resize(512);
    x[0] = dis(rng);
    Real xmin = x[0];
    for (size_t i = 1; i < x.size(); ++i)
    {
        x[i] = x[i-1] + dis(rng);
    }
    Real xmax = x.back();

    y = x;
    dydx.resize(x.size(), 1);
    d2ydx2.resize(x.size(), 0);
    d3ydx3.resize(x.size(), 0);

    sh = septic_hermite(std::move(x), std::move(y), std::move(dydx), std::move(d2ydx2), std::move(d3ydx3));

    for (Real t = xmin; t <= xmax; t += Real(0.125))
    {
        CHECK_ULP_CLOSE(t, sh(t), 25);
        CHECK_ULP_CLOSE(Real(1), sh.prime(t), 850);
    }

    Real x0 = 0;
    Real dx = 1;
    y.resize(10);
    dydx.resize(10, 1);
    d2ydx2.resize(10, 0);
    d3ydx3.resize(10, 0);
    for (size_t i = 0; i < y.size(); ++i)
    {
        y[i] = i;
    }
    auto csh = cardinal_septic_hermite(std::move(y), std::move(dydx), std::move(d2ydx2), std::move(d3ydx3), x0, dx);
    for (Real t = 0; t <= 9; t += Real(0.125))
    {
        CHECK_ULP_CLOSE(t, csh(t), 15);
        CHECK_ULP_CLOSE(Real(1), csh.prime(t), 15);
        CHECK_ULP_CLOSE(Real(0), csh.double_prime(t), 15);
    }

    std::vector<std::array<Real, 4>> data(10);
    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i][0] = i;
        data[i][1] = 1;
        data[i][2] = 0;
        data[i][3] = 0;
    }
    auto csh_aos = cardinal_septic_hermite_aos(std::move(data), x0, dx);
    for (Real t = 0; t <= 9; t += Real(0.125))
    {
        CHECK_ULP_CLOSE(t, csh_aos(t), 15);
        CHECK_ULP_CLOSE(Real(1), csh_aos.prime(t), 15);
        CHECK_ULP_CLOSE(Real(0), csh_aos.double_prime(t), 15);
    }

    // Now check the boundaries:
    auto [tlo, thi] = csh.domain();
    int samples = 5000;
    int i = 0;
    while (i++ < samples)
    {
        CHECK_ULP_CLOSE(Real(tlo), csh(tlo), 2);
        CHECK_ULP_CLOSE(Real(thi), csh(thi), 8);
        CHECK_ULP_CLOSE(Real(tlo), csh_aos(tlo), 2);
        CHECK_ULP_CLOSE(Real(thi), csh_aos(thi), 8);
        CHECK_ULP_CLOSE(Real(1), csh.prime(tlo), 2);
        CHECK_ULP_CLOSE(Real(1), csh.prime(thi), 700);
        CHECK_ULP_CLOSE(Real(1), csh_aos.prime(tlo), 2);
        CHECK_ULP_CLOSE(Real(1), csh_aos.prime(thi), 700);
        CHECK_MOLLIFIED_CLOSE(Real(0), csh.double_prime(tlo), std::numeric_limits<Real>::epsilon());
        CHECK_MOLLIFIED_CLOSE(Real(0), csh.double_prime(thi), 1200*std::numeric_limits<Real>::epsilon());
        CHECK_MOLLIFIED_CLOSE(Real(0), csh_aos.double_prime(tlo), std::numeric_limits<Real>::epsilon());
        CHECK_MOLLIFIED_CLOSE(Real(0), csh_aos.double_prime(thi), 1200*std::numeric_limits<Real>::epsilon());

        tlo = boost::math::nextafter(tlo, (std::numeric_limits<Real>::max)());
        thi = boost::math::nextafter(thi, std::numeric_limits<Real>::lowest());
    }

}

template<typename Real>
void test_quadratic()
{
    std::vector<Real> x{0,1,2,3,4,5,6,7,8,9};
    std::vector<Real> y(x.size());
    for (size_t i = 0; i < y.size(); ++i)
    {
        y[i] = x[i]*x[i]/2;
    }

    std::vector<Real> dydx(x.size());
    for (size_t i = 0; i < y.size(); ++i)
    {
        dydx[i] = x[i];
    }

    std::vector<Real> d2ydx2(x.size(), 1);
    std::vector<Real> d3ydx3(x.size(), 0);

    auto sh = septic_hermite(std::move(x), std::move(y), std::move(dydx), std::move(d2ydx2), std::move(d3ydx3));

    for (Real t = 0; t <= 9; t += Real(0.0078125))
    {
        CHECK_ULP_CLOSE(t*t/2, sh(t), 100);
        CHECK_ULP_CLOSE(t, sh.prime(t), 32);
    }

    boost::random::mt19937 rng;
    boost::random::uniform_real_distribution<Real> dis(Real(0.5), Real(1));
    x.resize(8);
    x[0] = dis(rng);
    Real xmin = x[0];
    for (size_t i = 1; i < x.size(); ++i)
    {
        x[i] = x[i-1] + dis(rng);
    }
    Real xmax = x.back();

    y.resize(x.size());
    for (size_t i = 0; i < y.size(); ++i)
    {
        y[i] = x[i]*x[i]/2;
    }

    dydx.resize(x.size());
    for (size_t i = 0; i < y.size(); ++i)
    {
        dydx[i] = x[i];
    }

    d2ydx2.resize(x.size(), 1);
    d3ydx3.resize(x.size(), 0); 

    sh = septic_hermite(std::move(x), std::move(y), std::move(dydx), std::move(d2ydx2), std::move(d3ydx3));

    for (Real t = xmin; t <= xmax; t += Real(0.125))
    {
        CHECK_ULP_CLOSE(t*t/2, sh(t), 50);
        CHECK_ULP_CLOSE(t, sh.prime(t), 300);
    }

    y.resize(10);
    for (size_t i = 0; i < y.size(); ++i)
    {
        y[i] = i*i/Real(2);
    }

    dydx.resize(y.size());
    for (size_t i = 0; i < y.size(); ++i)
    {
        dydx[i] = i;
    }

    d2ydx2.resize(y.size(), 1);
    d3ydx3.resize(y.size(), 0);

    Real x0 = 0;
    Real dx = 1;
    auto csh = cardinal_septic_hermite(std::move(y), std::move(dydx), std::move(d2ydx2), std::move(d3ydx3), x0, dx);
    for (Real t = x0; t <= 9; t += Real(0.125))
    {
        CHECK_ULP_CLOSE(t*t/2, csh(t), 24);
        CHECK_ULP_CLOSE(t, csh.prime(t), 24);
        CHECK_ULP_CLOSE(Real(1), csh.double_prime(t), 24);
    }

    std::vector<std::array<Real, 4>> data(10);
    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i][0] = i*i/Real(2);
        data[i][1] = i;
        data[i][2] = 1;
        data[i][3] = 0;
    }
    auto csh_aos = cardinal_septic_hermite_aos(std::move(data), x0, dx);
    for (Real t = x0; t <= 9; t += Real(0.125))
    {
        CHECK_ULP_CLOSE(t*t/2, csh_aos(t), 24);
        CHECK_ULP_CLOSE(t, csh_aos.prime(t), 24);
        CHECK_ULP_CLOSE(Real(1), csh_aos.double_prime(t), 24);
    }
}



template<typename Real>
void test_cubic()
{

    std::vector<Real> x{0,1,2,3,4,5,6,7};
    Real xmax = x.back();
    std::vector<Real> y(x.size());
    for (size_t i = 0; i < y.size(); ++i)
    {
        y[i] = x[i]*x[i]*x[i];
    }

    std::vector<Real> dydx(x.size());
    for (size_t i = 0; i < y.size(); ++i)
    {
        dydx[i] = 3*x[i]*x[i];
    }

    std::vector<Real> d2ydx2(x.size());
    for (size_t i = 0; i < y.size(); ++i)
    {
        d2ydx2[i] = 6*x[i];
    }
    std::vector<Real> d3ydx3(x.size(), 6);

    auto sh = septic_hermite(std::move(x), std::move(y), std::move(dydx), std::move(d2ydx2), std::move(d3ydx3));

    for (Real t = 0; t <= xmax; t += Real(0.0078125))
    {
        CHECK_ULP_CLOSE(t*t*t, sh(t), 151);
        CHECK_ULP_CLOSE(3*t*t, sh.prime(t), 151);
    }

    Real x0 = 0;
    Real dx = 1;
    y.resize(8);
    dydx.resize(8);
    d2ydx2.resize(8);
    d3ydx3.resize(8,6);
    for (size_t i = 0; i < y.size(); ++i)
    {
        y[i] = i*i*i;
        dydx[i] = 3*i*i;
        d2ydx2[i] = 6*i;
    }

    auto csh = cardinal_septic_hermite(std::move(y), std::move(dydx), std::move(d2ydx2), std::move(d3ydx3), x0, dx);

    for (Real t = 0; t <= xmax; t += Real(0.0078125))
    {
        CHECK_ULP_CLOSE(t*t*t, csh(t), 151);
        CHECK_ULP_CLOSE(3*t*t, csh.prime(t), 151);
        CHECK_ULP_CLOSE(6*t, csh.double_prime(t), 151);
    }

    std::vector<std::array<Real, 4>> data(8);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i][0] = i*i*i;
        data[i][1] = 3*i*i;
        data[i][2] = 6*i;
        data[i][3] = 6;
    }

    auto csh_aos = cardinal_septic_hermite_aos(std::move(data), x0, dx);

    for (Real t = 0; t <= xmax; t += Real(0.0078125))
    {
        CHECK_ULP_CLOSE(t*t*t, csh_aos(t), 151);
        CHECK_ULP_CLOSE(3*t*t, csh_aos.prime(t), 151);
        CHECK_ULP_CLOSE(6*t, csh_aos.double_prime(t), 151);
    }
}

template<typename Real>
void test_quartic()
{

    std::vector<Real> x{0,1,2,3,4,5,6,7,8,9};
    Real xmax = x.back();
    std::vector<Real> y(x.size());
    for (size_t i = 0; i < y.size(); ++i)
    {
        y[i] = x[i]*x[i]*x[i]*x[i];
    }

    std::vector<Real> dydx(x.size());
    for (size_t i = 0; i < y.size(); ++i)
    {
        dydx[i] = 4*x[i]*x[i]*x[i];
    }

    std::vector<Real> d2ydx2(x.size());
    for (size_t i = 0; i < y.size(); ++i)
    {
        d2ydx2[i] = 12*x[i]*x[i];
    }

    std::vector<Real> d3ydx3(x.size());
    for (size_t i = 0; i < y.size(); ++i)
    {
        d3ydx3[i] = 24*x[i];
    }

    auto sh = septic_hermite(std::move(x), std::move(y), std::move(dydx), std::move(d2ydx2), std::move(d3ydx3));

    for (Real t = 1; t <= xmax; t += Real(0.0078125)) {
        CHECK_ULP_CLOSE(t*t*t*t, sh(t), 117);
        CHECK_ULP_CLOSE(4*t*t*t, sh.prime(t), 117);
    }

    y.resize(10);
    dydx.resize(10);
    d2ydx2.resize(10);
    d3ydx3.resize(10);
    for (size_t i = 0; i < y.size(); ++i)
    {
        y[i] = i*i*i*i;
        dydx[i] = 4*i*i*i;
        d2ydx2[i] = 12*i*i;
        d3ydx3[i] = 24*i;
    }

    auto csh = cardinal_septic_hermite(std::move(y), std::move(dydx), std::move(d2ydx2), std::move(d3ydx3), Real(0), Real(1));

    for (Real t = 1; t <= xmax; t += Real(0.0078125))
    {
        CHECK_ULP_CLOSE(t*t*t*t, csh(t), 117);
        CHECK_ULP_CLOSE(4*t*t*t, csh.prime(t), 117);
        CHECK_ULP_CLOSE(12*t*t, csh.double_prime(t), 117);
    }

    std::vector<std::array<Real, 4>> data(10);
    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i][0] = i*i*i*i;
        data[i][1] = 4*i*i*i;
        data[i][2] = 12*i*i;
        data[i][3] = 24*i;
    }

    auto csh_aos = cardinal_septic_hermite_aos(std::move(data), Real(0), Real(1));
    for (Real t = 1; t <= xmax; t += Real(0.0078125))
    {
        CHECK_ULP_CLOSE(t*t*t*t, csh_aos(t), 117);
        CHECK_ULP_CLOSE(4*t*t*t, csh_aos.prime(t), 117);
        CHECK_ULP_CLOSE(12*t*t, csh_aos.double_prime(t), 117);
    }
}


template<typename Real>
void test_interpolation_condition()
{
    for (size_t n = 4; n < 50; ++n) {
        std::vector<Real> x(n);
        std::vector<Real> y(n);
        std::vector<Real> dydx(n);
        std::vector<Real> d2ydx2(n);
        std::vector<Real> d3ydx3(n);
        boost::random::mt19937 rd; 
        boost::random::uniform_real_distribution<Real> dis(0,1);
        Real x0 = dis(rd);
        x[0] = x0;
        y[0] = dis(rd);
        for (size_t i = 1; i < n; ++i) {
            x[i] = x[i-1] + dis(rd);
            y[i] = dis(rd);
            dydx[i] = dis(rd);
            d2ydx2[i] = dis(rd);
            d3ydx3[i] = dis(rd);
        }

        auto x_copy = x;
        auto y_copy = y;
        auto dydx_copy = dydx;
        auto d2ydx2_copy = d2ydx2;
        auto d3ydx3_copy = d3ydx3;
        auto s = septic_hermite(std::move(x_copy), std::move(y_copy), std::move(dydx_copy), std::move(d2ydx2_copy), std::move(d3ydx3_copy));

        for (size_t i = 0; i < x.size(); ++i)
        {
            CHECK_ULP_CLOSE(y[i], s(x[i]), 2);
            CHECK_ULP_CLOSE(dydx[i], s.prime(x[i]), 2);
        }
    }
}


int main()
{
    #ifdef __STDCPP_FLOAT32_T__
    test_constant<std::float32_t>();
    test_linear<std::float32_t>();
    test_quadratic<std::float32_t>();
    test_cubic<std::float32_t>();
    test_quartic<std::float32_t>();
    test_interpolation_condition<std::float32_t>();
    #else
    test_constant<float>();
    test_linear<float>();
    test_quadratic<float>();
    test_cubic<float>();
    test_quartic<float>();
    test_interpolation_condition<float>();
    #endif

    #ifdef __STDCPP_FLOAT64_T__
    test_constant<std::float64_t>();
    test_linear<std::float64_t>();
    test_quadratic<std::float64_t>();
    test_cubic<std::float64_t>();
    test_quartic<std::float64_t>();
    test_interpolation_condition<std::float64_t>();
    #else
    test_constant<double>();
    test_linear<double>();
    test_quadratic<double>();
    test_cubic<double>();
    test_quartic<double>();
    test_interpolation_condition<double>();
    #endif

    test_constant<long double>();
    test_linear<long double>();
    test_quadratic<long double>();
    test_cubic<long double>();
    test_quartic<long double>();
    test_interpolation_condition<long double>();

    #ifdef BOOST_HAS_FLOAT128
    test_constant<float128>();
    test_linear<float128>();
    test_quadratic<float128>();
    test_cubic<float128>();
    test_quartic<float128>();
    test_interpolation_condition<float128>();
    #endif

    return boost::math::test::report_errors();
}

/*
 * Copyright Nick Thompson, 2019
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "math_unit_test.hpp"
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>
#include <limits>
#include <boost/math/interpolators/cardinal_quintic_b_spline.hpp>
#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
using boost::multiprecision::float128;
#endif
using boost::math::interpolators::cardinal_quintic_b_spline;

#if __has_include(<stdfloat>)
#  include <stdfloat>
#endif

template<class Real>
void test_constant()
{
    Real c = Real(7.5);
    Real t0 = 0;
    Real h = Real(1)/Real(16);
    size_t n = 513;
    std::vector<Real> v(n, c);
    std::pair<Real, Real> left_endpoint_derivatives{0, 0};
    std::pair<Real, Real> right_endpoint_derivatives{0, 0};
    auto qbs = cardinal_quintic_b_spline<Real>(v.data(), v.size(), t0, h, left_endpoint_derivatives, right_endpoint_derivatives);

    size_t i = 0;
    while (i < n) {
      Real t = t0 + i*h;
      CHECK_ULP_CLOSE(c, qbs(t), 3);
      CHECK_MOLLIFIED_CLOSE(Real(0), qbs.prime(t), 400*std::numeric_limits<Real>::epsilon());
      CHECK_MOLLIFIED_CLOSE(Real(0), qbs.double_prime(t), 60000*std::numeric_limits<Real>::epsilon());
      ++i;
    }

    i = 0;
    while (i < n - 1) {
      Real t = t0 + i*h + h/2;
      CHECK_ULP_CLOSE(c, qbs(t), 5);
      CHECK_MOLLIFIED_CLOSE(Real(0), qbs.prime(t), 600*std::numeric_limits<Real>::epsilon());
      CHECK_MOLLIFIED_CLOSE(Real(0), qbs.double_prime(t), 30000*std::numeric_limits<Real>::epsilon());
      t = t0 + i*h + h/4;
      CHECK_ULP_CLOSE(c, qbs(t), 4);
      CHECK_MOLLIFIED_CLOSE(Real(0), qbs.prime(t), 600*std::numeric_limits<Real>::epsilon());
      CHECK_MOLLIFIED_CLOSE(Real(0), qbs.double_prime(t), 10000*std::numeric_limits<Real>::epsilon());
      ++i;
    }
}

template<class Real>
void test_constant_estimate_derivatives()
{
    Real c = Real(7.5);
    Real t0 = 0;
    Real h = Real(1)/Real(16);
    size_t n = 513;
    std::vector<Real> v(n, c);
    auto qbs = cardinal_quintic_b_spline<Real>(v.data(), v.size(), t0, h);

    size_t i = 0;
    while (i < n) {
      Real t = t0 + i*h;
      CHECK_ULP_CLOSE(c, qbs(t), 3);
      CHECK_MOLLIFIED_CLOSE(Real(0), qbs.prime(t), 1200*std::numeric_limits<Real>::epsilon());
      CHECK_MOLLIFIED_CLOSE(Real(0), qbs.double_prime(t), 200000*std::numeric_limits<Real>::epsilon());
      ++i;
    }

    i = 0;
    while (i < n - 1) {
      Real t = t0 + i*h + h/2;
      CHECK_ULP_CLOSE(c, qbs(t), 8);
      CHECK_MOLLIFIED_CLOSE(Real(0), qbs.prime(t), 1200*std::numeric_limits<Real>::epsilon());
      CHECK_MOLLIFIED_CLOSE(Real(0), qbs.double_prime(t), 80000*std::numeric_limits<Real>::epsilon());
      t = t0 + i*h + h/4;
      CHECK_ULP_CLOSE(c, qbs(t), 5);
      CHECK_MOLLIFIED_CLOSE(Real(0), qbs.prime(t), 1200*std::numeric_limits<Real>::epsilon());
      CHECK_MOLLIFIED_CLOSE(Real(0), qbs.double_prime(t), 38000*std::numeric_limits<Real>::epsilon());
      ++i;
    }
}


template<class Real>
void test_linear()
{
    using std::abs;
    Real m = Real(8.3);
    Real b = Real(7.2);
    Real t0 = 0;
    Real h = Real(1)/Real(16);
    size_t n = 512;
    std::vector<Real> y(n);
    for (size_t i = 0; i < n; ++i) {
      Real t = i*h;
      y[i] = m*t + b;
    }
    std::pair<Real, Real> left_endpoint_derivatives{m, 0};
    std::pair<Real, Real> right_endpoint_derivatives{m, 0};
    auto qbs = cardinal_quintic_b_spline<Real>(y.data(), y.size(), t0, h, left_endpoint_derivatives, right_endpoint_derivatives);

    size_t i = 0;
    while (i < n) {
      Real t = t0 + i*h;
      if (!CHECK_ULP_CLOSE(m*t+b, qbs(t), 3)) {
          std::cerr << "  Problem at t = " << t << "\n";
      }
      if(!CHECK_MOLLIFIED_CLOSE(m, qbs.prime(t), 100*abs(m*t+b)*std::numeric_limits<Real>::epsilon())) {
          std::cerr << "  Problem at t = " << t << "\n";
      }
      if(!CHECK_MOLLIFIED_CLOSE(0, qbs.double_prime(t), 10000*abs(m*t+b)*std::numeric_limits<Real>::epsilon())) {
          std::cerr << "  Problem at t = " << t << "\n";
      }
      ++i;
    }

    i = 0;
    while (i < n - 1) {
      Real t = t0 + i*h + h/2;
      if(!CHECK_ULP_CLOSE(m*t+b, qbs(t), 4)) {
          std::cerr << "  Problem at t = " << t << "\n";
      }
      CHECK_MOLLIFIED_CLOSE(m, qbs.prime(t), 1500*std::numeric_limits<Real>::epsilon());
      t = t0 + i*h + h/4;
      if(!CHECK_ULP_CLOSE(m*t+b, qbs(t), 4)) {
          std::cerr << "  Problem at t = " << t << "\n";
      }
      CHECK_MOLLIFIED_CLOSE(m, qbs.prime(t), 3000*std::numeric_limits<Real>::epsilon());
      ++i;
    }
}

template<class Real>
void test_linear_estimate_derivatives()
{
    using std::abs;
    Real m = Real(8.3);
    Real b = Real(7.2);
    Real t0 = 0;
    Real h = Real(1)/Real(16);
    size_t n = 512;
    std::vector<Real> y(n);
    for (size_t i = 0; i < n; ++i) {
      Real t = i*h;
      y[i] = m*t + b;
    }

    auto qbs = cardinal_quintic_b_spline<Real>(y.data(), y.size(), t0, h);

    size_t i = 0;
    while (i < n) {
      Real t = t0 + i*h;
      if (!CHECK_ULP_CLOSE(m*t+b, qbs(t), 3)) {
          std::cerr << "  Problem at t = " << t << "\n";
      }
      if(!CHECK_MOLLIFIED_CLOSE(m, qbs.prime(t), 100*abs(m*t+b)*std::numeric_limits<Real>::epsilon())) {
          std::cerr << "  Problem at t = " << t << "\n";
      }
      if(!CHECK_MOLLIFIED_CLOSE(0, qbs.double_prime(t), 20000*abs(m*t+b)*std::numeric_limits<Real>::epsilon())) {
          std::cerr << "  Problem at t = " << t << "\n";
      }
      ++i;
    }

    i = 0;
    while (i < n - 1) {
      Real t = t0 + i*h + h/2;
      if(!CHECK_ULP_CLOSE(m*t+b, qbs(t), 5)) {
          std::cerr << "  Problem at t = " << t << "\n";
      }
      CHECK_MOLLIFIED_CLOSE(m, qbs.prime(t), 1500*std::numeric_limits<Real>::epsilon());
      t = t0 + i*h + h/4;
      if(!CHECK_ULP_CLOSE(m*t+b, qbs(t), 4)) {
          std::cerr << "  Problem at t = " << t << "\n";
      }
      CHECK_MOLLIFIED_CLOSE(m, qbs.prime(t), 3000*std::numeric_limits<Real>::epsilon());
      ++i;
    }
}


template<class Real>
void test_quadratic()
{
    Real a = Real(1)/Real(16);
    Real b = Real(-3.5);
    Real c = Real(-9);
    Real t0 = 0;
    Real h = Real(1)/Real(16);
    size_t n = 513;
    std::vector<Real> y(n);
    for (size_t i = 0; i < n; ++i) {
      Real t = i*h;
      y[i] = a*t*t + b*t + c;
    }
    Real t_max = t0 + (n-1)*h;
    std::pair<Real, Real> left_endpoint_derivatives{b, 2*a};
    std::pair<Real, Real> right_endpoint_derivatives{2*a*t_max + b, 2*a};

    auto qbs = cardinal_quintic_b_spline<Real>(y, t0, h, left_endpoint_derivatives, right_endpoint_derivatives);

    size_t i = 0;
    while (i < n) {
      Real t = t0 + i*h;
      CHECK_ULP_CLOSE(a*t*t + b*t + c, qbs(t), 3);
      ++i;
    }

    i = 0;
    while (i < n -1) {
      Real t = t0 + i*h + h/2;
      if(!CHECK_ULP_CLOSE(a*t*t + b*t + c, qbs(t), 5)) {
          std::cerr << "  Problem at abscissa t = " << t << "\n";
      }

      t = t0 + i*h + h/4;
      if (!CHECK_ULP_CLOSE(a*t*t + b*t + c, qbs(t), 5)) {
          std::cerr << "  Problem abscissa t = " << t << "\n";
      }
      ++i;
    }
}


template<class Real>
void test_quadratic_estimate_derivatives()
{
    Real a = Real(1)/Real(16);
    Real b = Real(-3.5);
    Real c = Real(-9);
    Real t0 = 0;
    Real h = Real(1)/Real(16);
    size_t n = 513;
    std::vector<Real> y(n);
    for (size_t i = 0; i < n; ++i) {
      Real t = i*h;
      y[i] = a*t*t + b*t + c;
    }
    auto qbs = cardinal_quintic_b_spline<Real>(y, t0, h);

    size_t i = 0;
    while (i < n) {
      Real t = t0 + i*h;
      CHECK_ULP_CLOSE(a*t*t + b*t + c, qbs(t), 3);
      ++i;
    }

    i = 0;
    while (i < n -1) {
      Real t = t0 + i*h + h/2;
      if(!CHECK_ULP_CLOSE(a*t*t + b*t + c, qbs(t), 10)) {
          std::cerr << "  Problem at abscissa t = " << t << "\n";
      }

      t = t0 + i*h + h/4;
      if (!CHECK_ULP_CLOSE(a*t*t + b*t + c, qbs(t), 6)) {
          std::cerr << "  Problem abscissa t = " << t << "\n";
      }
      ++i;
    }
}


int main()
{
    #ifdef __STDCPP_FLOAT32_T__
    test_linear<std::float32_t>();
    #else
    test_linear<float>();
    #endif
    
    #ifdef __STDCPP_FLOAT64_T__
    test_constant<std::float64_t>();
    test_linear<std::float64_t>();
    test_constant_estimate_derivatives<std::float64_t>();
    test_linear_estimate_derivatives<std::float64_t>();
    test_quadratic<std::float64_t>();
    test_quadratic_estimate_derivatives<std::float64_t>();
    #else
    test_constant<double>();
    test_linear<double>();
    test_constant_estimate_derivatives<double>();
    test_linear_estimate_derivatives<double>();
    test_quadratic<double>();
    test_quadratic_estimate_derivatives<double>();
    #endif

    #ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test_constant<long double>();
    test_constant_estimate_derivatives<long double>();
    test_linear<long double>();
    test_linear_estimate_derivatives<long double>();
    test_quadratic<long double>();
    test_quadratic_estimate_derivatives<long double>();
    #endif

    #ifdef BOOST_HAS_FLOAT128
    test_constant<float128>();
    test_linear<float128>();
    test_linear_estimate_derivatives<float128>();
    #endif

    return boost::math::test::report_errors();
}

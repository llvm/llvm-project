/*
 * Copyright Nick Thompson, 2024
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */
#ifndef TEST_FUNCTIONS_FOR_OPTIMIZATION_HPP
#define TEST_FUNCTIONS_FOR_OPTIMIZATION_HPP
#include <array>
#include <vector>
#include <boost/math/constants/constants.hpp>
#if __has_include(<boost/units/systems/si/length.hpp>)
// This is the only system boost.units still works on.
// I imagine this will start to fail at some point,
// and we'll have to remove this test as well.
#if defined(__APPLE__)
#define BOOST_MATH_TEST_UNITS_COMPATIBILITY 1
#include <boost/units/systems/si/length.hpp>
#include <boost/units/systems/si/area.hpp>
#include <boost/units/cmath.hpp>
#include <boost/units/quantity.hpp>
#include <boost/units/systems/si/io.hpp>
using namespace boost::units;
using namespace boost::units::si;

// This *should* return an area, but see: https://github.com/boostorg/units/issues/58
// This sadly prevents std::atomic<quantity<area>>.
// Nonetheless, we *do* get some information making the argument type dimensioned,
// even if it would be better to get the full information:
double dimensioned_sphere(std::vector<quantity<length>> const & v) {
  quantity<area> r(0.0*meters*meters);
  for (auto const & x : v) {
    r += (x * x);
  }
  quantity<area> scale(1.0*meters*meters);
  return static_cast<double>(r/scale);
}
#endif
#endif

// Taken from: https://en.wikipedia.org/wiki/Test_functions_for_optimization
template <typename Real> Real ackley(std::array<Real, 2> const &v) {
  using std::sqrt;
  using std::cos;
  using std::exp;
  using boost::math::constants::two_pi;
  using boost::math::constants::e;
  Real x = v[0];
  Real y = v[1];
  Real arg1 = -sqrt((x * x + y * y) / 2) / 5;
  Real arg2 = cos(two_pi<Real>() * x) + cos(two_pi<Real>() * y);
  return -20 * exp(arg1) - exp(arg2 / 2) + 20 + e<Real>();
}

template <typename Real> auto rosenbrock_saddle(std::array<Real, 2> const &v) {
  auto x = v[0];
  auto y = v[1];
  return 100 * (x * x - y) * (x * x - y) + (1 - x) * (1 - x);
}


template <class Real> Real rastrigin(std::vector<Real> const &v) {
  using std::cos;
  using boost::math::constants::two_pi;
  auto A = static_cast<Real>(10);
  auto y = static_cast<Real>(10 * v.size());
  for (auto x : v) {
    y += x * x - A * cos(two_pi<Real>() * x);
  }
  return y;
}

// Useful for testing return-type != scalar argument type,
// and robustness to NaNs:
double sphere(std::vector<float> const &v) {
  double r = 0.0;
  for (auto x : v) {
    double x_ = static_cast<double>(x);
    r += x_ * x_;
  }
  if (r >= 1) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return r;
}

template<typename Real>
Real three_hump_camel(std::array<Real, 2> const & v) {
  Real x = v[0];
  Real y = v[1];
  auto xsq = x*x;
  return 2*xsq - (1 + Real(1)/Real(20))*xsq*xsq  + xsq*xsq*xsq/6 + x*y + y*y;
}

// Minima occurs at (3, 1/2) with value 0:
template<typename Real>
Real beale(std::array<Real, 2> const & v) {
  Real x = v[0];
  Real y = v[1];
  Real t1 = Real(3)/Real(2) -x + x*y;
  Real t2 = Real(9)/Real(4) -x  + x*y*y;
  Real t3 = Real(21)/Real(8) -x  + x*y*y*y;
  return t1*t1 + t2*t2 + t3*t3;
}


#endif

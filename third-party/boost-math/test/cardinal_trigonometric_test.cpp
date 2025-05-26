/*
 * Copyright Nick Thompson, 2019
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "math_unit_test.hpp"
#include <vector>
#include <random>
#include <boost/math/constants/constants.hpp>
#include <boost/math/interpolators/cardinal_trigonometric.hpp>
#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
#endif


using std::sin;
using boost::math::constants::two_pi;
using boost::math::interpolators::cardinal_trigonometric;

template<class Real>
void test_constant()
{
    Real t0 = 0;
    Real h = 1;
    for(size_t n = 1; n < 20; ++n)
    {
      Real c = 8;
      std::vector<Real> v(n, c);
      auto ct = cardinal_trigonometric<decltype(v)>(v, t0, h);
      CHECK_ULP_CLOSE(c, ct(0.3), 3);
      CHECK_ULP_CLOSE(c*h*n, ct.integrate(), 3);
      CHECK_ULP_CLOSE(c*c*h*n, ct.squared_l2(), 3);
      CHECK_MOLLIFIED_CLOSE(Real(0), ct.prime(0.8), 25*std::numeric_limits<Real>::epsilon());
      CHECK_MOLLIFIED_CLOSE(Real(0), ct.double_prime(0.8), 25*std::numeric_limits<Real>::epsilon());
    }
}

template<class Real>
void test_interpolation_condition()
{
  std::mt19937 gen(1234);
  std::uniform_real_distribution<Real> dis(1, 10);

  for(size_t n = 1; n < 20; ++n) {
    Real t0 = dis(gen);
    Real h = dis(gen);
    std::vector<Real> v(n);
    for (size_t i = 0; i < n; ++i) {
      v[i] = dis(gen);
    }
    auto ct = cardinal_trigonometric<decltype(v)>(v, t0, h);
    for (size_t i = 0; i < n; ++i) {
      Real arg = t0 + i*h;
      Real expected = v[i];
      Real computed = ct(arg);
      if(!CHECK_ULP_CLOSE(expected, computed, 5*n))
      {
        std::cerr << "  Samples: " << n << "\n";
      }
    }
  }

}


#ifdef BOOST_HAS_FLOAT128
void test_constant_q()
{
    __float128 t0 = 0;
    __float128 h = 1;
    for(size_t n = 1; n < 20; ++n)
    {
      __float128 c = 8;
      std::vector<__float128> v(n, c);
      auto ct = cardinal_trigonometric<decltype(v)>(v, t0, h);
      CHECK_ULP_CLOSE(boost::multiprecision::float128(c), boost::multiprecision::float128(ct(0.3)), 3);
      CHECK_ULP_CLOSE(boost::multiprecision::float128(c*h*n), boost::multiprecision::float128(ct.integrate()), 3);
    }
}
#endif


template<class Real>
void test_sampled_sine()
{
    using std::sin;
    using std::cos;
    for (unsigned n = 15; n < 50; ++n)
    {
      Real t0 = 0;
      Real T = 1;
      Real h = T/n;
      std::vector<Real> v(n);
      auto s = [&](Real t) { return sin(two_pi<Real>()*(t-t0)/T);};
      auto s_prime = [&](Real t) { return two_pi<Real>()*cos(two_pi<Real>()*(t-t0)/T)/T;};
      auto s_double_prime = [&](Real t) { return -two_pi<Real>()*two_pi<Real>()*sin(two_pi<Real>()*(t-t0)/T)/(T*T);};
      for(size_t j = 0; j < v.size(); ++j)
      {
          Real t = t0 + j*h;
          v[j] = s(t);
      }
      auto ct = cardinal_trigonometric<decltype(v)>(v, t0, h);
      CHECK_ULP_CLOSE(T, ct.period(), 3);
      std::mt19937 gen(1234);
      std::uniform_real_distribution<Real> dist(0, 500);

      unsigned j = 0;
      while (j++ < 50) {
        Real arg = dist(gen);
        Real expected = s(arg);
        Real computed = ct(arg);
        CHECK_MOLLIFIED_CLOSE(expected, computed, std::numeric_limits<Real>::epsilon()*4000);

        expected = s_prime(arg);
        computed = ct.prime(arg);
        CHECK_MOLLIFIED_CLOSE(expected, computed, 18000*std::numeric_limits<Real>::epsilon());

        expected = s_double_prime(arg);
        computed = ct.double_prime(arg);
        CHECK_MOLLIFIED_CLOSE(expected, computed, 100000*std::numeric_limits<Real>::epsilon());

      }
      CHECK_MOLLIFIED_CLOSE(Real(0), ct.integrate(), std::numeric_limits<Real>::epsilon());
    }
}

template<class Real>
void test_bump()
{
  using std::exp;
  using std::abs;
  using std::sqrt;
  using std::pow;
  auto bump = [](Real x)->Real { if (abs(x) >= 1) { return Real(0); } return exp(-Real(1)/(Real(1)-x*x)); };
  auto bump_prime = [](Real x)->Real {
      if (abs(x) >= 1) { return Real(0); }

      return -2*x*exp(-Real(1)/(Real(1)-x*x))/pow(1-x*x,2);
  };

  auto bump_double_prime = [](Real x)->Real {
      if (abs(x) >= 1) { return Real(0); }

      return (6*pow(x,4)-2)*exp(-Real(1)/(Real(1)-x*x))/pow(1-x*x,4);
  };


  Real t0 = -1;
  size_t n = 4096;
  Real h = Real(2)/Real(n);

  std::vector<Real> v(n);
  for(size_t i = 0; i < n; ++i)
  {
      Real t = t0 + i*h;
      v[i] = bump(t);
  }

  auto ct = cardinal_trigonometric<decltype(v)>(v, t0, h);
  std::mt19937 gen(323723);
  std::uniform_real_distribution<long double> dis(-0.9, 0.9);

  size_t i = 0;
  while (i++ < 1000)
  {
      Real t = static_cast<Real>(dis(gen));
      Real expected = bump(t);
      Real computed = ct(t);
      if(!CHECK_MOLLIFIED_CLOSE(expected, computed, 2*std::numeric_limits<Real>::epsilon())) {
          std::cerr << "  Problem occurred at abscissa " << t << "\n";
      }

      expected = bump_prime(t);
      computed = ct.prime(t);
      if(!CHECK_MOLLIFIED_CLOSE(expected, computed, 4000*std::numeric_limits<Real>::epsilon())) {
          std::cerr << "  Problem occurred at abscissa " << t << "\n";
      }

      expected = bump_double_prime(t);
      computed = ct.double_prime(t);
      if(!CHECK_MOLLIFIED_CLOSE(expected, computed, 4000*4000*std::numeric_limits<Real>::epsilon())) {
          std::cerr << "  Problem occurred at abscissa " << t << "\n";
      }


  }

  // Wolfram Alpha:
  // NIntegrate[Exp[-1/(1-x*x)],{x,-1,1}]
  CHECK_ULP_CLOSE(Real(0.443993816168079437823L), ct.integrate(), 3);

  // NIntegrate[Exp[-2/(1-x*x)],{x,-1,1}]
  CHECK_ULP_CLOSE(Real(0.1330861208449942715569473279553285713625791551628130055345002588895389L), ct.squared_l2(), 1);


}


int main()
{

#ifdef TEST1
    test_constant<float>();
    test_sampled_sine<float>();
    test_bump<float>();
    test_interpolation_condition<float>();
#endif


#ifdef TEST2
    test_constant<double>();
    test_sampled_sine<double>();
    test_bump<double>();
    test_interpolation_condition<double>();
#endif

#ifdef TEST3
    test_constant<long double>();
    test_sampled_sine<long double>();
    test_bump<long double>();
    test_interpolation_condition<long double>();
#endif

#ifdef TEST4
#ifdef BOOST_HAS_FLOAT128
test_constant_q();
#endif
#endif

    return boost::math::test::report_errors();
}

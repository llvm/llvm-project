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
#include <boost/math/special_functions/cardinal_b_spline.hpp>
#include <boost/math/interpolators/detail/cubic_b_spline_detail.hpp>
#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
using boost::multiprecision::float128;
#endif

using std::abs;
using boost::math::cardinal_b_spline;
using boost::math::cardinal_b_spline_prime;
using boost::math::forward_cardinal_b_spline;
using boost::math::cardinal_b_spline_double_prime;


template<class Real>
void test_box()
{
    Real t = cardinal_b_spline<0>(Real(1.1));
    Real expected = 0;
    CHECK_ULP_CLOSE(expected, t, 0);
    CHECK_ULP_CLOSE(expected, cardinal_b_spline_prime<0>(Real(1.1)), 0);

    CHECK_EQUAL(cardinal_b_spline<0>(Real(0.5)), Real(0.5f));
    CHECK_EQUAL(cardinal_b_spline_prime<0>(Real(0.5)), std::numeric_limits<Real>::infinity());

    t = cardinal_b_spline<0>(Real(-1.1));
    expected = 0;
    CHECK_ULP_CLOSE(expected, t, 0);

    Real h = Real(1)/Real(256);
    for (t = -Real(1)/Real(2)+h; t < Real(1)/Real(2); t += h)
    {
        expected = 1;
        CHECK_ULP_CLOSE(expected, cardinal_b_spline<0>(t), 0);
        expected = 0;
        CHECK_ULP_CLOSE(expected, cardinal_b_spline_prime<0>(Real(1.1)), 0);
    }

    for (t = h; t < 1; t += h)
    {
        expected = 1;
        CHECK_ULP_CLOSE(expected, forward_cardinal_b_spline<0>(t), 0);
    }
}

template<class Real>
void test_hat()
{
    Real t = cardinal_b_spline<1>(Real(2.1));
    Real expected = 0;
    CHECK_ULP_CLOSE(expected, t, 0);

    t = cardinal_b_spline<1>(Real(-2.1));
    expected = 0;
    CHECK_ULP_CLOSE(expected, t, 0);

    Real h = Real(1)/Real(256);
    for (t = -1; t <= 1; t += h)
    {
        expected = 1-abs(t);
        if(!CHECK_ULP_CLOSE(expected, cardinal_b_spline<1>(t), 0) )
        {
            std::cerr << "  Problem at t = " << t << "\n";
        }
        if (t == -Real(1)) {
            if (!CHECK_ULP_CLOSE(Real(1)/Real(2), cardinal_b_spline_prime<1>(t), 0)) {
                std::cout << "  Problem at t = " << t << "\n";
            }
        }
        else if (t == Real(1)) {
            CHECK_ULP_CLOSE(-Real(1)/Real(2), cardinal_b_spline_prime<1>(t), 0);
        }
        else if (t < 0) {
            CHECK_ULP_CLOSE(Real(1), cardinal_b_spline_prime<1>(t), 0);
        }
        else if (t == 0) {
            CHECK_ULP_CLOSE(Real(0), cardinal_b_spline_prime<1>(t), 0);
        }
        else if (t > 0) {
            CHECK_ULP_CLOSE(Real(-1), cardinal_b_spline_prime<1>(t), 0);
        }
    }

    for (t = 0; t < 2; t += h)
    {
        expected = 1 - abs(t-1);
        CHECK_ULP_CLOSE(expected, forward_cardinal_b_spline<1>(t), 0);
    }
}

template<class Real>
void test_quadratic()
{
    using std::abs;
    auto b2 = [](Real x) {
        Real absx = abs(x);
        if (absx >= 3/Real(2)) {
            return Real(0);
        }
        if (absx >= 1/Real(2)) {
            Real t = absx - 3/Real(2);
            return t*t/2;
        }
        Real t1 = absx - 1/Real(2);
        Real t2 = absx + 1/Real(2);
        return (2-t1*t1 -t2*t2)/2;
    };

    auto b2_prime = [&](Real x)->Real {
        Real absx = abs(x);
        Real signx  = 1;
        if (x < 0) {
            signx = -1;
        }
        if (absx >= 3/Real(2)) {
            return Real(0);
        }
        if (absx >= 1/Real(2)) {
            return (absx - 3/Real(2))*signx;
        }
        return -2*absx*signx;
    };


    Real h = 1/Real(256);
    for (Real t = -5; t <= 5; t += h) {
        Real expected = b2(t);
        CHECK_ULP_CLOSE(expected, cardinal_b_spline<2>(t), 0);
        expected = b2_prime(t);

        if (!CHECK_ULP_CLOSE(expected, cardinal_b_spline_prime<2>(t), 0))
        {
            std::cerr << "  Problem at t = " << t << "\n";
        }

    }
}

template<class Real>
void test_cubic()
{
    Real expected = Real(2)/Real(3);
    Real computed = cardinal_b_spline<3, Real>(0);
    CHECK_ULP_CLOSE(expected, computed, 0);

    expected = Real(1)/Real(6);
    computed = cardinal_b_spline<3, Real>(1);
    CHECK_ULP_CLOSE(expected, computed, 0);

    expected = Real(0);
    computed = cardinal_b_spline<3, Real>(2);
    CHECK_ULP_CLOSE(expected, computed, 0);

    Real h = 1/Real(256);
    for (Real t = -4; t <= 4; t += h) {
        expected = boost::math::detail::b3_spline_prime<Real>(t);
        computed = cardinal_b_spline_prime<3>(t);
        CHECK_ULP_CLOSE(expected, computed, 0);
        expected = boost::math::detail::b3_spline_double_prime<Real>(t);
        computed = cardinal_b_spline_double_prime<3>(t);
        if (!CHECK_ULP_CLOSE(expected, computed, 0)) {
            std::cerr << "  Problem at t = " << t << "\n";
        }
    }
}

template<class Real>
void test_quintic()
{
  Real expected = Real(11)/Real(20);
  Real computed = cardinal_b_spline<5, Real>(0);
  CHECK_ULP_CLOSE(expected, computed, 0);

  expected = Real(13)/Real(60);
  computed = cardinal_b_spline<5, Real>(1);
  CHECK_ULP_CLOSE(expected, computed, 1);

  expected = Real(1)/Real(120);
  computed = cardinal_b_spline<5, Real>(2);
  CHECK_ULP_CLOSE(expected, computed, 0);

  expected = Real(0);
  computed = cardinal_b_spline<5, Real>(3);
  CHECK_ULP_CLOSE(expected, computed, 0);

}

template<unsigned n, typename Real>
void test_b_spline_derivatives()
{
    Real h = 1/Real(256);
    Real supp = (n+Real(1))/Real(2);
    for (Real t = -supp - 1; t <= supp+1; t+= h)
    {
        Real expected = cardinal_b_spline<n-1>(t+Real(1)/Real(2)) - cardinal_b_spline<n-1>(t - Real(1)/Real(2));
        Real computed = cardinal_b_spline_prime<n>(t);
        CHECK_MOLLIFIED_CLOSE(expected, computed, std::numeric_limits<Real>::epsilon());

        expected = cardinal_b_spline<n-2>(t+1) - 2*cardinal_b_spline<n-2>(t) + cardinal_b_spline<n-2>(t-1);
        computed = cardinal_b_spline_double_prime<n>(t);
        CHECK_MOLLIFIED_CLOSE(expected, computed, 2*std::numeric_limits<Real>::epsilon());
    }
}

template<unsigned n, typename Real>
void test_partition_of_unity()
{
  std::mt19937 gen(323723);
  Real supp = (n+1.0)/2.0;
  std::uniform_real_distribution<Real> dis(-supp, -supp+1);

  for(size_t i = 0; i < 500; ++i) {
    Real x = dis(gen);
    Real one = 0;
    while (x < supp) {
        one += cardinal_b_spline<n>(x);
        x += 1;
    }
    if(!CHECK_ULP_CLOSE(Real(1), one, n)) {
      std::cerr << "  Partition of unity failure at n = " << n << "\n";
    }
  }
}


int main()
{
    test_box<float>();
    test_box<double>();
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test_box<long double>();
#endif

    test_hat<float>();
    test_hat<double>();
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test_hat<long double>();
#endif

    test_quadratic<float>();
    test_quadratic<double>();
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test_quadratic<long double>();
#endif

    test_cubic<float>();
    test_cubic<double>();
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test_cubic<long double>();
#endif

    test_quintic<float>();
    test_quintic<double>();
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test_quintic<long double>();
#endif

    test_partition_of_unity<1, double>();
    test_partition_of_unity<2, double>();
    test_partition_of_unity<3, double>();
    test_partition_of_unity<4, double>();
    test_partition_of_unity<5, double>();
    test_partition_of_unity<6, double>();

    test_b_spline_derivatives<3, double>();
    test_b_spline_derivatives<4, double>();
    test_b_spline_derivatives<5, double>();
    test_b_spline_derivatives<6, double>();
    test_b_spline_derivatives<7, double>();
    test_b_spline_derivatives<8, double>();
    test_b_spline_derivatives<9, double>();

#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test_b_spline_derivatives<3, long double>();
    test_b_spline_derivatives<4, long double>();
    test_b_spline_derivatives<5, long double>();
    test_b_spline_derivatives<6, long double>();
    test_b_spline_derivatives<7, long double>();
    test_b_spline_derivatives<8, long double>();
    test_b_spline_derivatives<9, long double>();
#endif

#ifdef BOOST_HAS_FLOAT128
    test_box<float128>();
    test_hat<float128>();
    test_quadratic<float128>();
    test_cubic<float128>();
    test_quintic<float128>();
#endif

    return boost::math::test::report_errors();
}

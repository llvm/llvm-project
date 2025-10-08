//  (C) Copyright Kilian Kilger 2025.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/results_collector.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>

using namespace std;
using namespace boost::math;
using namespace boost::math::policies;
using namespace boost::multiprecision;

typedef policy<
   policies::domain_error<errno_on_error>,
   policies::pole_error<errno_on_error>,
   policies::overflow_error<errno_on_error>,
   policies::evaluation_error<errno_on_error>
> c_policy;

template<typename T>
struct test_lower
{
   T operator()(T a, T x) const
   {
      return tgamma_lower(a, x, c_policy());
   }

   T expected(T a) const
   {
      return T(0.0);
   }
};

template<typename T>
struct test_upper
{
   T operator()(T a, T x) const
   {
      return tgamma(a, x, c_policy());
   }
   T expected(T a) const
   {
      return tgamma(a, c_policy());
   }
};

template<typename T>
struct test_gamma_p
{
   T operator()(T a, T x) const
   {
      return gamma_p(a, x, c_policy());
   }
   T expected(T) const
   {
      return T(0.0);
   }
};

template<typename T>
struct test_gamma_q
{
   T operator()(T a, T x) const
   {
      return gamma_q(a, x, c_policy());
   }
   T expected(T) const
   {
      return T(1.0);
   }
};

template<typename T, template<typename> class Fun>
void test_impl(T a)
{
   Fun<T> fn;
   errno = 0;
   T x = T(0.0);
   T result = fn(a, x);
   int saveErrno = errno;

   errno = 0;

   T expected = fn.expected(a);

   BOOST_CHECK(errno == saveErrno);
   BOOST_CHECK_EQUAL(result, expected);
}

template<template<typename> class Fun, typename T>
void test_type_dispatch(T val)
{
   if (val <= (std::numeric_limits<float>::max)())
      test_impl<float, Fun>(static_cast<float>(val));
   if (val <= (std::numeric_limits<double>::max)())
      test_impl<double, Fun>(static_cast<double>(val));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   test_impl<long double, Fun>(static_cast<long double>(val));
#endif
   test_impl<cpp_bin_float_50, Fun>(static_cast<cpp_bin_float_50>(val));
}

template<template<typename> class Fun>
void test_impl()
{
   test_type_dispatch<Fun, double>(1.0);
   test_type_dispatch<Fun, double>(0.1);
   test_type_dispatch<Fun, double>(0.5);
   test_type_dispatch<Fun, double>(0.6);
   test_type_dispatch<Fun, double>(1.3);
   test_type_dispatch<Fun, double>(1.5);
   test_type_dispatch<Fun, double>(2);
   test_type_dispatch<Fun, double>(100);
   test_type_dispatch<Fun, double>((std::numeric_limits<float>::max)());
   test_type_dispatch<Fun, double>((std::numeric_limits<double>::max)());
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS   
   test_type_dispatch<Fun, long double>((std::numeric_limits<long double>::max)());
#endif
}

void test_derivative()
{
   using namespace boost::math::detail;
   double derivative = 0;
   double result = gamma_incomplete_imp(1.0, 0.0, true, false, c_policy(), &derivative);
   BOOST_CHECK(errno == 0);
   BOOST_CHECK_EQUAL(derivative, tools::max_value<double>() / 2);
   BOOST_CHECK_EQUAL(result, 0);
}

BOOST_AUTO_TEST_CASE( test_main )
{
   test_impl<test_lower>();
   test_impl<test_upper>();
   test_impl<test_gamma_p>();
   test_impl<test_gamma_q>();
   test_derivative();
}

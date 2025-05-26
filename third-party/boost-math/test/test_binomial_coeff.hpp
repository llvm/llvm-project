// Copyright John Maddock 2006.
// Copyright Paul A. Bristow 2007, 2009
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error
#include <boost/math/concepts/real_concept.hpp>
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/math/special_functions/trunc.hpp>
#include <boost/math/tools/test.hpp>
#include "functor.hpp"
#include <boost/array.hpp>
#include <iostream>
#include <iomanip>

#include "handle_test_result.hpp"
#include "table_type.hpp"

#ifndef SC_
#define SC_(x) static_cast<typename table_type<T>::type>(BOOST_JOIN(x, L))
#endif

template <class T>
T binomial_wrapper(T n, T k)
{
#ifdef BINOMIAL_FUNCTION_TO_TEST
   return BINOMIAL_FUNCTION_TO_TEST(
      boost::math::itrunc(n),
      boost::math::itrunc(k));
#else
   return boost::math::binomial_coefficient<T>(
      boost::math::itrunc(n),
      boost::math::itrunc(k));
#endif
}

template <class T>
void test_binomial(T, const char* type_name)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(BINOMIAL_FUNCTION_TO_TEST))
   using namespace std;

   typedef T(*func_t)(T, T);
#if defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   func_t f = &binomial_wrapper<T>;
#else
   func_t f = &binomial_wrapper;
#endif

#include "binomial_data.ipp"

   boost::math::tools::test_result<T> result = boost::math::tools::test_hetero<T>(
      binomial_data,
      bind_func<T>(f, 0, 1),
      extract_result<T>(2));

   std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
      "Test results for small arguments and type " << type_name << std::endl << std::endl;
   std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
   handle_test_result(result, binomial_data[result.worst()], result.worst(), type_name, "binomial_coefficient", "Binomials: small arguments");
   std::cout << std::endl;

#include "binomial_large_data.ipp"

   result = boost::math::tools::test_hetero<T>(
      binomial_large_data,
      bind_func<T>(f, 0, 1),
      extract_result<T>(2));

   std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
      "Test results for large arguments and type " << type_name << std::endl << std::endl;
   std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
   handle_test_result(result, binomial_large_data[result.worst()], result.worst(), type_name, "binomial_coefficient", "Binomials: large arguments");
   std::cout << std::endl;
#endif

   //
   // Additional tests for full coverage:
   //
   BOOST_CHECK_THROW(boost::math::binomial_coefficient<T>(2, 3), std::domain_error);
   T tolerance = boost::math::tools::epsilon<T>() * 200;
#if (LDBL_MAX_10_EXP > 320) || defined(TEST_MPF_50) || defined(TEST_MPFR_50) || defined(TEST_CPP_DEC_FLOAT) || defined(TEST_FLOAT128) || defined(TEST_CPP_BIN_FLOAT)
   BOOST_IF_CONSTEXPR(std::numeric_limits<T>::max_exponent10 > 320)
   {
      BOOST_IF_CONSTEXPR(std::is_floating_point<T>::value == false)
         tolerance *= 2;
      BOOST_CHECK_CLOSE_FRACTION(boost::math::binomial_coefficient<T>(1072, 522), SC_(8.5549524921358966076960008392254468438388716743112653656397e320), tolerance);
   }
   else BOOST_IF_CONSTEXPR(std::numeric_limits<T>::has_infinity)
   {
      BOOST_CHECK_EQUAL(boost::math::binomial_coefficient<T>(1072, 522), std::numeric_limits<T>::infinity());
   }
#else
   BOOST_IF_CONSTEXPR(std::numeric_limits<T>::has_infinity)
   {
      BOOST_CHECK_EQUAL(boost::math::binomial_coefficient<T>(1072, 522), std::numeric_limits<T>::infinity());
   }
#endif
#if (LDBL_MAX_10_EXP > 4946) || defined(TEST_MPF_50) || defined(TEST_MPFR_50) || defined(TEST_CPP_DEC_FLOAT) || defined(TEST_FLOAT128) || defined(TEST_CPP_BIN_FLOAT)

   BOOST_IF_CONSTEXPR(std::numeric_limits<T>::max_exponent10 > 4946)
   {
      if (!std::is_floating_point<T>::value)
         tolerance *= 15;
      BOOST_CHECK_CLOSE_FRACTION(boost::math::binomial_coefficient<T>(16441, 8151), SC_(5.928641856224322477306131563286843903129818155323061805272e4946), tolerance);
   }
   else BOOST_IF_CONSTEXPR(std::numeric_limits<T>::has_infinity && (std::numeric_limits<T>::max_exponent10 < 4950))
   {
      BOOST_CHECK_EQUAL(boost::math::binomial_coefficient<T>(16441, 8151), std::numeric_limits<T>::infinity());
   }
#else
   BOOST_IF_CONSTEXPR(std::numeric_limits<T>::has_infinity)
   {
      BOOST_CHECK_EQUAL(boost::math::binomial_coefficient<T>(16441, 8151), std::numeric_limits<T>::infinity());
   }
#endif

}


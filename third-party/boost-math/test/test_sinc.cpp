//  (C) Copyright John Maddock 2018.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pch_light.hpp>
#include "test_sinc.hpp"
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/math/special_functions/next.hpp>
#include <boost/math/special_functions/sinhc.hpp>

//
// DESCRIPTION:
// ~~~~~~~~~~~~
//
// This file tests the sinc_pi function.  There are two sets of tests, spot
// tests which compare our results with selected values computed
// using the online special function calculator at 
// functions.wolfram.com, while the bulk of the accuracy tests
// use values generated with NTL::RR at 1000-bit precision
// and our generic versions of these functions.
//
// Note that when this file is first run on a new platform many of
// these tests will fail: the default accuracy is 1 epsilon which
// is too tight for most platforms.  In this situation you will 
// need to cast a human eye over the error rates reported and make
// a judgement as to whether they are acceptable.  Either way please
// report the results to the Boost mailing list.  Acceptable rates of
// error are marked up below as a series of regular expressions that
// identify the compiler/stdlib/platform/data-type/test-data/test-function
// along with the maximum expected peek and RMS mean errors for that
// test.
//

void expected_results()
{
   //
   // Define the max and mean errors expected for
   // various compilers and platforms.
   //
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      ".*",                          // test type(s)
      ".*",                          // test data group
      ".*", 3, 3);                   // test function
   //
   // Finish off by printing out the compiler/stdlib/platform names,
   // we do this to make it easier to mark up expected error rates.
   //
   std::cout << "Tests run with " << BOOST_COMPILER << ", " 
      << BOOST_STDLIB << ", " << BOOST_PLATFORM << std::endl;
}

template <class T>
void test_close_to_transition()
{
   T transition = 3.3f * boost::math::tools::forth_root_epsilon<T>();
   T val = transition;

   for (unsigned i = 0; i < 100; ++i)
   {
      boost::multiprecision::cpp_bin_float_50 extended = val;
      extended = sin(extended) / extended;
      T expected = extended.template convert_to<T>();
      T result = boost::math::sinc_pi(val);
      BOOST_CHECK_LE(boost::math::epsilon_difference(result, expected), 3);
      result = boost::math::sinc_pi(-val);
      BOOST_CHECK_LE(boost::math::epsilon_difference(result, expected), 3);
      val = boost::math::float_prior(val);
   }
   for (unsigned i = 0; i < 100; ++i)
   {
      boost::multiprecision::cpp_bin_float_50 extended = val;
      extended = sin(extended) / extended;
      T expected = extended.template convert_to<T>();
      T result = boost::math::sinc_pi(val);
      BOOST_CHECK_LE(boost::math::epsilon_difference(result, expected), 3);
      result = boost::math::sinc_pi(-val);
      BOOST_CHECK_LE(boost::math::epsilon_difference(result, expected), 3);
      val = boost::math::float_next(val);
   }

   BOOST_IF_CONSTEXPR(std::numeric_limits<T>::has_infinity)
   {
      BOOST_CHECK_EQUAL(boost::math::sinc_pi(std::numeric_limits<T>::infinity()), T(0));
      BOOST_CHECK_EQUAL(boost::math::sinc_pi(-std::numeric_limits<T>::infinity()), T(0));
      BOOST_IF_CONSTEXPR(boost::math::policies::policy<>::overflow_error_type::value == boost::math::policies::throw_on_error)
      {
         BOOST_CHECK_THROW(boost::math::sinhc_pi(std::numeric_limits<T>::infinity()), std::overflow_error);
         BOOST_CHECK_THROW(boost::math::sinhc_pi(-std::numeric_limits<T>::infinity()), std::overflow_error);
      }
      else
      {
         BOOST_CHECK_EQUAL(boost::math::sinhc_pi(std::numeric_limits<T>::infinity()), std::numeric_limits<T>::infinity());
         BOOST_CHECK_EQUAL(boost::math::sinhc_pi(-std::numeric_limits<T>::infinity()), std::numeric_limits<T>::infinity());
      }
   }
   BOOST_IF_CONSTEXPR(boost::math::policies::policy<>::overflow_error_type::value == boost::math::policies::throw_on_error)
   {
      BOOST_CHECK_THROW(boost::math::sinhc_pi(boost::math::tools::max_value<T>()), std::overflow_error);
      BOOST_CHECK_THROW(boost::math::sinhc_pi(-boost::math::tools::max_value<T>()), std::overflow_error);
   }
   else BOOST_IF_CONSTEXPR(std::numeric_limits<T>::has_infinity)
   {
      BOOST_CHECK_EQUAL(boost::math::sinhc_pi(boost::math::tools::max_value<T>()), std::numeric_limits<T>::infinity());
      BOOST_CHECK_EQUAL(boost::math::sinhc_pi(-boost::math::tools::max_value<T>()), std::numeric_limits<T>::infinity());
   }
}


BOOST_AUTO_TEST_CASE( test_main )
{
   BOOST_MATH_CONTROL_FP;

   expected_results();

   test_sinc(0.1F, "float");
   test_close_to_transition<float>();
   test_sinc(0.1, "double");
   test_close_to_transition<double>();
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   test_sinc(0.1L, "long double");
   test_close_to_transition<long double>();
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
   test_sinc(boost::math::concepts::real_concept(0.1), "real_concept");
#endif
#else
   std::cout << "<note>The long double tests have been disabled on this platform "
      "either because the long double overloads of the usual math functions are "
      "not available at all, or because they are too inaccurate for these tests "
      "to pass.</note>" << std::endl;
#endif
}



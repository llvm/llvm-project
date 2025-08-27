//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef SYCL_LANGUAGE_VERSION
#include <pch_light.hpp>
#endif

#ifdef __clang__
#  pragma clang diagnostic push 
#  pragma clang diagnostic ignored "-Wliteral-range"
#elif defined(__GNUC__)
#  pragma GCC diagnostic push 
#  pragma GCC diagnostic ignored "-Woverflow"
#endif
#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error
#include "test_igamma_inv.hpp"

#if !defined(TEST_FLOAT) && !defined(TEST_DOUBLE) && !defined(TEST_LDOUBLE) && !defined(TEST_REAL_CONCEPT)
#  define TEST_FLOAT
#  define TEST_DOUBLE
#  define TEST_LDOUBLE
#  define TEST_REAL_CONCEPT
#endif

//
// DESCRIPTION:
// ~~~~~~~~~~~~
//
// This file tests the incomplete gamma function inverses 
// gamma_p_inv and gamma_q_inv. There are three sets of tests:
// 1) Spot tests which compare our results with selected values 
// computed using the online special function calculator at 
// functions.wolfram.com, 
// 2) Accuracy tests use values generated with NTL::RR at 
// 1000-bit precision and our generic versions of these functions.
// 3) Round trip sanity checks, use the test data for the forward
// functions, and verify that we can get (approximately) back
// where we started.
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
   const char* largest_type;
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   if(boost::math::policies::digits<double, boost::math::policies::policy<> >() == boost::math::policies::digits<long double, boost::math::policies::policy<> >())
   {
      largest_type = "(long\\s+)?double";
   }
   else
   {
      largest_type = "long double";
   }
#else
   largest_type = "(long\\s+)?double";
#endif
   //
   // Large exponent range causes more extreme test cases to be evaluated:
   //
   if(std::numeric_limits<long double>::max_exponent > std::numeric_limits<double>::max_exponent)
   {
      add_expected_result(
         "[^|]*",                          // compiler
         "[^|]*",                          // stdlib
         "[^|]*",                          // platform
         largest_type,                     // test type(s)
         "[^|]*small[^|]*",                    // test data group
         "[^|]*", 200000, 10000);              // test function
      add_expected_result(
         "[^|]*",                          // compiler
         "[^|]*",                          // stdlib
         "[^|]*",                          // platform
         "real_concept",                     // test type(s)
         "[^|]*small[^|]*",                   // test data group
         "[^|]*", 98000, 12000);                  // test function
   }
   //
   // These high error rates are seen on on some Linux
   // architectures:
   //
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "linux.*",                          // platform
      largest_type,                     // test type(s)
      "[^|]*medium[^|]*",                   // test data group
      #ifdef SYCL_LANGUAGE_VERSION
      "[^|]*", 350, 50);
      #else
      "[^|]*", 350, 5);                  // test function
      #endif
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "linux.*",                          // platform
      largest_type,                     // test type(s)
      "[^|]*large[^|]*",                   // test data group
      #ifdef SYCL_LANGUAGE_VERSION
      "[^|]*", 150, 20);                  // test function
      #else
      "[^|]*", 150, 5);                  // test function
      #endif


   //
   // Catch all cases come last:
   //
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      largest_type,                     // test type(s)
      "[^|]*medium[^|]*",                   // test data group
      "[^|]*", 20, 5);                  // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      largest_type,                     // test type(s)
      "[^|]*large[^|]*",                    // test data group
      "[^|]*", 5, 2);                   // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      largest_type,                     // test type(s)
      "[^|]*small[^|]*",                    // test data group
      "[^|]*", 2100, 500);              // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      "float|double",                   // test type(s)
      "[^|]*small[^|]*",                    // test data group
      "gamma_p_inv", 500, 60);   // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      "float|double",                   // test type(s)
      "[^|]*",                          // test data group
      "gamma_q_inv", 350, 60);   // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      "float|double",                   // test type(s)
      "[^|]*",                          // test data group
      "[^|]*", 4, 2);                   // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      "real_concept",                     // test type(s)
      "[^|]*medium[^|]*",                   // test data group
      "[^|]*", 20, 5);                  // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      "real_concept",                     // test type(s)
      "[^|]*large[^|]*",                   // test data group
      "[^|]*", 1000, 500);                  // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      "real_concept",                     // test type(s)
      "[^|]*small[^|]*",                   // test data group
      "[^|]*", 3700, 500);                  // test function

   //
   // Finish off by printing out the compiler/stdlib/platform names,
   // we do this to make it easier to mark up expected error rates.
   //
   std::cout << "Tests run with " << BOOST_COMPILER << ", " 
      << BOOST_STDLIB << ", " << BOOST_PLATFORM << std::endl;
}

BOOST_AUTO_TEST_CASE( test_main )
{
   expected_results();
   BOOST_MATH_CONTROL_FP;

#ifndef BOOST_MATH_BUGGY_LARGE_FLOAT_CONSTANTS
#ifdef TEST_FLOAT
   test_spots(0.0F, "float");
#endif
#endif
#ifdef TEST_DOUBLE
   test_spots(0.0, "double");
#endif
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
#ifdef TEST_LDOUBLE
   test_spots(0.0L, "long double");
#endif
#if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x582))
#if defined(TEST_REAL_CONCEPT) && !defined(BOOST_MATH_NO_REAL_CONCEPT_TESTS)
   test_spots(boost::math::concepts::real_concept(0.1), "real_concept");
#endif
#endif
#endif

#ifndef BOOST_MATH_BUGGY_LARGE_FLOAT_CONSTANTS
#ifdef TEST_FLOAT
   test_gamma(0.1F, "float");
#endif
#endif
#ifdef TEST_DOUBLE
   test_gamma(0.1, "double");
#endif
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
#ifdef TEST_LDOUBLE
   test_gamma(0.1L, "long double");
#endif
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
#if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x582))
#if defined(TEST_REAL_CONCEPT) && !defined(BOOST_MATH_NO_REAL_CONCEPT_TESTS)
   test_gamma(boost::math::concepts::real_concept(0.1), "real_concept");
#endif
#endif
#endif
#else
   std::cout << "<note>The long double tests have been disabled on this platform "
      "either because the long double overloads of the usual math functions are "
      "not available at all, or because they are too inaccurate for these tests "
      "to pass.</note>" << std::endl;
#endif
   
}




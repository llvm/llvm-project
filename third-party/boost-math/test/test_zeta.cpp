//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "pch_light.hpp"
#ifndef BOOST_MATH_OVERFLOW_ERROR_POLICY
#  define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error
#endif
#include "test_zeta.hpp"

//
// DESCRIPTION:
// ~~~~~~~~~~~~
//
// This file tests the zeta function. There are two sets of tests:
// 1) Sanity checks: comparison to test values created with the
// online calculator at functions.wolfram.com
// 2) Accuracy tests use values generated with NTL::RR at 
// 1000-bit precision and our generic versions of these functions.
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

   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      largest_type,                  // test type(s)
      ".*Random values less than 1", // test data group
      ".*", 1200, 500);              // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "real_concept",                // test type(s)
      ".*Random values less than 1", // test data group
      ".*", 1200, 500);              // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      largest_type,                  // test type(s)
      ".*Integer.*",                 // test data group
      ".*", 30, 15);                 // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      largest_type,                  // test type(s)
      ".*",                          // test data group
      ".*", 3, 3);                   // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*Solaris.*",                 // platform
      "real_concept",                // test type(s)
      ".*",                          // test data group
      ".*", 60, 15);                  // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "real_concept",                // test type(s)
      ".*",                          // test data group
      ".*", 40, 15);                  // test function

   std::cout << "Tests run with " << BOOST_COMPILER << ", " 
      << BOOST_STDLIB << ", " << BOOST_PLATFORM << std::endl;
}

BOOST_AUTO_TEST_CASE( test_main )
{
   expected_results();
   BOOST_MATH_CONTROL_FP;

   test_spots(0.0f, "float");
   test_spots(0.0, "double");
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   test_spots(0.0L, "long double");
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
   test_spots(boost::math::concepts::real_concept(0.1), "real_concept");
#endif
#else
   std::cout << "<note>The long double tests have been disabled on this platform "
      "either because the long double overloads of the usual math functions are "
      "not available at all, or because they are too inaccurate for these tests "
      "to pass.</note>" << std::endl;
#endif

   test_zeta(0.1F, "float");
   test_zeta(0.1, "double");
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   test_zeta(0.1L, "long double");
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
   test_zeta(boost::math::concepts::real_concept(0.1), "real_concept");
#endif
#else
   std::cout << "<note>The long double tests have been disabled on this platform "
      "either because the long double overloads of the usual math functions are "
      "not available at all, or because they are too inaccurate for these tests "
      "to pass.</note>" << std::endl;
#endif
   
}




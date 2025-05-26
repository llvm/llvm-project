//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pch_light.hpp>
#include "test_tgamma_ratio.hpp"

//
// DESCRIPTION:
// ~~~~~~~~~~~~
//
// This file tests the gamma ratio functions tgamma_ratio,
// and tgamma_delta_ratio. The accuracy tests
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
   // HP-UX
   // This is a weird one, HP-UX and Mac OS X show up errors at float
   // precision, that don't show up on other platforms.
   // There appears to be some kind of rounding issue going on (not enough
   // precision in the input to get the answer right):
   //
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "HP-UX|Mac OS|linux|.*(bsd|BSD).*",      // platform
      "float",                          // test type(s)
      "[^|]*",                          // test data group
      "tgamma_ratio[^|]*", 35, 8);                 // test function
   //
   // Linux AMD x86em64 has slightly higher rates:
   //
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "linux.*",                          // platform
      largest_type,                     // test type(s)
      "[^|]*",               // test data group
      "tgamma_ratio[^|]*", 300, 100);                 // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "linux.*",                          // platform
      "real_concept",                     // test type(s)
      "[^|]*",               // test data group
      "tgamma_ratio[^|]*", 300, 100);                 // test function

   add_expected_result(
      "GNU.*",                          // compiler
      "[^|]*",                          // stdlib
      "Win32.*",                          // platform
      largest_type,                     // test type(s)
      "[^|]*",               // test data group
      "tgamma_ratio[^|]*", 300, 100);                 // test function
   //
   // Cygwin
   //
   add_expected_result(
      "GNU.*",                          // compiler
      "[^|]*",                          // stdlib
      "Cygwin*",                        // platform
      largest_type,                     // test type(s)
      "[^|]*",                          // test data group
      "tgamma_ratio[^|]*", 300, 100);   // test function
   //
   // Solaris:
   //
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      ".*Solaris.*",                    // platform
      largest_type,                     // test type(s)
      "[^|]*",               // test data group
      "tgamma_ratio[^|]*", 200, 100);                 // test function
   //
   // Catch all cases come last:
   //
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      largest_type,                     // test type(s)
      "[^|]*",                          // test data group
      "tgamma_delta_ratio[^|]*", 30, 20);                 // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      largest_type,                     // test type(s)
      "[^|]*",               // test data group
      "tgamma_ratio[^|]*", 100, 50);                 // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      "real_concept",                   // test type(s)
      "[^|]*",                          // test data group
      "tgamma_delta_ratio[^|]*", 50, 20); // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      "real_concept",                   // test type(s)
      "[^|]*",               // test data group
      "[^|]*", 250, 150);                 // test function

   //
   // Finish off by printing out the compiler/stdlib/platform names,
   // we do this to make it easier to mark up expected error rates.
   //
   std::cout << "Tests run with " << BOOST_COMPILER << ", " 
      << BOOST_STDLIB << ", " << BOOST_PLATFORM << std::endl;
}

BOOST_AUTO_TEST_CASE( test_main )
{
   BOOST_MATH_CONTROL_FP;
   expected_results();

#ifndef BOOST_MATH_BUGGY_LARGE_FLOAT_CONSTANTS
   test_tgamma_ratio(0.1F, "float");
#endif
   test_tgamma_ratio(0.1, "double");
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   test_tgamma_ratio(0.1L, "long double");
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
#if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x582)) && !defined(BOOST_MATH_NO_REAL_CONCEPT_TESTS)
   test_tgamma_ratio(boost::math::concepts::real_concept(0.1), "real_concept");
#endif
#endif
#else
   std::cout << "<note>The long double tests have been disabled on this platform "
      "either because the long double overloads of the usual math functions are "
      "not available at all, or because they are too inaccurate for these tests "
      "to pass.</note>" << std::endl;
#endif
   
#ifndef BOOST_MATH_BUGGY_LARGE_FLOAT_CONSTANTS
   test_spots(0.1F, "float");
#endif
   test_spots(0.1, "double");
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   test_spots(0.1L, "long double");
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
#if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x582))
   test_spots(boost::math::concepts::real_concept(0.1), "real_concept");
#endif
#endif
#else
   std::cout << "<note>The long double tests have been disabled on this platform "
      "either because the long double overloads of the usual math functions are "
      "not available at all, or because they are too inaccurate for these tests "
      "to pass.</note>" << std::endl;
#endif
}



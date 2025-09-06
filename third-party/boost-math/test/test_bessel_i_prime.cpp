//  Copyright (c) 2013 Anton Bikineev
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "pch_light.hpp"
#include "test_bessel_i_prime.hpp"

//
// DESCRIPTION:
// ~~~~~~~~~~~~
//
// This file tests the bessel I functions derivatives.  There are two sets of tests, spot
// tests which compare our results with selected values computed
// using the online special function calculator at 
// functions.wolfram.com, while the bulk of the accuracy tests
// use values generated with Boost.Multiprecision at 50 precision
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
   // Mac OS has higher error rates, why?
   //
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      "Mac OS",                      // platform
      largest_type,                  // test type(s)
      ".*",                          // test data group
#ifdef __aarch64__
      // Error rates for M1 macs are higher than x86_64 macs
      ".*", 4000, 1500);               // test function
#else
      ".*", 3500, 1500);               // test function
#endif
   //
   // Cygwin:
   //
   add_expected_result(
      "GNU.*",                      // Compiler
      ".*",                         // Stdlib
      "Cygwin*",                    // Platform
      largest_type,                 // test type(s)
      ".*",                         // test data group
      ".*", 3500, 1000);            // test function
   //
   // G++ on Linux, results vary a bit by processor type,
   // on Itanium results are *much* better than listed here,
   // but x86 appears to have much less accurate std::pow
   // that throws off the results for tgamma(long double)
   // which then impacts on the Bessel functions:
   //
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      "linux",                       // platform
      largest_type,                  // test type(s)
      ".*Random.*",                  // test data group
      ".*", 600, 200);               // test function

   add_expected_result(
      "GNU.*",                       // compiler
      ".*",                          // stdlib
      "Win32.*",                     // platform
      largest_type,                  // test type(s)
      ".*Random.*",                  // test data group
      ".*", 500, 200);               // test function
   add_expected_result(
      "GNU.*",                       // compiler
      ".*",                          // stdlib
      "Win32.*",                     // platform
      largest_type,                  // test type(s)
      ".*",                          // test data group
      ".*", 3500, 1000);                 // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      largest_type,                  // test type(s)
      ".*I'v.*Mathworld.*",          // test data group
      ".*", 4200, 2000);             // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*Solaris.*",                 // platform
      largest_type,                  // test type(s)
      ".*",                          // test data group
      ".*", 900, 300);               // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      largest_type,                  // test type(s)
      ".*",                          // test data group
      ".*", 40, 25);                 // test function
   //
   // Set error rates a little higher for real_concept - 
   // now that we use a series approximation for small z
   // that relies on tgamma the error rates are a little
   // higher only when a Lanczos approximation is not available.
   // All other types are unaffected.
   //
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "real_concept",                // test type(s)
      ".*I'v.*Mathworld.*",          // test data group
      ".*", 4500, 2000);             // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*Solaris.*",                          // platform
      "real_concept",                // test type(s)
      ".*",                          // test data group
      ".*", 800, 400);               // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "real_concept",                // test type(s)
      ".*",                          // test data group
      ".*", 600, 200);               // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "double",                      // test type(s)
      ".*",                          // test data group
      ".*", 2, 1);                   // test function
   //
   // Finish off by printing out the compiler/stdlib/platform names,
   // we do this to make it easier to mark up expected error rates.
   //
   std::cout << "Tests run with " << BOOST_COMPILER << ", " 
      << BOOST_STDLIB << ", " << BOOST_PLATFORM << std::endl;
}

BOOST_AUTO_TEST_CASE( test_main )
{
#ifdef TEST_GSL
   gsl_set_error_handler_off();
#endif
   expected_results();
   BOOST_MATH_CONTROL_FP;

#ifndef BOOST_MATH_BUGGY_LARGE_FLOAT_CONSTANTS
   test_bessel(0.1F, "float");
#endif
   test_bessel(0.1, "double");
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   test_bessel(0.1L, "long double");
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
   test_bessel(boost::math::concepts::real_concept(0.1), "real_concept");
#endif
#else
   std::cout << "<note>The long double tests have been disabled on this platform "
      "either because the long double overloads of the usual math functions are "
      "not available at all, or because they are too inaccurate for these tests "
      "to pass.</note>" << std::endl;
#endif
}





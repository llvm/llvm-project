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

#include "test_ibeta.hpp"

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
// This file tests the incomplete beta functions beta, 
// betac, ibeta and ibetac.  There are two sets of tests, spot
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
   // Darwin: just one special case for real_concept:
   //
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "Mac OS",                          // platform
      "real_concept",                   // test type(s)
      "(?i).*large.*",                      // test data group
      ".*", 400000, 50000);             // test function

   //
   // Linux - results depend quite a bit on the
   // processor type, and how good the std::pow
   // function is for that processor.
   //
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "linux",                          // platform
      largest_type,                     // test type(s)
      "(?i).*small.*",                  // test data group
      ".*", 350, 100);  // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "linux",                          // platform
      largest_type,                     // test type(s)
      "(?i).*medium.*",                     // test data group
      ".*", 300, 80);  // test function
   //
   // Deficiencies in pow function really kick in here for
   // large arguments.  Note also that the tests here get
   // *very* extreme due to the increased exponent range
   // of 80-bit long doubles.  Also effect Mac OS.
   //
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "linux|Mac OS",                          // platform
      largest_type,                     // test type(s)
      "(?i).*large.*",                      // test data group
      ".*", 200000, 10000);                 // test function
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "linux|Mac OS|Sun.*",             // platform
      "double",                     // test type(s)
      "(?i).*large.*",                      // test data group
      ".*", 40, 20);                 // test function
#endif
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "linux|Mac OS",                          // platform
      "real_concept",                   // test type(s)
      "(?i).*medium.*",                 // test data group
      ".*", 350, 100);  // test function

   //
   // HP-UX:
   //
   // Large value tests include some with *very* extreme
   // results, thanks to the large exponent range of
   // 128-bit long doubles.
   //
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "HP-UX",                          // platform
      largest_type,                     // test type(s)
      "(?i).*large.*",                      // test data group
      ".*", 200000, 10000);                 // test function
   //
   // Tru64:
   //
   add_expected_result(
      ".*Tru64.*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      largest_type,                     // test type(s)
      "(?i).*large.*",                      // test data group
      ".*", 130000, 10000);                 // test function
   //
   // Sun OS:
   //
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "Sun.*",                          // platform
      largest_type,                     // test type(s)
      "(?i).*large.*",                      // test data group
      ".*", 130000, 10000);                 // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "Sun.*",                          // platform
      largest_type,                     // test type(s)
      "(?i).*small.*",                      // test data group
      ".*", 130, 30);                 // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "Sun.*",                          // platform
      largest_type,                     // test type(s)
      "(?i).*medium.*",                 // test data group
      ".*", 250, 40);                   // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "Sun.*",                          // platform
      "real_concept",                   // test type(s)
      "(?i).*medium.*",                 // test data group
      ".*", 250, 40);                   // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "Sun.*",                          // platform
      "real_concept",                     // test type(s)
      "(?i).*small.*",                      // test data group
      ".*", 130, 30);                 // test function
   //
   // MinGW:
   //
   add_expected_result(
      "GNU[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "Win32[^|]*",                          // platform
      "real_concept",                   // test type(s)
      "(?i).*medium.*",                     // test data group
      ".*", 400, 50);  // test function
   add_expected_result(
      "GNU.*",                          // compiler
      ".*",                          // stdlib
      "Win32.*",                          // platform
      "double",                     // test type(s)
      "(?i).*large.*",                      // test data group
      ".*", 20, 10);                 // test function
   add_expected_result(
      "GNU.*",                          // compiler
      ".*",                          // stdlib
      "Win32.*",                          // platform
      largest_type,                     // test type(s)
      "(?i).*large.*",                      // test data group
      ".*", 200000, 10000);                 // test function
   //
   // Cygwin:
   //
   add_expected_result(
      "GNU[^|]*",                       // compiler
      "[^|]*",                          // stdlib
      "Cygwin*",                        // platform
      "real_concept",                   // test type(s)
      "(?i).*medium.*",                 // test data group
      ".*", 400, 50);                   // test function
   add_expected_result(
      "GNU.*",                          // compiler
      ".*",                             // stdlib
      "Cygwin*",                        // platform
      "double",                         // test type(s)
      "(?i).*large.*",                  // test data group
      ".*", 20, 10);                    // test function
   add_expected_result(
      "GNU.*",                          // compiler
      ".*",                             // stdlib
      "Cygwin*",                        // platform
      largest_type,                     // test type(s)
      "(?i).*large.*",                  // test data group
      ".*", 200000, 10000);             // test function

#ifdef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   //
   // No long doubles:
   //
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      BOOST_PLATFORM,                          // platform
      largest_type,                     // test type(s)
      "(?i).*large.*",                      // test data group
      ".*", 13000, 500);                 // test function
#endif
   //
   // Catch all cases come last:
   //
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      largest_type,                     // test type(s)
      "(?i).*small.*",                  // test data group
      ".*", 90, 25);  // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      largest_type,                     // test type(s)
      "(?i).*medium.*",                     // test data group
      ".*", 350, 50);  // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      largest_type,                     // test type(s)
      "(?i).*large.*",                      // test data group
      ".*", 5000, 500);                 // test function

   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      "real_concept",                   // test type(s)
      "(?i).*small.*",                      // test data group
      ".*", 90, 25);  // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      "real_concept",                   // test type(s)
      "(?i).*medium.*",                     // test data group
      ".*", 250, 50);  // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      "real_concept",                   // test type(s)
      "(?i).*large.*",                      // test data group
      ".*", 200000, 50000);             // test function

   // catch all default is 2eps for all types:
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      "[^|]*",                          // test type(s)
      "[^|]*",                          // test data group
      ".*", 2, 2);                      // test function
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
#ifdef TEST_GSL
   gsl_set_error_handler_off();
#endif
#ifdef TEST_FLOAT
   test_spots(0.0F);
#endif
#ifdef TEST_DOUBLE
   test_spots(0.0);
#endif
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
#ifdef TEST_LDOUBLE
   test_spots(0.0L);
#endif
#if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x582))
#if defined(TEST_REAL_CONCEPT) && !defined(BOOST_MATH_NO_REAL_CONCEPT_TESTS)
   test_spots(boost::math::concepts::real_concept(0.1));
#endif
#endif
#endif

#ifdef TEST_FLOAT
   test_beta(0.1F, "float");
#endif
#ifdef TEST_DOUBLE
   test_beta(0.1, "double");
#endif
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
#ifdef TEST_LDOUBLE
   test_beta(0.1L, "long double");
#endif
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
#ifdef TEST_REAL_CONCEPT
#if LDBL_MANT_DIG != 113
   //
   // TODO: why does this fail when we have a 128-bit long double
   // even though the regular long double tests pass?
   // Most likely there is a hidden issue in real_concept somewhere...
   //
   test_beta(boost::math::concepts::real_concept(0.1), "real_concept");
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






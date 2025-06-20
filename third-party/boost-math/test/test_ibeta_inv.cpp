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

#include"test_ibeta_inv.hpp"

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
// This file tests the incomplete beta function inverses 
// ibeta_inv and ibetac_inv. There are three sets of tests:
// 1) Spot tests which compare our results with selected values 
// computed using the online special function calculator at 
// functions.wolfram.com, 
// 2) TODO!!!! Accuracy tests use values generated with NTL::RR at 
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
   // Note that permitted max errors are really pretty high
   // at around 10000eps.  The reason for this is that even 
   // if the forward function is off by 1eps, it's enough to
   // throw out the inverse by ~7000eps.  In other words the
   // forward function may flatline, so that many x-values
   // all map to about the same p.  Trying to invert in this
   // region is almost futile.
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

#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   //
   // Linux etc,
   // Extended exponent range of long double
   // causes more extreme test cases to be executed:
   //
   if(std::numeric_limits<long double>::digits == 64)
   {
      add_expected_result(
         ".*",                          // compiler
         ".*",                          // stdlib
         ".*",                       // platform
         "double",                      // test type(s)
         ".*",                          // test data group
         ".*", 20, 10);            // test function
      add_expected_result(
         ".*",                          // compiler
         ".*",                          // stdlib
         ".*",                       // platform
         "long double",                      // test type(s)
         ".*",                          // test data group
         ".*", 200000, 100000);            // test function
      add_expected_result(
         ".*",                          // compiler
         ".*",                          // stdlib
         ".*",                          // platform
         "real_concept",                // test type(s)
         ".*",                          // test data group
         ".*", 5000000L, 500000);         // test function
   }
#endif
   //
   // MinGW,
   // Extended exponent range of long double
   // causes more extreme test cases to be executed:
   //
   add_expected_result(
      "GNU.*",                          // compiler
      ".*",                          // stdlib
      "Win32.*",                          // platform
      "double",                // test type(s)
      ".*",                          // test data group
      ".*", 10, 10);         // test function
   add_expected_result(
      "GNU.*",                          // compiler
      ".*",                          // stdlib
      "Win32.*",                          // platform
      largest_type,                // test type(s)
      ".*",                          // test data group
      ".*", 300000, 20000);         // test function

   //
   // HP-UX and Solaris:
   // Extended exponent range of long double
   // causes more extreme test cases to be executed:
   //
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      "HP-UX|Sun Solaris",           // platform
      "long double",                 // test type(s)
      ".*",                          // test data group
      ".*", 200000, 100000);         // test function

   //
   // HP Tru64:
   // Extended exponent range of long double
   // causes more extreme test cases to be executed:
   //
   add_expected_result(
      "HP Tru64.*",                  // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "long double",                 // test type(s)
      ".*",                          // test data group
      ".*", 200000, 100000);         // test function

   //
   // Catch all cases come last:
   //
   if (std::numeric_limits<long double>::digits > 100)
   {
      add_expected_result(
         ".*",                          // compiler
         ".*",                          // stdlib
         ".*",                          // platform
         largest_type,                  // test type(s)
         ".*",                          // test data group
         ".*", 200000, 5000);            // test function
   }
   else
   {
      add_expected_result(
         ".*",                          // compiler
         ".*",                          // stdlib
         ".*",                          // platform
         largest_type,                  // test type(s)
         ".*",                          // test data group
         ".*", 10000, 1000);            // test function
   }
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "real_concept",                // test type(s)
      ".*",                          // test data group
      ".*", 500000, 500000);         // test function

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
#if defined(TEST_REAL_CONCEPT) && !defined(BOOST_MATH_NO_REAL_CONCEPT_TESTS)
   test_spots(boost::math::concepts::real_concept(0.1));
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
   test_beta(boost::math::concepts::real_concept(0.1), "real_concept");
#endif
#endif
#else
   std::cout << "<note>The long double tests have been disabled on this platform "
      "either because the long double overloads of the usual math functions are "
      "not available at all, or because they are too inaccurate for these tests "
      "to pass.</note>" << std::endl;
#endif
   
}





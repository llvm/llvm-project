//  (C) Copyright John Maddock 2007.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef SYCL_LANGUAGE_VERSION
#include <pch_light.hpp>
#else
#define BOOST_MATH_PROMOTE_DOUBLE_POLICY false
#include <boost/math/tools/config.hpp>
#endif

#ifdef __clang__
#  pragma clang diagnostic push 
#  pragma clang diagnostic ignored "-Wliteral-range"
#elif defined(__GNUC__)
#  pragma GCC diagnostic push 
#  pragma GCC diagnostic ignored "-Woverflow"
#endif

#include "test_bessel_y.hpp"

//
// DESCRIPTION:
// ~~~~~~~~~~~~
//
// This file tests the bessel Y function.  There are two sets of tests, spot
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
      largest_type = "(long\\s+)?double|real_concept";
   }
   else
   {
      largest_type = "long double|real_concept";
   }
#else
   largest_type = "(long\\s+)?double";
#endif

   //
   // HP-UX and Solaris rates are very slightly higher:
   //
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      "HP-UX|Sun Solaris",                          // platform
      largest_type,                // test type(s)
      ".*(Y[nv]|y).*Random.*",           // test data group
      ".*", 30000, 30000);             // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      "HP-UX|Sun Solaris",                          // platform
      largest_type,                  // test type(s)
      ".*Y[01Nv].*",           // test data group
      ".*", 1300, 500);               // test function
   //
   // Tru64:
   //
   add_expected_result(
      ".*Tru64.*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      largest_type,                // test type(s)
      ".*(Y[nv]|y).*Random.*",           // test data group
      ".*", 30000, 30000);             // test function
   add_expected_result(
      ".*Tru64.*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      largest_type,                      // test type(s)
      ".*Y[01Nv].*",           // test data group
      ".*", 400, 200);               // test function

   //
   // Mac OS X rates are very slightly higher:
   //
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      "Mac OS",                          // platform
      largest_type,                // test type(s)
      ".*(Y[nv1]).*",           // test data group
      ".*", 600000, 100000);             // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      "Mac OS",                          // platform
      "long double|real_concept",        // test type(s)
      ".*Y[0].*",           // test data group
      ".*", 1200, 1000);               // test function

   //
   // Linux:
   //
      add_expected_result(
         ".*",                          // compiler
         ".*",                          // stdlib
         "linux",                          // platform
         largest_type,                  // test type(s)
         ".*Yv.*Random.*",              // test data group
         ".*", 200000, 200000);         // test function
      add_expected_result(
         ".*",                          // compiler
         ".*",                          // stdlib
         "linux",                          // platform
         largest_type,                  // test type(s)
         ".*Y[01v].*",              // test data group
         ".*", 2000, 1000);         // test function
      add_expected_result(
         ".*",                          // compiler
         ".*",                          // stdlib
         "linux",                          // platform
         largest_type,                  // test type(s)
         ".*Yn.*",              // test data group
         ".*", 30000, 30000);         // test function
   //
   // MinGW:
   //
      add_expected_result(
         "GNU.*",                          // compiler
         ".*",                          // stdlib
         "Win32.*",                          // platform
         largest_type,                  // test type(s)
         ".*Yv.*Random.*",              // test data group
         ".*", 200000, 200000);         // test function
      add_expected_result(
         "GNU.*",                          // compiler
         ".*",                          // stdlib
         "Win32.*",                          // platform
         largest_type,                  // test type(s)
         ".*Y[01v].*",              // test data group
         ".*", 2000, 1000);         // test function
      add_expected_result(
         "GNU.*",                          // compiler
         ".*",                          // stdlib
         "Win32.*",                          // platform
         largest_type,                  // test type(s)
         ".*Yn.*",              // test data group
         ".*", 30000, 30000);         // test function
   //
   // Cygwin:
   //
      add_expected_result(
         "GNU.*",                       // compiler
         ".*",                          // stdlib
         "Cygwin*",                     // platform
         largest_type,                  // test type(s)
         ".*Yv.*Random.*",              // test data group
         ".*", 400000, 300000);         // test function
      add_expected_result(
         "GNU.*",                       // compiler
         ".*",                          // stdlib
         "Cygwin*",                     // platform
         largest_type,                  // test type(s)
         ".*Y[01v].*",                  // test data group
         ".*", 2000, 1000);             // test function
      add_expected_result(
         "GNU.*",                       // compiler
         ".*",                          // stdlib
         "Cygwin*",                     // platform
         largest_type,                  // test type(s)
         ".*Yn.*",                      // test data group
         ".*", 30000, 30000);           // test function
   //
   // Solaris version of long double has it's own error rates,
   // again just a touch higher than msvc's 64-bit double:
   //
   add_expected_result(
      "GNU.*",                          // compiler
      ".*",                          // stdlib
      "Sun.*",                          // platform
      largest_type,                  // test type(s)
      "Y[0N].*Mathworld.*",              // test data group
      ".*", 2000, 2000);         // test function

#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   BOOST_IF_CONSTEXPR((std::numeric_limits<double>::digits != std::numeric_limits<long double>::digits)
      && (std::numeric_limits<long double>::digits < 90))
   {
      // some errors spill over into type double as well:
      add_expected_result(
         ".*",                          // compiler
         ".*",                          // stdlib
         ".*",                          // platform
         "double",                      // test type(s)
         ".*Y[Nn].*",              // test data group
         ".*", 20, 20);         // test function
      add_expected_result(
         ".*",                          // compiler
         ".*",                          // stdlib
         ".*",                          // platform
         "double",                      // test type(s)
         ".*Yv.*",              // test data group
         ".*", 80, 70);         // test function
   }
   else BOOST_IF_CONSTEXPR(std::numeric_limits<long double>::digits >= 90)
   {
      add_expected_result(
         ".*",                          // compiler
         ".*",                          // stdlib
         ".*",                          // platform
         "double",                      // test type(s)
         ".*Yv.*Mathworld.*",           // test data group
         ".*", 20, 5);         // test function
   }
#endif
   //
   // defaults are based on MSVC-8 on Win32:
   //
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "real_concept",                // test type(s)
      ".*(Y[nv]|y).*Random.*",           // test data group
      ".*", 2000, 2000);             // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      largest_type,                  // test type(s)
      ".*(Y[nv]|y).*Random.*",           // test data group
      #ifdef SYCL_LANGUAGE_VERSION
      ".*", 2000, 1000);
      #else
      ".*", 1500, 1000);               // test function
      #endif
   //
   // Fallback for sun has to go after the general cases above:
   //
   add_expected_result(
      "GNU.*",                          // compiler
      ".*",                          // stdlib
      "Sun.*",                          // platform
      largest_type,                  // test type(s)
      "Y[0N].*",              // test data group
      ".*", 200, 200);         // test function
   //
   // General fallback:
   //
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      largest_type,                  // test type(s)
      ".*",                          // test data group
      ".*", 80, 40);                 // test function
   //
   // One set of float tests has inexact input values, so there is a slight error:
   //
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "float",                       // test type(s)
      "Yv: Mathworld Data",    // test data group
      ".*", 20, 20);                 // test function
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





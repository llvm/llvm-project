//  Copyright John Maddock 2006.
//  Copyright Paul A. Bristow 2007
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
#include <boost/math/special_functions/erf.hpp>
#include "test_erf.hpp"

//
// DESCRIPTION:
// ~~~~~~~~~~~~
//
// This file tests the functions erf, erfc, and the inverses
// erf_inv and erfc_inv.  There are two sets of tests, spot
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
   // On MacOS X erfc has much higher error levels than
   // expected: given that the implementation is basically
   // just a rational function evaluation combined with
   // exponentiation, we conclude that exp and pow are less
   // accurate on this platform, especially when the result 
   // is outside the range of a double.
   //
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      "Mac OS",                      // platform
      largest_type,                  // test type(s)
      "Erf Function:.*Large.*",      // test data group
      "erfc", 4300, 1300);  // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      "Mac OS",                      // platform
      largest_type,                  // test type(s)
      "Erf Function:.*",             // test data group
      "erfc", 40, 10);  // test function

   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "real_concept",                // test type(s)
      "Erf Function:.*",             // test data group
      "erfc?", 20, 6);   // test function
   add_expected_result(
      ".*",                           // compiler
      ".*",                           // stdlib
      ".*",                           // platform
      "real_concept",                 // test type(s)
      "Inverse Erfc.*",               // test data group
      "erfc_inv", 80, 10);  // test function


   //
   // Catch all cases come last:
   //
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      ".*",                          // test type(s)
      "Erf Function:.*",             // test data group
      "erfc?", 2, 2);   // test function
   add_expected_result(
      ".*aCC.*",                     // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      ".*",                          // test type(s)
      "Inverse Erfc.*",               // test data group
      "erfc_inv", 80, 10);  // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      ".*",                          // test type(s)
      "Inverse Erf.*",               // test data group
      "erfc?_inv", 18, 4);  // test function

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
   test_spots(0.0F, "float");
   test_spots(0.0, "double");
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   test_spots(0.0L, "long double");
#if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x582)) && !defined(BOOST_MATH_NO_REAL_CONCEPT_TESTS)
   test_spots(boost::math::concepts::real_concept(0.1), "real_concept");
#endif
#endif

   expected_results();

   test_erf(0.1F, "float");
   test_erf(0.1, "double");
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   test_erf(0.1L, "long double");
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
#if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x582))
   test_erf(boost::math::concepts::real_concept(0.1), "real_concept");
#endif
#endif
#else
   std::cout << "<note>The long double tests have been disabled on this platform "
      "either because the long double overloads of the usual math functions are "
      "not available at all, or because they are too inaccurate for these tests "
      "to pass.</note>" << std::endl;
#endif
}

/*

Output:

test_erf.cpp
Compiling manifest to resources...
Linking...
Embedding manifest...
Autorun "i:\boost-06-05-03-1300\libs\math\test\Math_test\debug\test_erf.exe"
Running 1 test case...
Testing basic sanity checks for type float
Testing basic sanity checks for type double
Testing basic sanity checks for type long double
Testing basic sanity checks for type real_concept
Tests run with Microsoft Visual C++ version 8.0, Dinkumware standard library version 405, Win32
Testing Erf Function: Small Values with type float
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
boost::math::erf<float> Max = 0 RMS Mean=0
boost::math::erfc<float> Max = 0 RMS Mean=0
Testing Erf Function: Medium Values with type float
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
boost::math::erf<float> Max = 0 RMS Mean=0
boost::math::erfc<float> Max = 0 RMS Mean=0
Testing Erf Function: Large Values with type float
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
boost::math::erf<float> Max = 0 RMS Mean=0
boost::math::erfc<float> Max = 0 RMS Mean=0
Testing Inverse Erf Function with type float
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
boost::math::erf_inv<float> Max = 0 RMS Mean=0
Testing Inverse Erfc Function with type float
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
boost::math::erfc_inv<float> Max = 0 RMS Mean=0
Testing Erf Function: Small Values with type double
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
boost::math::erf<double> Max = 0 RMS Mean=0
boost::math::erfc<double> Max = 0.7857 RMS Mean=0.06415
    worst case at row: 149
    { 0.3343, 0.3636, 0.6364 }
Testing Erf Function: Medium Values with type double
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
boost::math::erf<double> Max = 0.9219 RMS Mean=0.1016
    worst case at row: 273
    { 0.5252, 0.5424, 0.4576 }
boost::math::erfc<double> Max = 1.08 RMS Mean=0.3224
    worst case at row: 287
    { 0.8461, 0.7685, 0.2315 }
Testing Erf Function: Large Values with type double
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
boost::math::erf<double> Max = 0 RMS Mean=0
boost::math::erfc<double> Max = 1.048 RMS Mean=0.2032
    worst case at row: 50
    { 20.96, 1, 4.182e-193 }
Testing Inverse Erf Function with type double
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
boost::math::erf_inv<double> Max = 1.124 RMS Mean=0.5082
    worst case at row: 98
    { 0.9881, 1.779 }
Testing Inverse Erfc Function with type double
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
boost::math::erfc_inv<double> Max = 1.124 RMS Mean=0.5006
    worst case at row: 98
    { 1.988, -1.779 }
Testing Erf Function: Small Values with type long double
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
boost::math::erf<long double> Max = 0 RMS Mean=0
boost::math::erfc<long double> Max = 0.7857 RMS Mean=0.06415
    worst case at row: 149
    { 0.3343, 0.3636, 0.6364 }
Testing Erf Function: Medium Values with type long double
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
boost::math::erf<long double> Max = 0.9219 RMS Mean=0.1016
    worst case at row: 273
    { 0.5252, 0.5424, 0.4576 }
boost::math::erfc<long double> Max = 1.08 RMS Mean=0.3224
    worst case at row: 287
    { 0.8461, 0.7685, 0.2315 }
Testing Erf Function: Large Values with type long double
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
boost::math::erf<long double> Max = 0 RMS Mean=0
boost::math::erfc<long double> Max = 1.048 RMS Mean=0.2032
    worst case at row: 50
    { 20.96, 1, 4.182e-193 }
Testing Inverse Erf Function with type long double
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
boost::math::erf_inv<long double> Max = 1.124 RMS Mean=0.5082
    worst case at row: 98
    { 0.9881, 1.779 }
Testing Inverse Erfc Function with type long double
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
boost::math::erfc_inv<long double> Max = 1.124 RMS Mean=0.5006
    worst case at row: 98
    { 1.988, -1.779 }
Testing Erf Function: Small Values with type real_concept
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
boost::math::erf<real_concept> Max = 1.271 RMS Mean=0.5381
    worst case at row: 144
    { 0.0109, 0.0123, 0.9877 }
boost::math::erfc<real_concept> Max = 0.7857 RMS Mean=0.07777
    worst case at row: 149
    { 0.3343, 0.3636, 0.6364 }
Testing Erf Function: Medium Values with type real_concept
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
boost::math::erf<real_concept> Max = 22.5 RMS Mean=4.224
    worst case at row: 233
    { -0.7852, -0.7332, 1.733 }
Peak error greater than expected value of 20
i:/boost-sandbox/math_toolkit/libs/math/test/handle_test_result.hpp(146): error in "test_main_caller( argc, argv )": check bounds.first >= max_error_found failed
boost::math::erfc<real_concept> Max = 97.77 RMS Mean=8.373
    worst case at row: 289
    { 0.9849, 0.8363, 0.1637 }
Peak error greater than expected value of 20
i:/boost-sandbox/math_toolkit/libs/math/test/handle_test_result.hpp(146): error in "test_main_caller( argc, argv )": check bounds.first >= max_error_found failed
Mean error greater than expected value of 6
i:/boost-sandbox/math_toolkit/libs/math/test/handle_test_result.hpp(151): error in "test_main_caller( argc, argv )": check bounds.second >= mean_error_found failed
Testing Erf Function: Large Values with type real_concept
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
boost::math::erf<real_concept> Max = 0 RMS Mean=0
boost::math::erfc<real_concept> Max = 1.395 RMS Mean=0.2908
    worst case at row: 11
    { 10.99, 1, 1.87e-054 }
Testing Inverse Erf Function with type real_concept
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
boost::math::erf_inv<real_concept> Max = 1.124 RMS Mean=0.5082
    worst case at row: 98
    { 0.9881, 1.779 }
Testing Inverse Erfc Function with type real_concept
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
boost::math::erfc_inv<real_concept> Max = 1.124 RMS Mean=0.5006
    worst case at row: 98
    { 1.988, -1.779 }
Test suite "Test Program" failed with:
  181 assertions out of 184 passed
  3 assertions out of 184 failed
  1 test case out of 1 failed
  Test case "test_main_caller( argc, argv )" failed with:
    181 assertions out of 184 passed
    3 assertions out of 184 failed
Build Time 0:15

*/

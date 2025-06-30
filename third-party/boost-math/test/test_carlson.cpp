//  Copyright 2006 John Maddock
// Copyright Paul A. Bristow 2007.

//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef NOMINMAX
#define NOMINMAX
#endif 
#include <pch_light.hpp>
#include "test_carlson.hpp"

//
// DESCRIPTION:
// ~~~~~~~~~~~~
//
// This file tests the Carlson Elliptic Integrals.  
// There are two sets of tests, spot
// tests which compare our results with the published test values, 
// in Numerical Computation of Real or Complex Elliptic Integrals,
// B. C. Carlson: http://arxiv.org/abs/math.CA/9409227
// However, the bulk of the accuracy tests
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
   // real long doubles:
   //
   if(boost::math::policies::digits<long double, boost::math::policies::policy<> >() > 53)
   {
      add_expected_result(
         ".*",                          // compiler
         ".*",                          // stdlib
         BOOST_PLATFORM,                          // platform
         largest_type,                  // test type(s)
         ".*RJ.*",      // test data group
         ".*", 1000, 50);  // test function
      add_expected_result(
         ".*",                          // compiler
         ".*",                          // stdlib
         BOOST_PLATFORM,                          // platform
         "real_concept",                  // test type(s)
         ".*RJ.*",      // test data group
         ".*", 1000, 50);  // test function
   }
   //
   // Catch all cases come last:
   //
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      largest_type,                  // test type(s)
      ".*RJ.*",      // test data group
      ".*", 250, 50);  // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "real_concept",                  // test type(s)
      ".*RJ.*",      // test data group
      ".*", 250, 50);  // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      largest_type,                  // test type(s)
      ".*",      // test data group
      ".*", 25, 8);  // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "real_concept",                  // test type(s)
      ".*",      // test data group
      ".*", 25, 8);  // test function
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

    boost::math::ellint_rj(1.778e-31, 1.407e+18, 10.05, -4.83e-10);

    test_spots(0.0F, "float");
    test_spots(0.0, "double");
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test_spots(0.0L, "long double");
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
    test_spots(boost::math::concepts::real_concept(0), "real_concept");
#endif
#else
   std::cout << "<note>The long double tests have been disabled on this platform "
      "either because the long double overloads of the usual math functions are "
      "not available at all, or because they are too inaccurate for these tests "
      "to pass.</note>" << std::endl;
#endif
}

/*

test_carlson.cpp
Linking...
Embedding manifest...
Autorun "i:\boost-06-05-03-1300\libs\math\test\Math_test\debug\test_carlson.exe"
Running 1 test case...
Tests run with Microsoft Visual C++ version 8.0, Dinkumware standard library version 405, Win32
Testing: RF: Random data
boost::math::ellint_rf<float> Max = 0 RMS Mean=0
Testing: RC: Random data
boost::math::ellint_rc<float> Max = 0 RMS Mean=0
Testing: RJ: Random data
boost::math::ellint_rf<float> Max = 0 RMS Mean=0
Testing: RD: Random data
boost::math::ellint_rd<float> Max = 0 RMS Mean=0
Testing: RF: Random data
boost::math::ellint_rf<double> Max = 2.949 RMS Mean=0.7498
    worst case at row: 377
    { 3.418e+025, 2.594e-005, 3.264e-012, 6.169e-012 }
Testing: RC: Random data
boost::math::ellint_rc<double> Max = 2.396 RMS Mean=0.6283
    worst case at row: 10
    { 1.97e-029, 3.224e-025, 2.753e+012 }
Testing: RJ: Random data
boost::math::ellint_rf<double> Max = 152.9 RMS Mean=11.15
    worst case at row: 633
    { 1.876e+016, 0.000278, 3.796e-006, -4.412e-005, -1.656e-005 }
Testing: RD: Random data
boost::math::ellint_rd<double> Max = 2.586 RMS Mean=0.8614
    worst case at row: 45
    { 2.111e-020, 8.757e-026, 1.923e-023, 1.004e+033 }
Testing: RF: Random data
boost::math::ellint_rf<long double> Max = 2.949 RMS Mean=0.7498
    worst case at row: 377
    { 3.418e+025, 2.594e-005, 3.264e-012, 6.169e-012 }
Testing: RC: Random data
boost::math::ellint_rc<long double> Max = 2.396 RMS Mean=0.6283
    worst case at row: 10
    { 1.97e-029, 3.224e-025, 2.753e+012 }
Testing: RJ: Random data
boost::math::ellint_rf<long double> Max = 152.9 RMS Mean=11.15
    worst case at row: 633
    { 1.876e+016, 0.000278, 3.796e-006, -4.412e-005, -1.656e-005 }
Testing: RD: Random data
boost::math::ellint_rd<long double> Max = 2.586 RMS Mean=0.8614
    worst case at row: 45
    { 2.111e-020, 8.757e-026, 1.923e-023, 1.004e+033 }
Testing: RF: Random data
boost::math::ellint_rf<real_concept> Max = 2.949 RMS Mean=0.7498
    worst case at row: 377
    { 3.418e+025, 2.594e-005, 3.264e-012, 6.169e-012 }
Testing: RC: Random data
boost::math::ellint_rc<real_concept> Max = 2.396 RMS Mean=0.6283
    worst case at row: 10
    { 1.97e-029, 3.224e-025, 2.753e+012 }
Testing: RJ: Random data
boost::math::ellint_rf<real_concept> Max = 152.9 RMS Mean=11.15
    worst case at row: 633
    { 1.876e+016, 0.000278, 3.796e-006, -4.412e-005, -1.656e-005 }
Testing: RD: Random data
boost::math::ellint_rd<real_concept> Max = 2.586 RMS Mean=0.8614
    worst case at row: 45
    { 2.111e-020, 8.757e-026, 1.923e-023, 1.004e+033 }
*** No errors detected

*/

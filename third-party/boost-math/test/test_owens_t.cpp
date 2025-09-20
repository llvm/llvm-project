// test_owens_t.cpp

// Copyright Paul A. Bristow 2012.
// Copyright Benjamin Sobotta 2012.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Tested using some 30 decimal digit accuracy values from:
// Fast and accurate calculation of Owen's T-function
// Mike Patefield, and David Tandy
// Journal of Statistical Software, 5 (5), 1-25 (2000).
// http://www.jstatsoft.org/v05/a05/paper  Table 3, page 15
// Values of T(h,a) accurate to thirty figures were calculated using 128 bit arithmetic by
// evaluating (9) with m = 48, the summation over k being continued until additional terms did
// not alter the result. The resultant values Tacc(h,a) say, were validated by evaluating (8) with
// m = 48 (i.e. 96 point Gaussian quadrature).

#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error

#ifdef _MSC_VER
#  pragma warning (disable : 4127) // conditional expression is constant
#  pragma warning (disable : 4305) // 'initializing' : truncation from 'double' to 'const float'
// ?? TODO get rid of these warnings?
#endif

#include <boost/math/concepts/real_concept.hpp> // for real_concept.
using ::boost::math::concepts::real_concept;

#include <boost/math/special_functions/owens_t.hpp> // for owens_t function.
using boost::math::owens_t;
#include <boost/math/distributions/normal.hpp>

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/array.hpp>

#include "handle_test_result.hpp"
#include "table_type.hpp"
#include "functor.hpp"
#include "boost/math/tools/test_value.hpp"
#include "test_owens_t.hpp"

//
// Defining TEST_CPP_DEC_FLOAT enables testing of multiprecision support.
// This requires the multiprecision library from sandbox/big_number.
// Note that these tests *do not pass*, but they do give an idea of the 
// error rates that can be expected....
//
#ifdef TEST_CPP_DEC_FLOAT
#include <boost/multiprecision/cpp_dec_float.hpp>

#undef SC_
#define SC_(x) BOOST_MATH_TEST_VALUE(x)
#endif

#include "owens_t_T7.hpp"

#include <iostream>
using std::cout;
using std::endl;
#include <limits>
using std::numeric_limits;

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
   // Catch all cases come last:
   //
   if(std::numeric_limits<long double>::digits > 100)
   {
      //
      // Arbitrary precision versions run out steam (and series iterations)
      // if we push them to too many digits:
      //
      add_expected_result(
         ".*",                            // compiler
         ".*",                            // stdlib
         ".*",                            // platform
         largest_type,                    // test type(s)
         ".*",      // test data group
         "owens_t", 10000000, 1000000);  // test function
   }
   else if(std::numeric_limits<long double>::digits > 60)
   {
      add_expected_result(
         ".*",                            // compiler
         ".*",                            // stdlib
         ".*",                            // platform
         largest_type,                    // test type(s)
         ".*",      // test data group
         "owens_t", 500, 100);  // test function
   }
   else
   {
      add_expected_result(
         ".*",                            // compiler
         ".*",                            // stdlib
         ".*",                            // platform
         largest_type,                    // test type(s)
         ".*",      // test data group
         "owens_t", 60, 5);  // test function
   }
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

  // Basic sanity-check spot values.

  // (Parameter value, arbitrarily zero, only communicates the floating point type).
  test_spots(0.0F); // Test float.
  test_spots(0.0); // Test double.
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
  test_spots(0.0L); // Test long double.
#if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x582)) && !defined(BOOST_MATH_NO_REAL_CONCEPT_TESTS)
  test_spots(boost::math::concepts::real_concept(0.)); // Test real concept.
#endif
#endif

  check_against_T7(0.0F); // Test float.
  check_against_T7(0.0); // Test double.
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
  check_against_T7(0.0L); // Test long double.
#if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x582)) && !defined(BOOST_MATH_NO_REAL_CONCEPT_TESTS)
  check_against_T7(boost::math::concepts::real_concept(0.)); // Test real concept.
#endif
#endif

  test_owens_t(0.0F, "float"); // Test float.
  test_owens_t(0.0, "double"); // Test double.
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
  test_owens_t(0.0L, "long double"); // Test long double.
#if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x582)) && !defined(BOOST_MATH_NO_REAL_CONCEPT_TESTS)
  test_owens_t(boost::math::concepts::real_concept(0.), "real_concept"); // Test real concept.
#endif
#endif
#if defined(TEST_CPP_DEC_FLOAT) && !defined(BOOST_MATH_STANDALONE)
  typedef boost::multiprecision::number<boost::multiprecision::cpp_dec_float<35> > cpp_dec_float_35;
  test_owens_t(cpp_dec_float_35(0), "cpp_dec_float_35"); // Test real concept.
  test_owens_t(boost::multiprecision::cpp_dec_float_50(0), "cpp_dec_float_50"); // Test real concept.
  test_owens_t(boost::multiprecision::cpp_dec_float_100(0), "cpp_dec_float_100"); // Test real concept.
#endif
  
} // BOOST_AUTO_TEST_CASE( test_main )

/*

Output:

  Description: Autorun "J:\Cpp\MathToolkit\test\Math_test\Debug\test_owens_t.exe"
  Running 1 test case...
  Tests run with Microsoft Visual C++ version 10.0, Dinkumware standard library version 520, Win32
  Tolerance = 3.57628e-006.
  Tolerance = 6.66134e-015.
  Tolerance = 6.66134e-015.
  Tolerance = 6.66134e-015.
  Tolerance = 1.78814e-005.
  Tolerance = 3.33067e-014.
  Tolerance = 3.33067e-014.
  Tolerance = 3.33067e-014.
  Testing Owens T (medium small values) with type float
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  boost::math::owens_t<float> Max = 0 RMS Mean=0
  
  
  Testing Owens T (large and diverse values) with type float
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  boost::math::owens_t<float> Max = 0 RMS Mean=0
  
  
  Testing Owens T (medium small values) with type double
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  boost::math::owens_t<double> Max = 4.375 RMS Mean=0.9728
      worst case at row: 81
      { 4.4206809997558594, 0.1269868016242981, 1.0900281236140834e-006 }
  
  
  Testing Owens T (large and diverse values) with type double
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  boost::math::owens_t<double> Max = 3.781 RMS Mean=0.6206
      worst case at row: 430
      { 3.4516773223876953, 0.0010718167759478092, 4.413983645332431e-007 }
  
  
  Testing Owens T (medium small values) with type long double
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  boost::math::owens_t<long double> Max = 4.375 RMS Mean=0.9728
      worst case at row: 81
      { 4.4206809997558594, 0.1269868016242981, 1.0900281236140834e-006 }
  
  
  Testing Owens T (large and diverse values) with type long double
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  boost::math::owens_t<long double> Max = 3.781 RMS Mean=0.6206
      worst case at row: 430
      { 3.4516773223876953, 0.0010718167759478092, 4.413983645332431e-007 }
  
  
  Testing Owens T (medium small values) with type real_concept
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  boost::math::owens_t<real_concept> Max = 4.375 RMS Mean=1.032
      worst case at row: 81
      { 4.4206809997558594, 0.1269868016242981, 1.0900281236140834e-006 }
  
  
  Testing Owens T (large and diverse values) with type real_concept
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  boost::math::owens_t<real_concept> Max = 21.04 RMS Mean=1.102
      worst case at row: 439
      { 3.4516773223876953, 0.98384737968444824, 0.00013923002576038691 }
  
  
  
  *** No errors detected


*/




//  Copyright John Maddock 2005.
//  Copyright Paul A. Bristow 2010
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pch_light.hpp>

#if defined __has_include
#  if __cplusplus > 202002L || _MSVC_LANG > 202002L 
#    if __has_include (<stdfloat>)
#    include <stdfloat>
#    endif
#  endif
#endif

#define SC_(x) static_cast<T>(BOOST_MATH_BIG_CONSTANT(T, 128, x))

#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/special_functions/log1p.hpp>
#include <boost/math/special_functions/expm1.hpp>
#include <boost/math/tools/big_constant.hpp>
#include "log1p_expm1_test.hpp"

//
// DESCRIPTION:
// ~~~~~~~~~~~~
//
// This file tests the functions log1p and expm1.  The accuracy tests
// use values generated with NTL::RR at 1000-bit precision
// and our generic versions of these functions.
//
// Note that this tests the generic versions of our functions using
// software emulated types.  This ensures these still get tested
// even though we mostly defer to std::expm1 and std::log1p at
// these precisions nowadays
//

void expected_results()
{
   //
   // Define the max and mean errors expected for
   // various compilers and platforms.
   //

   //
   // Catch all cases come last:
   //
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      ".*",                          // test type(s)
      ".*",                          // test data group
      ".*",                          // test function
      4,                             // Max Peek error
      3);                            // Max mean error

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
#ifdef __STDCPP_FLOAT32_T__
   test(std::float32_t(0), "std::float32_t");
#endif
#ifdef __STDCPP_FLOAT64_T__
   test(std::float64_t(0), "std::float64_t");
#endif
#ifdef __STDCPP_FLOAT128_T__
   // As of gcc-13 this does not work as float128_t has no std::math function overloads
   //test(std::float128_t(0), "std::float128_t");
#endif
}


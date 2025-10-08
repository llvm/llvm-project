//  Copyright John Maddock 2005.
//  Copyright Paul A. Bristow 2010
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pch_light.hpp>

#define SC_(x) static_cast<T>(BOOST_STRINGIZE(x))

#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/special_functions/log1p.hpp>
#include <boost/math/special_functions/expm1.hpp>
#include "log1p_expm1_test.hpp"
#include <boost/multiprecision/cpp_bin_float.hpp>

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
   test(boost::multiprecision::cpp_bin_float_single(0), "cpp_bin_float_single");
   test(boost::multiprecision::cpp_bin_float_double(0), "cpp_bin_float_double");
   test(boost::multiprecision::cpp_bin_float_double_extended(0), "cpp_bin_float_double_extended");
   test(boost::multiprecision::cpp_bin_float_quad(0), "cpp_bin_float_quad");
}


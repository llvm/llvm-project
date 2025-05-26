// test_nc_t.cpp

// Copyright John Maddock 2008, 2012.
// Copyright Paul A. Bristow 2012.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pch.hpp> // Need to include lib/math/test in path.

#ifdef _MSC_VER
#pragma warning (disable:4127 4512)
#endif

#include <boost/math/tools/test.hpp>
#include <boost/math/concepts/real_concept.hpp> // for real_concept
#include <boost/math/distributions/non_central_t.hpp> // for chi_squared_distribution.
#include <boost/math/distributions/normal.hpp> // for normal distribution (for comparison).
#include <boost/multiprecision/cpp_bin_float.hpp> // for normal distribution (for comparison).

#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp> // for test_main
#include <boost/test/results_collector.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp> // for BOOST_CHECK_CLOSE

#define SC_(x) static_cast<T>(BOOST_STRINGIZE(x))

#include "functor.hpp"
#include "handle_test_result.hpp"
#include "table_type.hpp"
#include "test_nc_t.hpp"

#include <iostream>
#include <iomanip>
using std::cout;
using std::endl;
#include <limits>
using std::numeric_limits;

#if defined(__GNUC__) && (__GNUC__ < 13) && (defined(__CYGWIN__) || defined(_WIN32))
//
// We either get an internal compiler error, or worse, the compiler prints nothing at all
// and exits with an error code :(
//
# define DISABLE_THIS_TEST
#endif


void expected_results()
{
   //
   // Define the max and mean errors expected for
   // various compilers and platforms.
   //
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      "cpp_bin_float_quad",             // test type(s)
      ".*large parameters.*",           // test data group
      "[^|]*", 300000, 100000);         // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      "cpp_bin_float_quad",                         // test type(s)
      "[^|]*PDF",                  // test data group
      "[^|]*", static_cast<std::uintmax_t>(1 / boost::math::tools::root_epsilon<boost::multiprecision::cpp_bin_float_quad>()), static_cast<std::uintmax_t>(1 / boost::math::tools::root_epsilon<boost::multiprecision::cpp_bin_float_quad>())); // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      "cpp_bin_float_quad",             // test type(s)
      "[^|]*",                          // test data group
      "[^|]*", 250, 50);                // test function

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
   // Basic sanity-check spot values.
   expected_results();
#ifndef DISABLE_THIS_TEST
#if !BOOST_WORKAROUND(BOOST_MSVC, < 1920)
   test_spots(boost::multiprecision::cpp_bin_float_quad(0));
#endif
   test_accuracy(boost::multiprecision::cpp_bin_float_quad(0), "cpp_bin_float_quad");
#endif
   // double precision tests only:
   //test_big_df(boost::multiprecision::cpp_bin_float_quad(0));
} // BOOST_AUTO_TEST_CASE( test_main )


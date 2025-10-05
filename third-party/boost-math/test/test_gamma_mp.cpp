// Copyright John Maddock 2006.
// Copyright Paul A. Bristow 2007, 2009
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/tools/config.hpp>
#ifndef BOOST_MATH_NO_MP_TESTS

#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/tools/stats.hpp>
#include <boost/math/tools/test.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <array>
#include "functor.hpp"

#include "handle_test_result.hpp"
#include "table_type.hpp"

#ifndef SC_
#define SC_(x) static_cast<typename table_type<T>::type>(BOOST_STRINGIZE(x))
#endif

template <class Real, class T>
void do_test_gamma(const T& data, const char* type_name, const char* test_name)
{
#if !(defined(ERROR_REPORTING_MODE) && (!defined(TGAMMA_FUNCTION_TO_TEST) || !defined(LGAMMA_FUNCTION_TO_TEST)))
   typedef Real                   value_type;

   typedef value_type (*pg)(value_type);
#ifdef TGAMMA_FUNCTION_TO_TEST
   pg funcp = TGAMMA_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = boost::math::tgamma<value_type>;
#else
   pg funcp = boost::math::tgamma;
#endif

   boost::math::tools::test_result<value_type> result;

   std::cout << "Testing " << test_name << " with type " << type_name
      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

   //
   // test tgamma against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(funcp, 0),
      extract_result<Real>(1));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "tgamma", test_name);
   //
   // test lgamma against data:
   //
#ifdef LGAMMA_FUNCTION_TO_TEST
   funcp = LGAMMA_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   funcp = boost::math::lgamma<value_type>;
#else
   funcp = boost::math::lgamma;
#endif
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(funcp, 0),
      extract_result<Real>(2));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "lgamma", test_name);

   std::cout << std::endl;
#endif
}

template <class T>
void test_gamma(T, const char* name)
{
   //
   // The actual test data is rather verbose, so it's in a separate file
   //
   // The contents are as follows, each row of data contains
   // three items, input value, gamma and lgamma:
   //
   // gamma and lgamma at integer and half integer values:
   // std::array<std::array<T, 3>, N> factorials;
   //
   // gamma and lgamma for z near 0:
   // std::array<std::array<T, 3>, N> near_0;
   //
   // gamma and lgamma for z near 1:
   // std::array<std::array<T, 3>, N> near_1;
   //
   // gamma and lgamma for z near 2:
   // std::array<std::array<T, 3>, N> near_2;
   //
   // gamma and lgamma for z near -10:
   // std::array<std::array<T, 3>, N> near_m10;
   //
   // gamma and lgamma for z near -55:
   // std::array<std::array<T, 3>, N> near_m55;
   //
   // The last two cases are chosen more or less at random,
   // except that one is even and the other odd, and both are
   // at negative poles.  The data near zero also tests near
   // a pole, the data near 1 and 2 are to probe lgamma as
   // the result -> 0.
   //
#  include "tgamma_mp_data.hpp"

   do_test_gamma<T>(factorials, name, "factorials");
   do_test_gamma<T>(near_0, name, "near 0");
   do_test_gamma<T>(near_1, name, "near 1");
   do_test_gamma<T>(near_2, name, "near 2");
   do_test_gamma<T>(near_m10, name, "near -10");
   do_test_gamma<T>(near_m55, name, "near -55");
}

void expected_results()
{
   //
   // Define the max and mean errors expected for
   // various compilers and platforms.
   //
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "cpp_bin_float_100|number<cpp_bin_float<85> >",           // test type(s)
      ".*",                          // test data group
      "lgamma", 600000, 300000);      // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "number<cpp_bin_float<[56]5> >",           // test type(s)
      ".*",                          // test data group
      "lgamma", 7000, 3000);      // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "number<cpp_bin_float<75> >",           // test type(s)
      ".*",                          // test data group
      "lgamma", 40000, 15000);      // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      ".*",                          // test type(s)
      ".*",                          // test data group
      "lgamma", 600, 200);            // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      ".*",                          // test type(s)
      ".*",                          // test data group
      "[tl]gamma", 120, 50);            // test function
   //
   // Finish off by printing out the compiler/stdlib/platform names,
   // we do this to make it easier to mark up expected error rates.
   //
   std::cout << "Tests run with " << BOOST_COMPILER << ", "
      << BOOST_STDLIB << ", " << BOOST_PLATFORM << std::endl;
}

BOOST_AUTO_TEST_CASE(test_main)
{
   expected_results();
   using namespace boost::multiprecision;
#if !defined(TEST) || (TEST == 1)
   test_gamma(number<cpp_bin_float<38> >(0), "number<cpp_bin_float<38> >");
   test_gamma(number<cpp_bin_float<45> >(0), "number<cpp_bin_float<45> >");
#endif
#if !defined(TEST) || (TEST == 2)
   test_gamma(cpp_bin_float_50(0), "cpp_bin_float_50");
   test_gamma(number<cpp_bin_float<55> >(0), "number<cpp_bin_float<55> >");
   test_gamma(number<cpp_bin_float<65> >(0), "number<cpp_bin_float<65> >");
#endif
#if !defined(TEST) || (TEST == 3)
   test_gamma(number<cpp_bin_float<75> >(0), "number<cpp_bin_float<75> >");
   test_gamma(number<cpp_bin_float<85> >(0), "number<cpp_bin_float<85> >");
   test_gamma(cpp_bin_float_100(0), "cpp_bin_float_100");
#endif
}
#else // No mp tests
int main(void) { return 0; }
#endif

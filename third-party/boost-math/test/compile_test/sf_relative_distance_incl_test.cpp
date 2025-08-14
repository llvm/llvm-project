//  Copyright John Maddock 2015.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/special_functions/next.hpp>
// #includes all the files that it needs to.
//
#include <boost/math/special_functions/relative_difference.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
   check_result<float>(boost::math::relative_difference<float>(f, f));
   check_result<double>(boost::math::relative_difference<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::relative_difference<long double>(l, l));
#endif
   check_result<float>(boost::math::epsilon_difference<float>(f, f));
   check_result<double>(boost::math::epsilon_difference<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::epsilon_difference<long double>(l, l));
#endif
}

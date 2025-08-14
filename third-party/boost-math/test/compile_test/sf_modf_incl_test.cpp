//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/special_functions/modf.hpp>
// #includes all the files that it needs to.
//
#include <boost/math/special_functions/modf.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

float ff;
double dd;
long double lldd;

int ii;
long ll;
#ifdef BOOST_HAS_LONG_LONG
boost::long_long_type llll;
#endif

void compile_and_link_test()
{
   check_result<float>(boost::math::modf(f, &ff));
   check_result<double>(boost::math::modf(d, &dd));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::modf(l, &lldd));
#endif
   check_result<float>(boost::math::modf(f, &ii));
   check_result<double>(boost::math::modf(d, &ii));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::modf(l, &ii));
#endif
   check_result<float>(boost::math::modf(f, &ll));
   check_result<double>(boost::math::modf(d, &ll));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::modf(l, &ll));
#endif
#ifdef BOOST_HAS_LONG_LONG
   check_result<float>(boost::math::modf(f, &llll));
   check_result<double>(boost::math::modf(d, &llll));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::modf(l, &llll));
#endif
#endif
}

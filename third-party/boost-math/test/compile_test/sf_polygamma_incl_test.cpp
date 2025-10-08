//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/special_functions/digamma.hpp>
// #includes all the files that it needs to.
//
#include <boost/math/special_functions/polygamma.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
   check_result<float>(boost::math::trigamma<float>(f));
   check_result<double>(boost::math::trigamma<double>(d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::trigamma<long double>(l));
#endif
   check_result<float>(boost::math::polygamma<float>(1, f));
   check_result<double>(boost::math::polygamma<double>(1, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::polygamma<long double>(1, l));
#endif
}

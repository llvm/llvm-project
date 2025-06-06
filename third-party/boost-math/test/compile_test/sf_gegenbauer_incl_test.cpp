//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/special_functions/gegenbauer.hpp>
// #includes all the files that it needs to.
//
#include <boost/math/special_functions/gegenbauer.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
   check_result<float>(boost::math::gegenbauer<float>(1, f, f));
   check_result<double>(boost::math::gegenbauer<double>(1, d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::gegenbauer<long double>(1, l, l));
#endif

   check_result<float>(boost::math::gegenbauer_derivative<float>(1, f, f, 1));
   check_result<double>(boost::math::gegenbauer_derivative<double>(1, d, d, 1));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::gegenbauer_derivative<long double>(1, l, l, 1));
#endif

   check_result<float>(boost::math::gegenbauer_prime<float>(1, f, f));
   check_result<double>(boost::math::gegenbauer_prime<double>(1, d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::gegenbauer_prime<long double>(1, l, l));
#endif
}

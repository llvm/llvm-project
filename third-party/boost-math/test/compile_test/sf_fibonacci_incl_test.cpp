//  Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/special_functions/fibonacci.hpp>
// #includes all the files that it needs to.
//
#include <boost/math/special_functions/fibonacci.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
   check_result<float>(boost::math::fibonacci<float>(f));
   check_result<double>(boost::math::fibonacci<double>(d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::fibonacci<long double>(l));
#endif

   check_result<float>(boost::math::unchecked_fibonacci<float>(f));
   check_result<double>(boost::math::unchecked_fibonacci<double>(d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::unchecked_fibonacci<long double>(l));
#endif
}

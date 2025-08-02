//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/special_functions/trunc.hpp>
// #includes all the files that it needs to.
//
#include <boost/math/special_functions/trunc.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
   check_result<float>(boost::math::trunc<float>(f));
   check_result<double>(boost::math::trunc<double>(d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::trunc<long double>(l));
#endif
   check_result<int>(boost::math::itrunc<float>(f));
   check_result<int>(boost::math::itrunc<double>(d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<int>(boost::math::itrunc<long double>(l));
#endif
   check_result<long>(boost::math::ltrunc<float>(f));
   check_result<long>(boost::math::ltrunc<double>(d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long>(boost::math::ltrunc<long double>(l));
#endif
#ifdef BOOST_HAS_LONG_LONG
   check_result<boost::long_long_type>(boost::math::lltrunc<float>(f));
   check_result<boost::long_long_type>(boost::math::lltrunc<double>(d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<boost::long_long_type>(boost::math::lltrunc<long double>(l));
#endif
#endif
}

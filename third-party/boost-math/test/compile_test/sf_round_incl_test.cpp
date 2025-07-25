//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/special_functions/round.hpp>
// #includes all the files that it needs to.
//
#include <boost/math/special_functions/round.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
   check_result<float>(boost::math::round<float>(f));
   check_result<double>(boost::math::round<double>(d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::round<long double>(l));
#endif
   check_result<int>(boost::math::iround<float>(f));
   check_result<int>(boost::math::iround<double>(d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<int>(boost::math::iround<long double>(l));
#endif
   check_result<long>(boost::math::lround<float>(f));
   check_result<long>(boost::math::lround<double>(d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long>(boost::math::lround<long double>(l));
#endif
#ifdef BOOST_HAS_LONG_LONG
   check_result<boost::long_long_type>(boost::math::llround<float>(f));
   check_result<boost::long_long_type>(boost::math::llround<double>(d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<boost::long_long_type>(boost::math::llround<long double>(l));
#endif
#endif
}

//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/special_functions/lambert_w.hpp>
// #includes all the files that it needs to.
//
#include <boost/math/special_functions/lambert_w.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
   check_result<float>(boost::math::lambert_w0<float>(f));
   check_result<double>(boost::math::lambert_w0<double>(d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::lambert_w0<long double>(l));
#endif

   check_result<float>(boost::math::lambert_w0_prime<float>(f));
   check_result<double>(boost::math::lambert_w0_prime<double>(d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::lambert_w0_prime<long double>(l));
#endif

   check_result<float>(boost::math::lambert_wm1<float>(f));
   check_result<double>(boost::math::lambert_wm1<double>(d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::lambert_wm1<long double>(l));
#endif

   check_result<float>(boost::math::lambert_wm1_prime<float>(f));
   check_result<double>(boost::math::lambert_wm1_prime<double>(d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::lambert_wm1_prime<long double>(l));
#endif
}

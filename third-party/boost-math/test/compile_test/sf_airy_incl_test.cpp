//  Copyright John Maddock 2012.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/special_functions/bessel.hpp>
// #includes all the files that it needs to.
//
#include <boost/math/special_functions/airy.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
   check_result<float>(boost::math::airy_ai<float>(f));
   check_result<double>(boost::math::airy_ai<double>(d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::airy_ai<long double>(l));
#endif

   check_result<float>(boost::math::airy_bi<float>(f));
   check_result<double>(boost::math::airy_bi<double>(d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::airy_bi<long double>(l));
#endif

   check_result<float>(boost::math::airy_ai_prime<float>(f));
   check_result<double>(boost::math::airy_ai_prime<double>(d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::airy_ai_prime<long double>(l));
#endif

   check_result<float>(boost::math::airy_bi_prime<float>(f));
   check_result<double>(boost::math::airy_bi_prime<double>(d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::airy_bi_prime<long double>(l));
#endif

}

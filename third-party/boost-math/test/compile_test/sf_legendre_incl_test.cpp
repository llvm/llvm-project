//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/special_functions/legendre.hpp>
// #includes all the files that it needs to.
//
#include <boost/math/special_functions/legendre.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
   check_result<float>(boost::math::legendre_p<float>(i, f));
   check_result<double>(boost::math::legendre_p<double>(i, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::legendre_p<long double>(i, l));
#endif
   check_result<float>(boost::math::legendre_p_prime<float>(i, f));
   check_result<double>(boost::math::legendre_p_prime<double>(i, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::legendre_p_prime<long double>(i, l));
#endif

   check_result<float>(boost::math::legendre_p<float>(i, i, f));
   check_result<double>(boost::math::legendre_p<double>(i, i, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::legendre_p<long double>(i, i, l));
#endif

   check_result<float>(boost::math::legendre_q<float>(u, f));
   check_result<double>(boost::math::legendre_q<double>(u, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::legendre_q<long double>(u, l));
#endif
}

//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/special_functions/legendre.hpp>
// #includes all the files that it needs to.
//
#include <boost/math/special_functions/legendre_stieltjes.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
   boost::math::legendre_stieltjes<float> lsf(3);
   boost::math::legendre_stieltjes<double> lsd(3);
   check_result<float>(lsf(f));
   check_result<double>(lsd(d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   boost::math::legendre_stieltjes<long double> lsl(3);
   check_result<long double>(lsl(l));
#endif
}

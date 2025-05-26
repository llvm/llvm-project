//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/special_functions/erf.hpp>
// #includes all the files that it needs to.
//
#include <boost/math/special_functions/erf.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
   check_result<float>(boost::math::erf<float>(f));
   check_result<double>(boost::math::erf<double>(d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::erf<long double>(l));
#endif

   check_result<float>(boost::math::erfc<float>(f));
   check_result<double>(boost::math::erfc<double>(d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::erfc<long double>(l));
#endif

   check_result<float>(boost::math::erf_inv<float>(f));
   check_result<double>(boost::math::erf_inv<double>(d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::erf_inv<long double>(l));
#endif

   check_result<float>(boost::math::erfc_inv<float>(f));
   check_result<double>(boost::math::erfc_inv<double>(d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::erfc_inv<long double>(l));
#endif
}

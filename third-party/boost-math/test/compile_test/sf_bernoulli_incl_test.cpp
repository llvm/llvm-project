//  Copyright John Maddock 2012.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/special_functions/bernoulli.hpp>
// #includes all the files that it needs to.
//
#include <ostream>
#include <boost/math/special_functions/bernoulli.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
   check_result<float>(boost::math::bernoulli_b2n<float>(i));
   check_result<double>(boost::math::bernoulli_b2n<double>(i));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::bernoulli_b2n<long double>(i));
#endif

   check_result<float>(boost::math::tangent_t2n<float>(i));
   check_result<double>(boost::math::tangent_t2n<double>(i));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::tangent_t2n<long double>(i));
#endif
#ifdef BOOST_MATH_HAVE_CONSTEXPR_TABLES
   constexpr float ce_f = boost::math::unchecked_bernoulli_b2n<float>(2);
   constexpr float ce_d = boost::math::unchecked_bernoulli_b2n<double>(2);
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   constexpr float ce_l = boost::math::unchecked_bernoulli_b2n<long double>(2);
   std::ostream cnull(0);
   cnull << ce_f << ce_d << ce_l << std::endl;
#endif
#endif
}

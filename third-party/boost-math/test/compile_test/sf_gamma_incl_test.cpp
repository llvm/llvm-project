//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/special_functions/gamma.hpp>
// #includes all the files that it needs to.
//
#include <boost/math/special_functions/gamma.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
   check_result<float>(boost::math::tgamma<float>(f));
   check_result<double>(boost::math::tgamma<double>(d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::tgamma<long double>(l));
#endif

   check_result<float>(boost::math::lgamma<float>(f));
   check_result<double>(boost::math::lgamma<double>(d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::lgamma<long double>(l));
#endif

   check_result<float>(boost::math::gamma_p<float>(f, f));
   check_result<double>(boost::math::gamma_p<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::gamma_p<long double>(l, l));
#endif

   check_result<float>(boost::math::gamma_q<float>(f, f));
   check_result<double>(boost::math::gamma_q<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::gamma_q<long double>(l, l));
#endif

   check_result<float>(boost::math::gamma_p_inv<float>(f, f));
   check_result<double>(boost::math::gamma_p_inv<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::gamma_p_inv<long double>(l, l));
#endif

   check_result<float>(boost::math::gamma_q_inv<float>(f, f));
   check_result<double>(boost::math::gamma_q_inv<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::gamma_q_inv<long double>(l, l));
#endif

   check_result<float>(boost::math::gamma_p_inva<float>(f, f));
   check_result<double>(boost::math::gamma_p_inva<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::gamma_p_inva<long double>(l, l));
#endif

   check_result<float>(boost::math::gamma_q_inva<float>(f, f));
   check_result<double>(boost::math::gamma_q_inva<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::gamma_q_inva<long double>(l, l));
#endif

   check_result<float>(boost::math::gamma_p_derivative<float>(f, f));
   check_result<double>(boost::math::gamma_p_derivative<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::gamma_p_derivative<long double>(l, l));
#endif

   check_result<float>(boost::math::tgamma_ratio<float>(f, f));
   check_result<double>(boost::math::tgamma_ratio<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::tgamma_ratio<long double>(l, l));
#endif

   check_result<float>(boost::math::tgamma_delta_ratio<float>(f, f));
   check_result<double>(boost::math::tgamma_delta_ratio<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::tgamma_delta_ratio<long double>(l, l));
#endif
}

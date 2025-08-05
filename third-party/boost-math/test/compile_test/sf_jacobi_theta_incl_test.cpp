//  Copyright Evan Miller 2020.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Basic sanity check that header <boost/math/special_functions/jacobi_theta.hpp>
// #includes all the files that it needs to.
//
#include <boost/math/special_functions/jacobi_theta.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
    // Q parameter
   check_result<float>(boost::math::jacobi_theta1<float>(f, f));
   check_result<double>(boost::math::jacobi_theta1<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::jacobi_theta1<long double>(l, l));
#endif

   check_result<float>(boost::math::jacobi_theta2<float>(f, f));
   check_result<double>(boost::math::jacobi_theta2<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::jacobi_theta2<long double>(l, l));
#endif

   check_result<float>(boost::math::jacobi_theta3<float>(f, f));
   check_result<double>(boost::math::jacobi_theta3<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::jacobi_theta3<long double>(l, l));
#endif

   check_result<float>(boost::math::jacobi_theta4<float>(f, f));
   check_result<double>(boost::math::jacobi_theta4<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::jacobi_theta4<long double>(l, l));
#endif

    // Tau parameter
   check_result<float>(boost::math::jacobi_theta1tau<float>(f, f));
   check_result<double>(boost::math::jacobi_theta1tau<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::jacobi_theta1tau<long double>(l, l));
#endif

   check_result<float>(boost::math::jacobi_theta2tau<float>(f, f));
   check_result<double>(boost::math::jacobi_theta2tau<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::jacobi_theta2tau<long double>(l, l));
#endif

   check_result<float>(boost::math::jacobi_theta3tau<float>(f, f));
   check_result<double>(boost::math::jacobi_theta3tau<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::jacobi_theta3tau<long double>(l, l));
#endif

   check_result<float>(boost::math::jacobi_theta4tau<float>(f, f));
   check_result<double>(boost::math::jacobi_theta4tau<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::jacobi_theta4tau<long double>(l, l));
#endif

   // Minus 1 flavors
   check_result<float>(boost::math::jacobi_theta3m1<float>(f, f));
   check_result<double>(boost::math::jacobi_theta3m1<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::jacobi_theta3m1<long double>(l, l));
#endif

   check_result<float>(boost::math::jacobi_theta4m1<float>(f, f));
   check_result<double>(boost::math::jacobi_theta4m1<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::jacobi_theta4m1<long double>(l, l));
#endif

   check_result<float>(boost::math::jacobi_theta3m1tau<float>(f, f));
   check_result<double>(boost::math::jacobi_theta3m1tau<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::jacobi_theta3m1tau<long double>(l, l));
#endif

   check_result<float>(boost::math::jacobi_theta4m1tau<float>(f, f));
   check_result<double>(boost::math::jacobi_theta4m1tau<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::jacobi_theta4m1tau<long double>(l, l));
#endif
}

//  Copyright John Maddock 2012.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/special_functions/bessel.hpp>
// #includes all the files that it needs to.
//
#include <boost/math/special_functions/jacobi_elliptic.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
   check_result<float>(boost::math::jacobi_elliptic<float>(f, f, static_cast<float*>(0), static_cast<float*>(0)));
   check_result<double>(boost::math::jacobi_elliptic<double>(d, d, static_cast<double*>(0), static_cast<double*>(0)));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::jacobi_elliptic<long double>(l, l, static_cast<long double*>(0), static_cast<long double*>(0)));
#endif

   check_result<float>(boost::math::jacobi_sn<float>(f, f));
   check_result<double>(boost::math::jacobi_sn<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::jacobi_sn<long double>(l, l));
#endif

   check_result<float>(boost::math::jacobi_cn<float>(f, f));
   check_result<double>(boost::math::jacobi_cn<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::jacobi_cn<long double>(l, l));
#endif

   check_result<float>(boost::math::jacobi_dn<float>(f, f));
   check_result<double>(boost::math::jacobi_dn<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::jacobi_dn<long double>(l, l));
#endif

   check_result<float>(boost::math::jacobi_cd<float>(f, f));
   check_result<double>(boost::math::jacobi_cd<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::jacobi_cd<long double>(l, l));
#endif

   check_result<float>(boost::math::jacobi_dc<float>(f, f));
   check_result<double>(boost::math::jacobi_dc<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::jacobi_dc<long double>(l, l));
#endif

   check_result<float>(boost::math::jacobi_ns<float>(f, f));
   check_result<double>(boost::math::jacobi_ns<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::jacobi_ns<long double>(l, l));
#endif

   check_result<float>(boost::math::jacobi_sd<float>(f, f));
   check_result<double>(boost::math::jacobi_sd<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::jacobi_sd<long double>(l, l));
#endif

   check_result<float>(boost::math::jacobi_ds<float>(f, f));
   check_result<double>(boost::math::jacobi_ds<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::jacobi_ds<long double>(l, l));
#endif

   check_result<float>(boost::math::jacobi_nc<float>(f, f));
   check_result<double>(boost::math::jacobi_nc<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::jacobi_nc<long double>(l, l));
#endif

   check_result<float>(boost::math::jacobi_nd<float>(f, f));
   check_result<double>(boost::math::jacobi_nd<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::jacobi_nd<long double>(l, l));
#endif

   check_result<float>(boost::math::jacobi_sc<float>(f, f));
   check_result<double>(boost::math::jacobi_sc<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::jacobi_sc<long double>(l, l));
#endif

   check_result<float>(boost::math::jacobi_cs<float>(f, f));
   check_result<double>(boost::math::jacobi_cs<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::jacobi_cs<long double>(l, l));
#endif

}

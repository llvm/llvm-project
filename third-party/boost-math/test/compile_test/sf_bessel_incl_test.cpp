//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/special_functions/bessel.hpp>
// #includes all the files that it needs to.
//
#include <boost/math/special_functions/bessel.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
   check_result<float>(boost::math::cyl_bessel_j<float>(f, f));
   check_result<double>(boost::math::cyl_bessel_j<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::cyl_bessel_j<long double>(l, l));
#endif

   check_result<float>(boost::math::cyl_neumann<float>(f, f));
   check_result<double>(boost::math::cyl_neumann<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::cyl_neumann<long double>(l, l));
#endif

   check_result<float>(boost::math::cyl_bessel_i<float>(f, f));
   check_result<double>(boost::math::cyl_bessel_i<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::cyl_bessel_i<long double>(l, l));
#endif

   check_result<float>(boost::math::cyl_bessel_k<float>(f, f));
   check_result<double>(boost::math::cyl_bessel_k<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::cyl_bessel_k<long double>(l, l));
#endif

   check_result<float>(boost::math::sph_bessel<float>(u, f));
   check_result<double>(boost::math::sph_bessel<double>(u, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::sph_bessel<long double>(u, l));
#endif

   check_result<float>(boost::math::sph_neumann<float>(u, f));
   check_result<double>(boost::math::sph_neumann<double>(u, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::sph_neumann<long double>(u, l));
#endif
}

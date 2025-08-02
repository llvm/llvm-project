//  Copyright John Maddock 2012.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/special_functions/bessel.hpp>
// #includes all the files that it needs to.
//
#include <boost/math/special_functions/hankel.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

inline void check_result_imp(std::complex<float>, std::complex<float>){}
inline void check_result_imp(std::complex<double>, std::complex<double>){}
inline void check_result_imp(std::complex<long double>, std::complex<long double>){}

void compile_and_link_test()
{
   check_result<std::complex<float> >(boost::math::cyl_hankel_1<float>(f, f));
   check_result<std::complex<double> >(boost::math::cyl_hankel_1<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<std::complex<long double> >(boost::math::cyl_hankel_1<long double>(l, l));
#endif

   check_result<std::complex<float> >(boost::math::cyl_hankel_2<float>(f, f));
   check_result<std::complex<double> >(boost::math::cyl_hankel_2<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<std::complex<long double> >(boost::math::cyl_hankel_2<long double>(l, l));
#endif

   check_result<std::complex<float> >(boost::math::sph_hankel_1<float>(f, f));
   check_result<std::complex<double> >(boost::math::sph_hankel_1<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<std::complex<long double> >(boost::math::sph_hankel_1<long double>(l, l));
#endif

   check_result<std::complex<float> >(boost::math::sph_hankel_2<float>(f, f));
   check_result<std::complex<double> >(boost::math::sph_hankel_2<double>(d, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<std::complex<long double> >(boost::math::sph_hankel_2<long double>(l, l));
#endif

}

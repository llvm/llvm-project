//  Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header
// #includes all the files that it needs to.
#include <boost/math/differentiation/lanczos_smoothing.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
    float f_temp = 1;
    boost::math::differentiation::discrete_lanczos_derivative f_lanczos(f_temp);
    check_result<float>(f_lanczos.get_spacing());

    double d_temp = 1;
    boost::math::differentiation::discrete_lanczos_derivative d_lanczos(d_temp);
    check_result<double>(d_lanczos.get_spacing());
    
    #ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    long double ld_temp = 1;
    boost::math::differentiation::discrete_lanczos_derivative ld_lanczos(ld_temp);
    check_result<long double>(ld_lanczos.get_spacing());
    #endif
}

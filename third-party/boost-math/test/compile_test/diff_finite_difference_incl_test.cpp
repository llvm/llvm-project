//  Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header
// #includes all the files that it needs to.
#include <boost/math/differentiation/finite_difference.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
    check_result<float>(boost::math::differentiation::finite_difference_derivative([](float x){return x;}, f));
    check_result<double>(boost::math::differentiation::finite_difference_derivative([](double x){return x;}, d));
    #ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    check_result<long double>(boost::math::differentiation::finite_difference_derivative([](long double x){return x;}, static_cast<long double>(0)));
    #endif

    check_result<float>(boost::math::differentiation::complex_step_derivative([](std::complex<float> x){return x;}, f));
    check_result<double>(boost::math::differentiation::complex_step_derivative([](std::complex<double> x){return x;}, d));
    #ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    check_result<long double>(boost::math::differentiation::complex_step_derivative([](std::complex<long double> x){return x;}, 
                                                                                       static_cast<long double>(0)));
    #endif
}

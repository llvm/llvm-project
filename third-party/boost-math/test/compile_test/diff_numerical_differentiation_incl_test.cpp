//  Copyright Nick Thompson 2017.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/special_functions/gamma.hpp>
// #includes all the files that it needs to.
//
#include <boost/math/differentiation/finite_difference.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
    auto f = [](double x) { return x; };
    double x = 0;
    check_result<double>(boost::math::differentiation::finite_difference_derivative(f, x));

    auto g = [](std::complex<double> x){ return x;};
    check_result<double>(boost::math::differentiation::complex_step_derivative(g, x));
}

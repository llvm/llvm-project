//  Copyright Nick Thompson 2017.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header
// #includes all the files that it needs to.
//
#include <boost/math/quadrature/trapezoidal.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

double func(double x) { return x; }

void compile_and_link_test()
{
    double a = 0;
    double b = 1;
    check_result<double>(boost::math::quadrature::trapezoidal(func, a, b));
}

//  Copyright Nick Thompson 2017.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/special_functions/gamma.hpp>
// #includes all the files that it needs to.
//
#include <boost/math/differentiation/autodiff.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

template <typename T>
T fourth_power(T const& x) {
   T x4 = x * x;  // retval in operator*() uses x4's memory via NRVO.
   x4 *= x4;      // No copies of x4 are made within operator*=() even when squaring.
   return x4;     // x4 uses y's memory in main() via NRVO.
}

void compile_and_link_test()
{
   using namespace boost::math::differentiation;
   auto const x = make_fvar<double, 5>(2.0);  // Find derivatives at x=2.
   auto const y = fourth_power(x);
   
   check_result<double>(y.derivative(1));
}

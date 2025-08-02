//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/ccmath/modf.hpp>
#include "test_compile_result.hpp"

void compile_and_link_test()
{
   float i_f;
   check_result<float>(boost::math::ccmath::modf(1.0f, &i_f));

   double i_d;
   check_result<double>(boost::math::ccmath::modf(1.0, &i_d));

#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   long double i_ld;
   check_result<long double>(boost::math::ccmath::modf(1.0l, &i_ld));
#endif
}

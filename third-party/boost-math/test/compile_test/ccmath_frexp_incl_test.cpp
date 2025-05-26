//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/ccmath/frexp.hpp>
#include "test_compile_result.hpp"

void compile_and_link_test()
{
   int i;
   check_result<float>(boost::math::ccmath::frexp(1.0f, &i));
   check_result<double>(boost::math::ccmath::frexp(1.0, &i));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::ccmath::frexp(1.0l, &i));
#endif
}

//  (C) Copyright Matt Borland 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/ccmath/signbit.hpp>
#include "test_compile_result.hpp"

void compile_and_link_test()
{
   check_result<bool>(boost::math::ccmath::signbit(1.0F));
   check_result<bool>(boost::math::ccmath::signbit(1.0));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<bool>(boost::math::ccmath::signbit(1.0L));
#endif
}

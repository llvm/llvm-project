//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/ccmath/logb.hpp>
#include "test_compile_result.hpp"

void compile_and_link_test()
{
   check_result<float>(boost::math::ccmath::logb(1.0f));
   check_result<double>(boost::math::ccmath::logb(1.0));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::ccmath::logb(1.0l));
#endif
}

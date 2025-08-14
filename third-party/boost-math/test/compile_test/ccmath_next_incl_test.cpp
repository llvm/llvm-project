//  (C) Copyright Matt Borland 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/ccmath/next.hpp>
#include "test_compile_result.hpp"

void compile_and_link_test()
{
   check_result<float>(boost::math::ccmath::nextafter(1.0F, 1.05F));
   check_result<double>(boost::math::ccmath::nextafter(1.0, 1.0));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::ccmath::nexttoward(1.0L, 1.0L));
#endif
}

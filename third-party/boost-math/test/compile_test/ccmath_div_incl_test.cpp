//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/ccmath/div.hpp>
#include "test_compile_result.hpp"

void compile_and_link_test()
{
   check_result<std::div_t>(boost::math::ccmath::div(1, 1));
   check_result<std::ldiv_t>(boost::math::ccmath::div(1l, 1l));
   check_result<std::lldiv_t>(boost::math::ccmath::div(1ll, 1ll));
}

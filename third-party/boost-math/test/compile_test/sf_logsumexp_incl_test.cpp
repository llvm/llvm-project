//  Copyright Matt Borland 2022
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/special_functions/log1p.hpp>
// #includes all the files that it needs to.
//
#include <boost/math/special_functions/logsumexp.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
   check_result<float>(boost::math::logsumexp<float>(1.0f, 1.0f, 1.0f));
   check_result<double>(boost::math::logsumexp<double>(1.0, 1.0, 1.0));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::logsumexp<long double>(1.0l, 1.0l, 1.0l));
#endif
}

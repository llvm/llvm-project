//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/tools/polynomial.hpp>
// #includes all the files that it needs to.
//
#include <boost/math/tools/random_vector.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
    auto f_test = boost::math::generate_random_vector<float>(0, 128);
    check_result<float>(f_test.front());

    auto d_test = boost::math::generate_random_vector<double>(0, 128);
    check_result<double>(d_test.front());

#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    auto ld_test = boost::math::generate_random_vector<long double>(0, 128);
    check_result<long double>(ld_test.front());
#endif

    auto int_test = boost::math::generate_random_vector<int>(0, 128);
    check_result<int>(int_test.front());
}

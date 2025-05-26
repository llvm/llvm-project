//  Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/tools/polynomial_gcd.hpp>
// #includes all the files that it needs to.
//
#include <boost/math/tools/polynomial_gcd.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
    boost::math::tools::polynomial<int> p_int;
    check_result<int>(boost::math::tools::content(p_int));
    check_result<int>(boost::math::tools::leading_coefficient(p_int));
        
    boost::math::tools::polynomial<long> p_long;
    check_result<long>(boost::math::tools::content(p_long));
    check_result<long>(boost::math::tools::leading_coefficient(p_long));
}

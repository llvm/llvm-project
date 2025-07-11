//  Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header
// #includes all the files that it needs to.
//
#ifndef BOOST_MATH_STANDALONE
#include <boost/math/tools/convert_from_string.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
    const char* val_c_str = "0.0";
    
    check_result<float>(boost::math::tools::convert_from_string<float>(val_c_str));
    check_result<double>(boost::math::tools::convert_from_string<double>(val_c_str));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    check_result<long double>(boost::math::tools::convert_from_string<long double>(val_c_str));
#endif
}

#endif

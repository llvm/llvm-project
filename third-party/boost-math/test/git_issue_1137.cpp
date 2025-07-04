//  Copyright Matt Borland 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  See: https://github.com/boostorg/math/issues/1137

#define BOOST_MATH_POLY_METHOD 0
#define BOOST_MATH_RATIONAL_METHOD 0

#include <boost/math/tools/config.hpp>

int main()
{
    static_assert(BOOST_MATH_POLY_METHOD == 0, "User defined as 0");
    static_assert(BOOST_MATH_RATIONAL_METHOD == 0, "User defined as 0");

    return 0;
}

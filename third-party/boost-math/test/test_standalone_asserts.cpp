//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_STANDALONE
#define BOOST_MATH_STANDALONE
#endif

#include <boost/math/tools/assert.hpp>

int main(void)
{
    constexpr unsigned two = 2;
    
    BOOST_MATH_ASSERT(two == 2);
    BOOST_MATH_ASSERT_MSG(two == 2, "Fails");

    BOOST_MATH_STATIC_ASSERT(two == 2);
    BOOST_MATH_STATIC_ASSERT_MSG(two == 2, "Fails");
}

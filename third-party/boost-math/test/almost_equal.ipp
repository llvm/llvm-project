#ifndef BOOST_MATH_ALMOST_EQUAL_HPP
#define BOOST_MATH_ALMOST_EQUAL_HPP

// Copyright (c) 2006 Johan Rade

// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cmath>

template<class ValType>
bool almost_equal(ValType a, ValType b)
{
    const ValType e = static_cast<ValType>(0.00001);
    return (a - e * std::abs(a) <= b + e * std::abs(b))
        && (a + e * std::abs(a) >= b - e * std::abs(b));
}

#endif

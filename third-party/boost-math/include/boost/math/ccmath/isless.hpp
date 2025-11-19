//  (C) Copyright Matt Borland 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_CCMATH_ISLESS_HPP
#define BOOST_MATH_CCMATH_ISLESS_HPP

#include <boost/math/ccmath/detail/config.hpp>

#ifdef BOOST_MATH_NO_CCMATH
#error "The header <boost/math/isless.hpp> can only be used in C++17 and later."
#endif

#include <boost/math/ccmath/isnan.hpp>

namespace boost::math::ccmath {

template <typename T1, typename T2 = T1>
inline constexpr bool isless(T1 x, T2 y) noexcept
{
    if (BOOST_MATH_IS_CONSTANT_EVALUATED(x))
    {
        if (boost::math::ccmath::isnan(x) || boost::math::ccmath::isnan(y))
        {
            return false;
        }
        else
        {
            return x < y;
        }
    }
    else
    {
        using std::isless;
        return isless(x, y);
    }
}

} // Namespaces

#endif // BOOST_MATH_CCMATH_ISLESS_HPP

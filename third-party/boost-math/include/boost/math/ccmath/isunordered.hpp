//  (C) Copyright Matt Borland 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_CCMATH_ISUNORDERED_HPP
#define BOOST_MATH_CCMATH_ISUNORDERED_HPP

#include <boost/math/ccmath/detail/config.hpp>

#ifdef BOOST_MATH_NO_CCMATH
#error "The header <boost/math/isunordered.hpp> can only be used in C++17 and later."
#endif

#include <boost/math/ccmath/isnan.hpp>

namespace boost::math::ccmath {

template <typename T>
inline constexpr bool isunordered(const T x, const T y) noexcept
{
    if(BOOST_MATH_IS_CONSTANT_EVALUATED(x))
    {
        return boost::math::ccmath::isnan(x) || boost::math::ccmath::isnan(y);
    }
    else
    {
        using std::isunordered;
        return isunordered(x, y);
    }
}

} // Namespaces

#endif // BOOST_MATH_CCMATH_ISUNORDERED_HPP

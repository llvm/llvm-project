//  Copyright (c) 2024 Matt Borland
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TOOLS_UTILITY
#define BOOST_MATH_TOOLS_UTILITY

#include <boost/math/tools/config.hpp>

#ifndef BOOST_MATH_HAS_GPU_SUPPORT

#include <utility>

namespace boost {
namespace math {

template <typename T>
constexpr T min BOOST_MATH_PREVENT_MACRO_SUBSTITUTION (const T& a, const T& b)
{
    return (std::min)(a, b);
}

template <typename T>
constexpr T max BOOST_MATH_PREVENT_MACRO_SUBSTITUTION (const T& a, const T& b)
{
    return (std::max)(a, b);
}

template <typename T>
void swap BOOST_MATH_PREVENT_MACRO_SUBSTITUTION (T& a, T& b)
{
    return (std::swap)(a, b);
}

} // namespace math
} // namespace boost

#else

namespace boost {
namespace math {

template <typename T>
BOOST_MATH_GPU_ENABLED constexpr T min BOOST_MATH_PREVENT_MACRO_SUBSTITUTION (const T& a, const T& b)
{ 
    return a < b ? a : b; 
}

template <typename T>
BOOST_MATH_GPU_ENABLED constexpr T max BOOST_MATH_PREVENT_MACRO_SUBSTITUTION (const T& a, const T& b)
{ 
    return a > b ? a : b;
}

template <typename T>
BOOST_MATH_GPU_ENABLED constexpr void swap BOOST_MATH_PREVENT_MACRO_SUBSTITUTION (T& a, T& b)
{ 
    T t(a); 
    a = b; 
    b = t;
}

} // namespace math
} // namespace boost

#endif // BOOST_MATH_HAS_GPU_SUPPORT

#endif // BOOST_MATH_TOOLS_UTILITY

//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_CCMATH_DETAIL_SWAP_HPP
#define BOOST_MATH_CCMATH_DETAIL_SWAP_HPP

namespace boost::math::ccmath::detail {

template <typename T>
inline constexpr void swap(T& x, T& y) noexcept
{
    T temp = x;
    x = y;
    y = temp;
}

}

#endif // BOOST_MATH_CCMATH_DETAIL_SWAP_HPP

//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_CCMATH_FLOOR_HPP
#define BOOST_MATH_CCMATH_FLOOR_HPP

#include <boost/math/ccmath/detail/config.hpp>

#ifdef BOOST_MATH_NO_CCMATH
#error "The header <boost/math/floor.hpp> can only be used in C++17 and later."
#endif

#include <boost/math/ccmath/abs.hpp>
#include <boost/math/ccmath/isinf.hpp>
#include <boost/math/ccmath/isnan.hpp>
#include <limits>

namespace boost::math::ccmath {

namespace detail {

template <typename T>
inline constexpr T floor_pos_impl(T arg) noexcept
{
    constexpr auto max_comp_val = T(1) / std::numeric_limits<T>::epsilon();

    if (arg >= max_comp_val)
    {
        return arg;
    }

    T result = 1;

    if(result <= arg)
    {
        while(result < arg)
        {
            result *= 2;
        }
        while(result > arg)
        {
            --result;
        }

        return result;
    }
    else
    {
        return T(0);
    }
}

template <typename T>
inline constexpr T floor_neg_impl(T arg) noexcept
{
    T result = -1;

    if(result > arg)
    {
        while(result > arg)
        {
            result *= 2;
        }
        while(result < arg)
        {
            ++result;
        }
        if(result != arg)
        {
            --result;
        }
    }

    return result;
}

template <typename T>
inline constexpr T floor_impl(T arg) noexcept
{
    if(arg > 0)
    {
        return floor_pos_impl(arg);
    }
    else
    {
        return floor_neg_impl(arg);
    }
}

} // Namespace detail

template <typename Real, std::enable_if_t<!std::is_integral_v<Real>, bool> = true>
inline constexpr Real floor(Real arg) noexcept
{
    if(BOOST_MATH_IS_CONSTANT_EVALUATED(arg))
    {
        return boost::math::ccmath::abs(arg) == Real(0) ? arg :
               boost::math::ccmath::isinf(arg) ? arg :
               boost::math::ccmath::isnan(arg) ? arg :
               boost::math::ccmath::detail::floor_impl(arg);
    }
    else
    {
        using std::floor;
        return floor(arg);
    }
}

template <typename Z, std::enable_if_t<std::is_integral_v<Z>, bool> = true>
inline constexpr double floor(Z arg) noexcept
{
    return boost::math::ccmath::floor(static_cast<double>(arg));
}

inline constexpr float floorf(float arg) noexcept
{
    return boost::math::ccmath::floor(arg);
}

#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
inline constexpr long double floorl(long double arg) noexcept
{
    return boost::math::ccmath::floor(arg);
}
#endif

} // Namespaces

#endif // BOOST_MATH_CCMATH_FLOOR_HPP

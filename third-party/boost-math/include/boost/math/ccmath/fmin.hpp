//  (C) Copyright Matt Borland 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_CCMATH_FMIN_HPP
#define BOOST_MATH_CCMATH_FMIN_HPP

#include <boost/math/ccmath/detail/config.hpp>

#ifdef BOOST_MATH_NO_CCMATH
#error "The header <boost/math/fmin.hpp> can only be used in C++17 and later."
#endif

#include <boost/math/tools/promotion.hpp>
#include <boost/math/ccmath/isnan.hpp>

namespace boost::math::ccmath {

namespace detail {

template <typename T>
constexpr T fmin_impl(const T x, const T y) noexcept
{
    if (x < y)
    {
        return x;
    }
    else
    {
        return y;
    }
}

} // Namespace detail

template <typename Real, std::enable_if_t<!std::is_integral_v<Real>, bool> = true>
constexpr Real fmin(Real x, Real y) noexcept
{
    if (BOOST_MATH_IS_CONSTANT_EVALUATED(x))
    {
        if (boost::math::ccmath::isnan(x))
        {
            return y;
        }
        else if (boost::math::ccmath::isnan(y))
        {
            return x;
        }
        
        return boost::math::ccmath::detail::fmin_impl(x, y);
    }
    else
    {
        using std::fmin;
        return fmin(x, y);
    }
}

template <typename T1, typename T2>
constexpr auto fmin(T1 x, T2 y) noexcept
{
    if (BOOST_MATH_IS_CONSTANT_EVALUATED(x))
    {
        using promoted_type = boost::math::tools::promote_args_t<T1, T2>;
        return boost::math::ccmath::fmin(static_cast<promoted_type>(x), static_cast<promoted_type>(y));
    }
    else
    {
        using std::fmin;
        return fmin(x, y);
    }
}

constexpr float fminf(float x, float y) noexcept
{
    return boost::math::ccmath::fmin(x, y);
}

#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
constexpr long double fminl(long double x, long double y) noexcept
{
    return boost::math::ccmath::fmin(x, y);
}
#endif

} // Namespace boost::math::ccmath

#endif // BOOST_MATH_CCMATH_FMIN_HPP

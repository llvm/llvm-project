//  (C) Copyright Matt Borland 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_CCMATH_FDIM_HPP
#define BOOST_MATH_CCMATH_FDIM_HPP

#include <boost/math/ccmath/detail/config.hpp>

#ifdef BOOST_MATH_NO_CCMATH
#error "The header <boost/math/fdim.hpp> can only be used in C++17 and later."
#endif

#include <boost/math/tools/promotion.hpp>
#include <boost/math/ccmath/isnan.hpp>

namespace boost::math::ccmath {

namespace detail {

template <typename T>
constexpr T fdim_impl(const T x, const T y) noexcept
{
    if (x <= y)
    {
        return 0;
    }
    else if ((y < 0) && (x > (std::numeric_limits<T>::max)() + y))
    {
        return std::numeric_limits<T>::infinity();
    }
    else
    {
        return x - y;
    }
}

} // Namespace detail

template <typename Real, std::enable_if_t<!std::is_integral_v<Real>, bool> = true>
constexpr Real fdim(Real x, Real y) noexcept
{
    if (BOOST_MATH_IS_CONSTANT_EVALUATED(x))
    {
        if (boost::math::ccmath::isnan(x))
        {
            return x;
        }
        else if (boost::math::ccmath::isnan(y))
        {
            return y;
        }

        return boost::math::ccmath::detail::fdim_impl(x, y);
    }
    else
    {
        using std::fdim;
        return fdim(x, y);
    }
}

template <typename T1, typename T2>
constexpr auto fdim(T1 x, T2 y) noexcept
{
    if (BOOST_MATH_IS_CONSTANT_EVALUATED(x))
    {
        using promoted_type = boost::math::tools::promote_args_t<T1, T2>;
        return boost::math::ccmath::fdim(promoted_type(x), promoted_type(y));
    }
    else
    {
        using std::fdim;
        return fdim(x, y);
    }
}

constexpr float fdimf(float x, float y) noexcept
{
    return boost::math::ccmath::fdim(x, y);
}

#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
constexpr long double fdiml(long double x, long double y) noexcept
{
    return boost::math::ccmath::fdim(x, y);
}
#endif

} // Namespace boost::math::ccmath

#endif // BOOST_MATH_CCMATH_FDIM_HPP

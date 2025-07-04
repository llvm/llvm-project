//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_CCMATH_CEIL_HPP
#define BOOST_MATH_CCMATH_CEIL_HPP

#include <boost/math/ccmath/detail/config.hpp>

#ifdef BOOST_MATH_NO_CCMATH
#error "The header <boost/math/ceil.hpp> can only be used in C++17 and later."
#endif

#include <boost/math/ccmath/floor.hpp>
#include <boost/math/ccmath/abs.hpp>
#include <boost/math/ccmath/isinf.hpp>
#include <boost/math/ccmath/isnan.hpp>

namespace boost::math::ccmath {

namespace detail {

template <typename T>
inline constexpr T ceil_impl(T arg) noexcept
{
    T result = boost::math::ccmath::floor(arg);

    if(result == arg)
    {
        return result;
    }
    else
    {
        return result + 1;
    }
}

} // Namespace detail

template <typename Real, std::enable_if_t<!std::is_integral_v<Real>, bool> = true>
inline constexpr Real ceil(Real arg) noexcept
{
    if(BOOST_MATH_IS_CONSTANT_EVALUATED(arg))
    {
        return boost::math::ccmath::abs(arg) == Real(0) ? arg :
               boost::math::ccmath::isinf(arg) ? arg :
               boost::math::ccmath::isnan(arg) ? arg :
               boost::math::ccmath::detail::ceil_impl(arg);
    }
    else
    {
        using std::ceil;
        return ceil(arg);
    }
}

template <typename Z, std::enable_if_t<std::is_integral_v<Z>, bool> = true>
inline constexpr double ceil(Z arg) noexcept
{
    return boost::math::ccmath::ceil(static_cast<double>(arg));
}

inline constexpr float ceilf(float arg) noexcept
{
    return boost::math::ccmath::ceil(arg);
}

#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
inline constexpr long double ceill(long double arg) noexcept
{
    return boost::math::ccmath::ceil(arg);
}
#endif

} // Namespaces

#endif // BOOST_MATH_CCMATH_CEIL_HPP

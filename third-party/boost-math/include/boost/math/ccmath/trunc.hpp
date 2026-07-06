//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_CCMATH_TRUNC_HPP
#define BOOST_MATH_CCMATH_TRUNC_HPP

#include <boost/math/ccmath/detail/config.hpp>

#ifdef BOOST_MATH_NO_CCMATH
#error "The header <boost/math/trunc.hpp> can only be used in C++17 and later."
#endif

#include <boost/math/ccmath/abs.hpp>
#include <boost/math/ccmath/isinf.hpp>
#include <boost/math/ccmath/isnan.hpp>
#include <boost/math/ccmath/floor.hpp>
#include <boost/math/ccmath/ceil.hpp>

namespace boost::math::ccmath {

namespace detail {

template <typename T>
inline constexpr T trunc_impl(T arg) noexcept
{
    return (arg > 0) ? boost::math::ccmath::floor(arg) : boost::math::ccmath::ceil(arg);
}

} // Namespace detail

template <typename Real, std::enable_if_t<!std::is_integral_v<Real>, bool> = true>
inline constexpr Real trunc(Real arg) noexcept
{
    if(BOOST_MATH_IS_CONSTANT_EVALUATED(arg))
    {
        return boost::math::ccmath::abs(arg) == Real(0) ? arg :
               boost::math::ccmath::isinf(arg) ? arg :
               boost::math::ccmath::isnan(arg) ? arg :
               boost::math::ccmath::detail::trunc_impl(arg);
    }
    else
    {
        using std::trunc;
        return trunc(arg);
    }
}

template <typename Z, std::enable_if_t<std::is_integral_v<Z>, bool> = true>
inline constexpr double trunc(Z arg) noexcept
{
    return boost::math::ccmath::trunc(static_cast<double>(arg));
}

inline constexpr float truncf(float arg) noexcept
{
    return boost::math::ccmath::trunc(arg);
}

#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
inline constexpr long double truncl(long double arg) noexcept
{
    return boost::math::ccmath::trunc(arg);
}
#endif

} // Namespaces

#endif // BOOST_MATH_CCMATH_TRUNC_HPP

//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_CCMATH_ROUND_HPP
#define BOOST_MATH_CCMATH_ROUND_HPP

#include <stdexcept>
#include <boost/math/ccmath/detail/config.hpp>

#ifdef BOOST_MATH_NO_CCMATH
#error "The header <boost/math/round.hpp> can only be used in C++17 and later."
#endif

#include <boost/math/ccmath/abs.hpp>
#include <boost/math/ccmath/isinf.hpp>
#include <boost/math/ccmath/isnan.hpp>
#include <boost/math/ccmath/modf.hpp>

namespace boost::math::ccmath {

namespace detail {

// Computes the nearest integer value to arg (in floating-point format), 
// rounding halfway cases away from zero, regardless of the current rounding mode.
template <typename T>
inline constexpr T round_impl(T arg) noexcept
{
    T iptr = 0;
    const T x = boost::math::ccmath::modf(arg, &iptr);
    constexpr T half = T(1)/2;

    if(x >= half && iptr > 0)
    {
        return iptr + 1;
    }
    else if(boost::math::ccmath::abs(x) >= half && iptr < 0)
    {
        return iptr - 1;
    }
    else
    {
        return iptr;
    }
}

template <typename ReturnType, typename T>
inline constexpr ReturnType int_round_impl(T arg)
{
    const T rounded_arg = round_impl(arg);

    if(rounded_arg > static_cast<T>((std::numeric_limits<ReturnType>::max)()))
    {
        if constexpr (std::is_same_v<ReturnType, long long>)
        {
            throw std::domain_error("Rounded value cannot be represented by a long long type without overflow");
        }
        else
        {
            throw std::domain_error("Rounded value cannot be represented by a long type without overflow");
        }
    }
    else
    {
        return static_cast<ReturnType>(rounded_arg);
    }
}

} // Namespace detail

template <typename Real, std::enable_if_t<!std::is_integral_v<Real>, bool> = true>
inline constexpr Real round(Real arg) noexcept
{
    if(BOOST_MATH_IS_CONSTANT_EVALUATED(arg))
    {
        return boost::math::ccmath::abs(arg) == Real(0) ? arg :
               boost::math::ccmath::isinf(arg) ? arg :
               boost::math::ccmath::isnan(arg) ? arg :
               boost::math::ccmath::detail::round_impl(arg);
    }
    else
    {
        using std::round;
        return round(arg);
    }
}

template <typename Z, std::enable_if_t<std::is_integral_v<Z>, bool> = true>
inline constexpr double round(Z arg) noexcept
{
    return boost::math::ccmath::round(static_cast<double>(arg));
}

inline constexpr float roundf(float arg) noexcept
{
    return boost::math::ccmath::round(arg);
}

#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
inline constexpr long double roundl(long double arg) noexcept
{
    return boost::math::ccmath::round(arg);
}
#endif

template <typename Real, std::enable_if_t<!std::is_integral_v<Real>, bool> = true>
inline constexpr long lround(Real arg)
{
    if(BOOST_MATH_IS_CONSTANT_EVALUATED(arg))
    {
        return boost::math::ccmath::abs(arg) == Real(0) ? 0l :
               boost::math::ccmath::isinf(arg) ? 0l :
               boost::math::ccmath::isnan(arg) ? 0l :
               boost::math::ccmath::detail::int_round_impl<long>(arg);
    }
    else
    {
        using std::lround;
        return lround(arg);
    }
}

template <typename Z, std::enable_if_t<std::is_integral_v<Z>, bool> = true>
inline constexpr long lround(Z arg)
{
    return boost::math::ccmath::lround(static_cast<double>(arg));
}

inline constexpr long lroundf(float arg)
{
    return boost::math::ccmath::lround(arg);
}

#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
inline constexpr long lroundl(long double arg)
{
    return boost::math::ccmath::lround(arg);
}
#endif

template <typename Real, std::enable_if_t<!std::is_integral_v<Real>, bool> = true>
inline constexpr long long llround(Real arg)
{
    if(BOOST_MATH_IS_CONSTANT_EVALUATED(arg))
    {
        return boost::math::ccmath::abs(arg) == Real(0) ? 0ll :
               boost::math::ccmath::isinf(arg) ? 0ll :
               boost::math::ccmath::isnan(arg) ? 0ll :
               boost::math::ccmath::detail::int_round_impl<long long>(arg);
    }
    else
    {
        using std::llround;
        return llround(arg);
    }
}

template <typename Z, std::enable_if_t<std::is_integral_v<Z>, bool> = true>
inline constexpr long llround(Z arg)
{
    return boost::math::ccmath::llround(static_cast<double>(arg));
}

inline constexpr long long llroundf(float arg)
{
    return boost::math::ccmath::llround(arg);
}

#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
inline constexpr long long llroundl(long double arg)
{
    return boost::math::ccmath::llround(arg);
}
#endif

} // Namespaces

#endif // BOOST_MATH_CCMATH_ROUND_HPP

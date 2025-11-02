//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_CCMATH_COPYSIGN_HPP
#define BOOST_MATH_CCMATH_COPYSIGN_HPP

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <boost/math/tools/is_constant_evaluated.hpp>
#include <boost/math/tools/promotion.hpp>
#include <boost/math/tools/config.hpp>
#include <boost/math/ccmath/abs.hpp>
#include <boost/math/ccmath/signbit.hpp>

namespace boost::math::ccmath {

namespace detail {

template <typename T>
constexpr T copysign_impl(const T mag, const T sgn) noexcept
{
    if (boost::math::ccmath::signbit(sgn))
    {
        return -boost::math::ccmath::abs(mag);
    }
    else
    {
        return boost::math::ccmath::abs(mag);
    }
}

} // Namespace detail

template <typename Real, std::enable_if_t<!std::is_integral_v<Real>, bool> = true>
constexpr Real copysign(Real mag, Real sgn) noexcept
{
    if(BOOST_MATH_IS_CONSTANT_EVALUATED(mag))
    {
        return boost::math::ccmath::detail::copysign_impl(mag, sgn);
    }
    else
    {
        using std::copysign;
        return copysign(mag, sgn);
    }
}

template <typename T1, typename T2>
constexpr auto copysign(T1 mag, T2 sgn) noexcept
{
    if (BOOST_MATH_IS_CONSTANT_EVALUATED(mag))
    {        
        using promoted_type = boost::math::tools::promote_args_t<T1, T2>;
        return boost::math::ccmath::copysign(static_cast<promoted_type>(mag), static_cast<promoted_type>(sgn));
    }
    else
    {
        using std::copysign;
        return copysign(mag, sgn);
    }
}

constexpr float copysignf(float mag, float sgn) noexcept
{
    return boost::math::ccmath::copysign(mag, sgn);
}

#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
constexpr long double copysignl(long double mag, long double sgn) noexcept
{
    return boost::math::ccmath::copysign(mag, sgn);
}
#endif

} // Namespaces

#endif // BOOST_MATH_CCMATH_COPYSIGN_HPP

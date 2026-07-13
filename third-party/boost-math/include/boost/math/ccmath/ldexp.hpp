//  (C) Copyright Matt Borland 2021.
//  (C) Copyright John Maddock 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_CCMATH_LDEXP_HPP
#define BOOST_MATH_CCMATH_LDEXP_HPP

#include <boost/math/ccmath/detail/config.hpp>

#ifdef BOOST_MATH_NO_CCMATH
#error "The header <boost/math/ldexp.hpp> can only be used in C++17 and later."
#endif

#include <stdexcept>
#include <boost/math/ccmath/abs.hpp>
#include <boost/math/ccmath/isinf.hpp>
#include <boost/math/ccmath/isnan.hpp>

namespace boost::math::ccmath {

namespace detail {

template <typename Real>
inline constexpr Real ldexp_impl(Real arg, int exp) noexcept
{
    while(exp > 0)
    {
        arg *= 2;
        --exp;
    }
    while(exp < 0)
    {
        arg /= 2;
        ++exp;
    }

    return arg;
}

} // Namespace detail

template <typename Real, std::enable_if_t<!std::is_integral_v<Real>, bool> = true>
inline constexpr Real ldexp(Real arg, int exp) noexcept
{
    if(BOOST_MATH_IS_CONSTANT_EVALUATED(arg))
    {
        return boost::math::ccmath::abs(arg) == Real(0) ? arg :
               (boost::math::ccmath::isinf)(arg) ? arg :
               (boost::math::ccmath::isnan)(arg) ? arg :
               boost::math::ccmath::detail::ldexp_impl(arg, exp);
    }
    else
    {
        using std::ldexp;
        return ldexp(arg, exp);
    }
}

template <typename Z, std::enable_if_t<std::is_integral_v<Z>, bool> = true>
inline constexpr double ldexp(Z arg, int exp) noexcept
{
    return boost::math::ccmath::ldexp(static_cast<double>(arg), exp);
}

inline constexpr float ldexpf(float arg, int exp) noexcept
{
    return boost::math::ccmath::ldexp(arg, exp);
}

#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
inline constexpr long double ldexpl(long double arg, int exp) noexcept
{
    return boost::math::ccmath::ldexp(arg, exp);
}
#endif

} // Namespaces

#endif // BOOST_MATH_CCMATH_LDEXP_HPP

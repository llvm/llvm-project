//  (C) Copyright Christopher Kormanyos 1999 - 2021.
//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_CCMATH_FREXP_HPP
#define BOOST_MATH_CCMATH_FREXP_HPP

#include <boost/math/ccmath/detail/config.hpp>

#ifdef BOOST_MATH_NO_CCMATH
#error "The header <boost/math/frexp.hpp> can only be used in C++17 and later."
#endif

#include <boost/math/ccmath/isinf.hpp>
#include <boost/math/ccmath/isnan.hpp>
#include <boost/math/ccmath/isfinite.hpp>

namespace boost::math::ccmath {

namespace detail
{

template <typename Real>
inline constexpr Real frexp_zero_impl(Real arg, int* exp)
{
    *exp = 0;
    return arg;
}

template <typename Real>
inline constexpr Real frexp_impl(Real arg, int* exp)
{
    const bool negative_arg = (arg < Real(0));
    
    Real f = negative_arg ? -arg : arg;
    int e2 = 0;
    constexpr Real two_pow_32 = Real(4294967296);

    while (f >= two_pow_32)
    {
        f = f / two_pow_32;
        e2 += 32;
    }

    while(f >= Real(1))
    {
        f = f / Real(2);
        ++e2;
    }
    
    if(exp != nullptr)
    {
        *exp = e2;
    }

    return !negative_arg ? f : -f;
}

} // namespace detail

template <typename Real, std::enable_if_t<!std::is_integral_v<Real>, bool> = true>
inline constexpr Real frexp(Real arg, int* exp)
{
    if(BOOST_MATH_IS_CONSTANT_EVALUATED(arg))
    {
        return arg == Real(0)  ? detail::frexp_zero_impl(arg, exp) : 
               arg == Real(-0) ? detail::frexp_zero_impl(arg, exp) :
               boost::math::ccmath::isinf(arg) ? detail::frexp_zero_impl(arg, exp) : 
               boost::math::ccmath::isnan(arg) ? detail::frexp_zero_impl(arg, exp) :
               boost::math::ccmath::detail::frexp_impl(arg, exp);
    }
    else
    {
        using std::frexp;
        return frexp(arg, exp);
    }
}

template <typename Z, std::enable_if_t<std::is_integral_v<Z>, bool> = true>
inline constexpr double frexp(Z arg, int* exp)
{
    return boost::math::ccmath::frexp(static_cast<double>(arg), exp);
}

inline constexpr float frexpf(float arg, int* exp)
{
    return boost::math::ccmath::frexp(arg, exp);
}

#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
inline constexpr long double frexpl(long double arg, int* exp)
{
    return boost::math::ccmath::frexp(arg, exp);
}
#endif

}

#endif // BOOST_MATH_CCMATH_FREXP_HPP

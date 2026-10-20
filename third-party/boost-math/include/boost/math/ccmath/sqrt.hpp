//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Constexpr implementation of sqrt function

#ifndef BOOST_MATH_CCMATH_SQRT
#define BOOST_MATH_CCMATH_SQRT

#include <boost/math/ccmath/detail/config.hpp>

#ifdef BOOST_MATH_NO_CCMATH
#error "The header <boost/math/sqrt.hpp> can only be used in C++17 and later."
#endif

#include <boost/math/ccmath/abs.hpp>
#include <boost/math/ccmath/isnan.hpp>
#include <boost/math/ccmath/isinf.hpp>
#include <boost/math/tools/is_constant_evaluated.hpp>

namespace boost::math::ccmath { 

namespace detail {

template <typename Real>
constexpr Real sqrt_impl_2(Real x, Real s, Real s2)
{
    return !(s < s2) ? s2 : sqrt_impl_2(x, (x / s + s) / 2, s);
}

template <typename Real>
constexpr Real sqrt_impl_1(Real x, Real s)
{
    return sqrt_impl_2(x, (x / s + s) / 2, s);
}

template <typename Real>
constexpr Real sqrt_impl(Real x)
{
    return sqrt_impl_1(x, x > 1 ? x : Real(1));
}

} // namespace detail

template <typename Real, std::enable_if_t<!std::is_integral_v<Real>, bool> = true>
constexpr Real sqrt(Real x)
{
    if(BOOST_MATH_IS_CONSTANT_EVALUATED(x))
    {
        if (boost::math::ccmath::isnan(x) || 
           (boost::math::ccmath::isinf(x) && x > 0) ||
            boost::math::ccmath::abs(x) == Real(0))
        {
            return x;
        }
        // Domain error is implementation defined so return NAN
        else if (boost::math::ccmath::isinf(x) && x < 0)
        {
            return std::numeric_limits<Real>::quiet_NaN();
        }

        return detail::sqrt_impl<Real>(x);
    }
    else
    {
        using std::sqrt;
        return sqrt(x);
    }
}

template <typename Z, std::enable_if_t<std::is_integral_v<Z>, bool> = true>
constexpr double sqrt(Z x)
{
    return detail::sqrt_impl<double>(static_cast<double>(x));
}

} // Namespaces

#endif // BOOST_MATH_CCMATH_SQRT

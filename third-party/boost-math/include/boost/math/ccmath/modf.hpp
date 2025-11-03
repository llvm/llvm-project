//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_CCMATH_MODF_HPP
#define BOOST_MATH_CCMATH_MODF_HPP

#include <boost/math/ccmath/detail/config.hpp>

#ifdef BOOST_MATH_NO_CCMATH
#error "The header <boost/math/modf.hpp> can only be used in C++17 and later."
#endif

#include <boost/math/ccmath/abs.hpp>
#include <boost/math/ccmath/isinf.hpp>
#include <boost/math/ccmath/isnan.hpp>
#include <boost/math/ccmath/trunc.hpp>

namespace boost::math::ccmath {

namespace detail {

template <typename Real>
inline constexpr Real modf_error_impl(Real x, Real* iptr)
{
    *iptr = x;
    return boost::math::ccmath::abs(x) == Real(0) ? x :
           x > Real(0) ? Real(0) : -Real(0);
}

template <typename Real>
inline constexpr Real modf_nan_impl(Real x, Real* iptr)
{
    *iptr = x;
    return x;
}

template <typename Real>
inline constexpr Real modf_impl(Real x, Real* iptr)
{
    *iptr = boost::math::ccmath::trunc(x);
    return (x - *iptr);
}

} // Namespace detail

template <typename Real>
inline constexpr Real modf(Real x, Real* iptr)
{
    if(BOOST_MATH_IS_CONSTANT_EVALUATED(x))
    {
        return boost::math::ccmath::abs(x) == Real(0) ? detail::modf_error_impl(x, iptr) :
               boost::math::ccmath::isinf(x) ? detail::modf_error_impl(x, iptr) :
               boost::math::ccmath::isnan(x) ? detail::modf_nan_impl(x, iptr) :
               boost::math::ccmath::detail::modf_impl(x, iptr);
    }
    else
    {
        using std::modf;
        return modf(x, iptr);
    }
}

inline constexpr float modff(float x, float* iptr)
{
    return boost::math::ccmath::modf(x, iptr);
}

#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
inline constexpr long double modfl(long double x, long double* iptr)
{
    return boost::math::ccmath::modf(x, iptr);
}
#endif

} // Namespaces

#endif // BOOST_MATH_CCMATH_MODF_HPP

//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_CCMATH_SCALBLN_HPP
#define BOOST_MATH_CCMATH_SCALBLN_HPP

#include <boost/math/ccmath/detail/config.hpp>

#ifdef BOOST_MATH_NO_CCMATH
#error "The header <boost/math/scalbln.hpp> can only be used in C++17 and later."
#endif

#include <cfloat>
#include <boost/math/ccmath/scalbn.hpp>
#include <boost/math/ccmath/abs.hpp>
#include <boost/math/ccmath/isinf.hpp>
#include <boost/math/ccmath/isnan.hpp>

namespace boost::math::ccmath {

template <typename Real, std::enable_if_t<!std::is_integral_v<Real>, bool> = true>
inline constexpr Real scalbln(Real arg, long exp) noexcept
{
    if(BOOST_MATH_IS_CONSTANT_EVALUATED(arg))
    {
        return boost::math::ccmath::abs(arg) == Real(0) ? arg :
               boost::math::ccmath::isinf(arg) ? arg :
               boost::math::ccmath::isnan(arg) ? arg :
               boost::math::ccmath::detail::scalbn_impl(arg, exp);
    }
    else
    {
        using std::scalbln;
        return scalbln(arg, exp);
    }
}

template <typename Z, std::enable_if_t<std::is_integral_v<Z>, bool> = true>
inline constexpr double scalbln(Z arg, long exp) noexcept
{
    return boost::math::ccmath::scalbln(static_cast<double>(arg), exp);
}

inline constexpr float scalblnf(float arg, long exp) noexcept
{
    return boost::math::ccmath::scalbln(arg, exp);
}

#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
inline constexpr long double scalblnl(long double arg, long exp) noexcept
{
    return boost::math::ccmath::scalbln(arg, exp);
}
#endif

} // Namespaces

#endif // BOOST_MATH_CCMATH_SCALBLN_HPP

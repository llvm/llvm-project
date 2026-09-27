//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_CCMATH_ISFINITE
#define BOOST_MATH_CCMATH_ISFINITE

#include <boost/math/ccmath/detail/config.hpp>

#ifdef BOOST_MATH_NO_CCMATH
#error "The header <boost/math/isfinite.hpp> can only be used in C++17 and later."
#endif

#include <boost/math/ccmath/isinf.hpp>
#include <boost/math/ccmath/isnan.hpp>

namespace boost::math::ccmath {

template <typename T>
inline constexpr bool isfinite(T x)
{
    if(BOOST_MATH_IS_CONSTANT_EVALUATED(x))
    {
        // bool isfinite (IntegralType arg) is a set of overloads accepting the arg argument of any integral type
        // equivalent to casting the integral argument arg to double (e.g. static_cast<double>(arg))
        if constexpr (std::is_integral_v<T>)
        {
            return !boost::math::ccmath::isinf(static_cast<double>(x)) && !boost::math::ccmath::isnan(static_cast<double>(x));
        }
        else
        {
            return !boost::math::ccmath::isinf(x) && !boost::math::ccmath::isnan(x);
        }
    }
    else
    {
        using std::isfinite;

        if constexpr (!std::is_integral_v<T>)
        {
            return isfinite(x);
        }
        else
        {
            return isfinite(static_cast<double>(x));
        }
    }
}

}

#endif // BOOST_MATH_CCMATH_ISFINITE

//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_CCMATH_FPCLASSIFY
#define BOOST_MATH_CCMATH_FPCLASSIFY

#include <boost/math/ccmath/detail/config.hpp>

#ifdef BOOST_MATH_NO_CCMATH
#error "The header <boost/math/fpclassify.hpp> can only be used in C++17 and later."
#endif

#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/math/ccmath/abs.hpp>
#include <boost/math/ccmath/isinf.hpp>
#include <boost/math/ccmath/isnan.hpp>
#include <boost/math/ccmath/isfinite.hpp>

namespace boost::math::ccmath {

template <typename T, std::enable_if_t<!std::is_integral_v<T>, bool> = true>
inline constexpr int fpclassify BOOST_MATH_PREVENT_MACRO_SUBSTITUTION(T x)
{
    if(BOOST_MATH_IS_CONSTANT_EVALUATED(x))
    {
        return (boost::math::ccmath::isnan)(x) ? FP_NAN :
               (boost::math::ccmath::isinf)(x) ? FP_INFINITE :
               boost::math::ccmath::abs(x) == T(0) ? FP_ZERO :
               boost::math::ccmath::abs(x) > 0 && boost::math::ccmath::abs(x) < (std::numeric_limits<T>::min)() ? FP_SUBNORMAL : FP_NORMAL;
    }
    else
    {
        using boost::math::fpclassify;
        return (fpclassify)(x);
    }
}

template <typename Z, std::enable_if_t<std::is_integral_v<Z>, bool> = true>
inline constexpr int fpclassify(Z x)
{
    return boost::math::ccmath::fpclassify(static_cast<double>(x));
}

}

#endif // BOOST_MATH_CCMATH_FPCLASSIFY

//  (C) Copyright Matt Borland 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_CCMATH_FMA_HPP
#define BOOST_MATH_CCMATH_FMA_HPP

#include <boost/math/ccmath/detail/config.hpp>

#ifdef BOOST_MATH_NO_CCMATH
#error "The header <boost/math/fma.hpp> can only be used in C++17 and later."
#endif

#include <boost/math/ccmath/isinf.hpp>
#include <boost/math/ccmath/isnan.hpp>

namespace boost::math::ccmath {

namespace detail {

template <typename T>
constexpr T fma_imp(const T x, const T y, const T z) noexcept
{
    #if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && !defined(__INTEL_LLVM_COMPILER)
    if constexpr (std::is_same_v<T, float>)
    {
        return __builtin_fmaf(x, y, z);
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        return __builtin_fma(x, y, z);
    }
    else if constexpr (std::is_same_v<T, long double>)
    {
        return __builtin_fmal(x, y, z);
    }
    #endif
    
    // If we can't use compiler intrinsics hope that -fma flag optimizes this call to fma instruction
    return (x * y) + z;
}

} // Namespace detail

template <typename Real, std::enable_if_t<!std::is_integral_v<Real>, bool> = true>
constexpr Real fma(Real x, Real y, Real z) noexcept
{
    if (BOOST_MATH_IS_CONSTANT_EVALUATED(x))
    {
        if (x == 0 && boost::math::ccmath::isinf(y))
        {
            return std::numeric_limits<Real>::quiet_NaN();
        }
        else if (y == 0 && boost::math::ccmath::isinf(x))
        {
            return std::numeric_limits<Real>::quiet_NaN();
        }
        else if (boost::math::ccmath::isnan(x))
        {
            return std::numeric_limits<Real>::quiet_NaN();
        }
        else if (boost::math::ccmath::isnan(y))
        {
            return std::numeric_limits<Real>::quiet_NaN();
        }
        else if (boost::math::ccmath::isnan(z))
        {
            return std::numeric_limits<Real>::quiet_NaN();
        }

        return boost::math::ccmath::detail::fma_imp(x, y, z);
    }
    else
    {
        using std::fma;
        return fma(x, y, z);
    }
}

template <typename T1, typename T2, typename T3>
constexpr auto fma(T1 x, T2 y, T3 z) noexcept
{
    if (BOOST_MATH_IS_CONSTANT_EVALUATED(x))
    {
        // If the type is an integer (e.g. epsilon == 0) then set the epsilon value to 1 so that type is at a minimum 
        // cast to double
        constexpr auto T1p = std::numeric_limits<T1>::epsilon() > 0 ? std::numeric_limits<T1>::epsilon() : 1;
        constexpr auto T2p = std::numeric_limits<T2>::epsilon() > 0 ? std::numeric_limits<T2>::epsilon() : 1;
        constexpr auto T3p = std::numeric_limits<T3>::epsilon() > 0 ? std::numeric_limits<T3>::epsilon() : 1;

        using promoted_type = 
                              #ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
                              std::conditional_t<T1p <= LDBL_EPSILON && T1p <= T2p, T1,
                              std::conditional_t<T2p <= LDBL_EPSILON && T2p <= T1p, T2,
                              std::conditional_t<T3p <= LDBL_EPSILON && T3p <= T2p, T3,
                              #endif
                              std::conditional_t<T1p <= DBL_EPSILON && T1p <= T2p, T1,
                              std::conditional_t<T2p <= DBL_EPSILON && T2p <= T1p, T2, 
                              std::conditional_t<T3p <= DBL_EPSILON && T3p <= T2p, T3, double
                              #ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
                              >>>>>>;
                              #else
                              >>>;
                              #endif

        return boost::math::ccmath::fma(promoted_type(x), promoted_type(y), promoted_type(z));
    }
    else
    {
        using std::fma;
        return fma(x, y, z);
    }
}

constexpr float fmaf(float x, float y, float z) noexcept
{
    return boost::math::ccmath::fma(x, y, z);
}

#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
constexpr long double fmal(long double x, long double y, long double z) noexcept
{
    return boost::math::ccmath::fma(x, y, z);
}
#endif

} // Namespace boost::math::ccmath

#endif // BOOST_MATH_CCMATH_FMA_HPP

//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cmath>
#include <cfloat>
#include <cstdint>
#include <limits>
#include <utility>
#include <type_traits>
#include <boost/math/ccmath/modf.hpp>
#include <boost/math/ccmath/isnan.hpp>
#include <boost/math/ccmath/isinf.hpp>

#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
#endif

template <typename T>
inline constexpr T floating_point_value(const T val)
{
    T i = 0;
    const T ans = boost::math::ccmath::modf(val, &i);

    return ans;
}

template <typename T>
inline constexpr T integral_value(const T val)
{
    T i = 0;
    boost::math::ccmath::modf(val, &i);

    return i;
}

template <typename T>
inline constexpr std::pair<T, T> pair_value(const T val)
{
    T i = 0;
    T ans = boost::math::ccmath::modf(val, &i);

    return std::make_pair(i, ans);
}

template <typename T>
constexpr void test()
{
    if constexpr (std::numeric_limits<T>::has_quiet_NaN)
    {
        constexpr std::pair<T, T> NaN_val = pair_value(std::numeric_limits<T>::quiet_NaN());
        static_assert(boost::math::ccmath::isnan(NaN_val.first));
        static_assert(boost::math::ccmath::isnan(NaN_val.second));
    }

    // if x is +-0, +-0 is returned and +-0 is stored in *iptr
    static_assert(floating_point_value(T(0)) == 0);
    static_assert(floating_point_value(-T(0)) == -0);
    static_assert(integral_value(T(0)) == 0);
    static_assert(integral_value(-T(0)) == -0);

    // if x is +- inf, +-0 is returned and +-inf is stored in *iptr
    static_assert(floating_point_value(std::numeric_limits<T>::infinity()) == 0);
    static_assert(floating_point_value(-std::numeric_limits<T>::infinity()) == -0);
    static_assert(integral_value(std::numeric_limits<T>::infinity()) == std::numeric_limits<T>::infinity());
    static_assert(integral_value(-std::numeric_limits<T>::infinity()) == -std::numeric_limits<T>::infinity());

    // The returned value is exact, the current rounding mode is ignored
    // The return value and *iptr each have the same type and sign as x
    static_assert(integral_value(T(123.45)) == 123);
    static_assert(integral_value(T(-234.56)) == -234);
    static_assert(floating_point_value(T(1.0/2)) == T(1.0/2));
    static_assert(floating_point_value(T(-1.0/3)) == T(-1.0/3));
}

#if !defined(BOOST_MATH_NO_CONSTEXPR_DETECTION) && !defined(BOOST_MATH_USING_BUILTIN_CONSTANT_P)
int main()
{
    test<float>();
    test<double>();

    #ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test<long double>();
    #endif
    
    #ifdef BOOST_HAS_FLOAT128
    test<boost::multiprecision::float128>();
    #endif

    return 0;
}
#else
int main()
{
    return 0;
}
#endif

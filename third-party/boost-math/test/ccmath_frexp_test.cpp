//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cmath>
#include <cfloat>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <boost/math/ccmath/frexp.hpp>
#include <boost/math/ccmath/isnan.hpp>
#include <boost/math/ccmath/isinf.hpp>

#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
#endif

template <typename T>
inline constexpr T base_helper(const T val)
{
    int i = 0;
    const T ans = boost::math::ccmath::frexp(val, &i);

    return ans;
}

template <typename T>
inline constexpr int exp_helper(const T val)
{
    int i = 0;
    boost::math::ccmath::frexp(val, &i);

    return i;
}

template <typename T>
constexpr void test()
{
    if constexpr (std::numeric_limits<T>::has_quiet_NaN)
    {
        static_assert(boost::math::ccmath::isnan(base_helper(std::numeric_limits<T>::quiet_NaN())), "If the arg is NaN, NaN is returned");
    }

    static_assert(!base_helper(T(0)), "If the arg is +- 0 the value is returned");
    static_assert(!base_helper(T(-0)), "If the arg is +- 0 the value is returned");
    static_assert(boost::math::ccmath::isinf(base_helper(std::numeric_limits<T>::infinity())), "If the arg is +- inf the value is returned");
    static_assert(boost::math::ccmath::isinf(base_helper(-std::numeric_limits<T>::infinity())), "If the arg is +- inf the value is returned");

    // N[125/32, 30]
    // 3.90625000000000000000000000000
    // 0.976562500000000000000000000000 * 2^2
    constexpr T test_base = base_helper(T(125.0/32));
    static_assert(test_base == T(0.9765625));
    constexpr int test_exp = exp_helper(T(125.0/32));
    static_assert(test_exp == 2);
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

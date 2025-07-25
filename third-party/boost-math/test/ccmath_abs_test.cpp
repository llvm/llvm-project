//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <boost/math/ccmath/abs.hpp>
#include <boost/math/ccmath/fabs.hpp>
#include <boost/math/tools/config.hpp>

#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
#endif

template <typename T>
void test()
{
    static_assert(boost::math::ccmath::abs(T(3)) == 3);
    static_assert(boost::math::ccmath::abs(T(-3)) == 3);
    static_assert(boost::math::ccmath::abs(T(-0)) == 0);
    static_assert(boost::math::ccmath::abs(-std::numeric_limits<T>::infinity()) == std::numeric_limits<T>::infinity());

    if constexpr (std::numeric_limits<T>::has_quiet_NaN)
    {
        static_assert(boost::math::ccmath::abs(-std::numeric_limits<T>::quiet_NaN()) != std::numeric_limits<T>::quiet_NaN());
    }
}

template <typename T>
void gpp_test()
{
    static_assert(std::sin(T(0)) == 0);
    
    constexpr T sin_1 = boost::math::ccmath::abs(std::sin(T(-1)));
    static_assert(sin_1 > 0);
    static_assert(sin_1 == T(0.841470984807896506652502321630298999622563060798371065672751709L));
}

template <typename T>
void fabs_test()
{
    static_assert(boost::math::ccmath::fabs(T(3)) == 3);
    static_assert(boost::math::ccmath::fabs(T(-3)) == 3);
    static_assert(boost::math::ccmath::fabs(T(-0)) == 0);
    static_assert(boost::math::ccmath::fabs(-std::numeric_limits<T>::infinity()) == std::numeric_limits<T>::infinity());

    if constexpr (std::numeric_limits<T>::has_quiet_NaN)
    {
        static_assert(boost::math::ccmath::fabs(-std::numeric_limits<T>::quiet_NaN()) != std::numeric_limits<T>::quiet_NaN());
    }
}

// Only test on platforms that provide BOOST_MATH_IS_CONSTANT_EVALUATED
#ifndef BOOST_MATH_NO_CONSTEXPR_DETECTION
int main()
{
    test<float>();
    test<double>();
    
    #ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test<long double>();
    #endif

    #if defined(BOOST_MATH_TEST_FLOAT128) && !defined(BOOST_MATH_USING_BUILTIN_CONSTANT_P)
    test<boost::multiprecision::float128>();
    #endif

    test<int>();
    test<long>();
    test<long long>();
    test<std::int32_t>();
    test<std::int64_t>();

    // Types that are convertible to int
    test<short>();
    test<signed char>();

    // fabs
    fabs_test<float>();
    fabs_test<double>();

    #ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    fabs_test<long double>();
    #endif

    #if defined(BOOST_MATH_TEST_FLOAT128) && !defined(BOOST_MATH_USING_BUILTIN_CONSTANT_P) && defined(BOOST_MATH_TEST_FLOAT128)
    fabs_test<boost::multiprecision::float128>();
    #endif

    // Tests using glibcxx extensions that allow for some constexpr cmath
    #if __GNUC__ >= 10
    gpp_test<float>();
    gpp_test<double>();

    #ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    gpp_test<long double>();
    #endif
    
    #endif // glibcxx tests

    return 0;
}
#else
int main()
{
    return 0;
}
#endif

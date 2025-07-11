//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cmath>
#include <cfloat>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <boost/math/ccmath/logb.hpp>
#include <boost/math/ccmath/isnan.hpp>
#include <boost/math/ccmath/isinf.hpp>

#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
#endif

template <typename T>
constexpr void test()
{
    if constexpr (std::numeric_limits<T>::has_quiet_NaN)
    {
        static_assert(boost::math::ccmath::isnan(boost::math::ccmath::logb(std::numeric_limits<T>::quiet_NaN())), "if arg is NaN, NaN is returned");
    }

    static_assert(boost::math::ccmath::isinf(boost::math::ccmath::logb(T(0))), "If arg is +-0, -infinity is returned");
    static_assert(boost::math::ccmath::isinf(boost::math::ccmath::logb(T(-0))), "If arg is +-0, -infinity is returned");
    static_assert(boost::math::ccmath::isinf(boost::math::ccmath::logb(std::numeric_limits<T>::infinity())),
                  "if arg is +-infinity, infinity is returned");
    static_assert(boost::math::ccmath::isinf(boost::math::ccmath::logb(-std::numeric_limits<T>::infinity())),
                  "if arg is +-infinity, infinity is returned");

    // 123.45 = 1.92891 * 2^6
    constexpr T test_exp = boost::math::ccmath::logb(T(123.45));
    static_assert(test_exp == T(6));
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

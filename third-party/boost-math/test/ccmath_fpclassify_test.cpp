//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cmath>
#include <cfloat>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <boost/math/ccmath/fpclassify.hpp>

#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
#endif

template <typename T>
void test()
{
    if constexpr (std::numeric_limits<T>::has_quiet_NaN)
    {
        static_assert(boost::math::ccmath::fpclassify(std::numeric_limits<T>::quiet_NaN()) == FP_NAN);
    }

    static_assert(boost::math::ccmath::fpclassify(T(0)) == FP_ZERO);
    static_assert(boost::math::ccmath::fpclassify(std::numeric_limits<T>::infinity()) == FP_INFINITE);
    static_assert(boost::math::ccmath::fpclassify((std::numeric_limits<T>::min)() / T(2)) == FP_SUBNORMAL);
    static_assert(boost::math::ccmath::fpclassify(T(1)) == FP_NORMAL);
}

#ifndef BOOST_MATH_NO_CONSTEXPR_DETECTION
int main()
{
    test<float>();
    test<double>();

    #ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test<long double>();
    #endif
    
    #if defined(BOOST_MATH_TEST_FLOAT128) && !defined(BOOST_MATH_USING_BUILTIN_CONSTANT_P) && defined(BOOST_MATH_TEST_FLOAT128)
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

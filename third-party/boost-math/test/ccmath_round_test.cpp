//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cmath>
#include <cfloat>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <type_traits>
#include <boost/math/ccmath/round.hpp>
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
        static_assert(boost::math::ccmath::isnan(boost::math::ccmath::round(std::numeric_limits<T>::quiet_NaN())), "If x is NaN, NaN is returned");
        static_assert(boost::math::ccmath::lround(std::numeric_limits<T>::quiet_NaN()) == T(0), "If x is NaN, 0 is returned");
        static_assert(boost::math::ccmath::llround(std::numeric_limits<T>::quiet_NaN()) == T(0), "If x is NaN, 0 is returned");
    }

    static_assert(boost::math::ccmath::round(T(0)) == T(0));
    static_assert(boost::math::ccmath::lround(T(0)) == 0l);
    static_assert(boost::math::ccmath::llround(T(0)) == 0ll);

    static_assert(boost::math::ccmath::round(T(-0)) == T(-0));
    static_assert(boost::math::ccmath::lround(T(-0)) == -0l);
    static_assert(boost::math::ccmath::llround(T(-0)) == -0ll);

    static_assert(boost::math::ccmath::isinf(boost::math::ccmath::round(std::numeric_limits<T>::infinity())));
    static_assert(boost::math::ccmath::lround(std::numeric_limits<T>::infinity()) == 0l);
    static_assert(boost::math::ccmath::llround(std::numeric_limits<T>::infinity()) == 0ll);

    static_assert(boost::math::ccmath::round(T(2.3)) == T(2));
    static_assert(boost::math::ccmath::round(T(2.5)) == T(3));
    static_assert(boost::math::ccmath::round(T(2.7)) == T(3));
    static_assert(boost::math::ccmath::round(T(-2.3)) == T(-2));
    static_assert(boost::math::ccmath::round(T(-2.5)) == T(-3));
    static_assert(boost::math::ccmath::round(T(-2.7)) == T(-3));

    static_assert(boost::math::ccmath::lround(T(2.3)) == 2l);
    static_assert(boost::math::ccmath::lround(T(2.5)) == 3l);
    static_assert(boost::math::ccmath::lround(T(2.7)) == 3l);
    static_assert(boost::math::ccmath::lround(T(-2.3)) == -2l);
    static_assert(boost::math::ccmath::lround(T(-2.5)) == -3l);
    static_assert(boost::math::ccmath::lround(T(-2.7)) == -3l);

    static_assert(boost::math::ccmath::llround(T(2.3)) == 2ll);
    static_assert(boost::math::ccmath::llround(T(2.5)) == 3ll);
    static_assert(boost::math::ccmath::llround(T(2.7)) == 3ll);
    static_assert(boost::math::ccmath::llround(T(-2.3)) == -2ll);
    static_assert(boost::math::ccmath::llround(T(-2.5)) == -3ll);
    static_assert(boost::math::ccmath::llround(T(-2.7)) == -3ll);
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

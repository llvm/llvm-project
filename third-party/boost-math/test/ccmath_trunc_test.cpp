//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cmath>
#include <cfloat>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <boost/math/ccmath/trunc.hpp>
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
        static_assert(boost::math::ccmath::isnan(boost::math::ccmath::trunc(std::numeric_limits<T>::quiet_NaN())), "If x is NaN, NaN is returned");
    }

    static_assert(!boost::math::ccmath::trunc(T(0)), "If x is +- 0, it is returned, unmodified");
    static_assert(!boost::math::ccmath::trunc(T(-0)), "If x is +- 0, it is returned, unmodified");
    
    static_assert(boost::math::ccmath::isinf(boost::math::ccmath::trunc(std::numeric_limits<T>::infinity())), 
                  "If x is +- inf, it is returned, unmodified");
    static_assert(boost::math::ccmath::isinf(boost::math::ccmath::trunc(-std::numeric_limits<T>::infinity())), 
                  "If x is +- inf, it is returned, unmodified");

    static_assert(boost::math::ccmath::trunc(T(2)) == T(2));
    static_assert(boost::math::ccmath::trunc(T(2.4)) == T(2));
    static_assert(boost::math::ccmath::trunc(T(2.9)) == T(2));
    static_assert(boost::math::ccmath::trunc(T(-2.7)) == T(-2));
    static_assert(boost::math::ccmath::trunc(T(-2)) == T(-2));
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

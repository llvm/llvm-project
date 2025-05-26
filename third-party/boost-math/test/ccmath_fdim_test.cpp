//  (C) Copyright Matt Borland 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cmath>
#include <cfloat>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <boost/math/ccmath/fdim.hpp>
#include <boost/math/ccmath/isnan.hpp>
#include <boost/math/ccmath/isinf.hpp>

#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
#endif

template <typename T>
constexpr void test()
{
    static_assert(boost::math::ccmath::isnan(boost::math::ccmath::fdim(std::numeric_limits<T>::quiet_NaN(), T(1))), "If x is NaN, NaN is returned");
    static_assert(boost::math::ccmath::isnan(boost::math::ccmath::fdim(T(1), std::numeric_limits<T>::quiet_NaN())), "If y is NaN, NaN is returned");

    static_assert(boost::math::ccmath::fdim(T(4), T(1)) == T(3));
    static_assert(boost::math::ccmath::fdim(T(1), T(4)) == T(0));
    static_assert(boost::math::ccmath::fdim(T(4), T(-1)) == T(5));
    static_assert(boost::math::ccmath::fdim(T(1), T(-4)) == T(5));

    static_assert(boost::math::ccmath::isinf(boost::math::ccmath::fdim(std::numeric_limits<T>::infinity(), T(-1))));
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

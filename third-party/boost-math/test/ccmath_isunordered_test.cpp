//  (C) Copyright Matt Borland 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cmath>
#include <cfloat>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <boost/math/ccmath/isunordered.hpp>

#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
#endif

template <typename T>
constexpr void test()
{
    static_assert(boost::math::ccmath::isunordered(std::numeric_limits<T>::quiet_NaN(), T(0)));
    static_assert(boost::math::ccmath::isunordered(T(0), std::numeric_limits<T>::quiet_NaN()));
    static_assert(boost::math::ccmath::isunordered(std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN()));

    static_assert(!boost::math::ccmath::isunordered(T(0), std::numeric_limits<T>::infinity()));
    static_assert(!boost::math::ccmath::isunordered(std::numeric_limits<T>::infinity(), T(0)));
    static_assert(!boost::math::ccmath::isunordered(T(1), T(2)));
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

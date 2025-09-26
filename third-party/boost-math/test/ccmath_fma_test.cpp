//  (C) Copyright Matt Borland 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cmath>
#include <cfloat>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <boost/math/ccmath/fma.hpp>
#include <boost/math/ccmath/isnan.hpp>
#include <boost/math/ccmath/isinf.hpp>
#include <boost/math/ccmath/abs.hpp>

#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
#endif

#if !defined(BOOST_MATH_NO_CONSTEXPR_DETECTION) && !defined(BOOST_MATH_USING_BUILTIN_CONSTANT_P)
template <typename T>
constexpr void test()
{
    // Error handling
    static_assert(boost::math::ccmath::isnan(boost::math::ccmath::fma(std::numeric_limits<T>::infinity(), T(0), T(1))));
    static_assert(boost::math::ccmath::isnan(boost::math::ccmath::fma(T(0), std::numeric_limits<T>::infinity(), T(1))));

    static_assert(boost::math::ccmath::isnan(boost::math::ccmath::fma(std::numeric_limits<T>::infinity(), T(0), std::numeric_limits<T>::quiet_NaN())));
    static_assert(boost::math::ccmath::isnan(boost::math::ccmath::fma(T(0), std::numeric_limits<T>::infinity(), std::numeric_limits<T>::quiet_NaN())));

    static_assert(boost::math::ccmath::isnan(boost::math::ccmath::fma(std::numeric_limits<T>::quiet_NaN(), T(1), T(1))));
    static_assert(boost::math::ccmath::isnan(boost::math::ccmath::fma(T(1), std::numeric_limits<T>::quiet_NaN(), T(1))));

    static_assert(boost::math::ccmath::isnan(boost::math::ccmath::fma(T(1), T(1), std::numeric_limits<T>::quiet_NaN())));

    // Functionality
    static_assert(boost::math::ccmath::fma(T(1), T(2), T(3)) == T(5));
    static_assert(boost::math::ccmath::fma(T(2), T(3), T(1)) == T(7));

    // Correct promoted types
    if constexpr (!std::is_same_v<T, float>)
    {
        constexpr auto test_type = boost::math::ccmath::fma(T(1), 1.0, 1.0f);
        static_assert(std::is_same_v<T, std::remove_cv_t<decltype(test_type)>>);
    }
    else
    {
        constexpr auto test_type = boost::math::ccmath::fma(1.0f, 1, 1.0);
        static_assert(std::is_same_v<double, std::remove_cv_t<decltype(test_type)>>);
    }
}

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

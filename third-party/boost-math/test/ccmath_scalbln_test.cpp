//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cmath>
#include <cfloat>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <boost/math/ccmath/scalbln.hpp>
#include <boost/math/ccmath/isnan.hpp>
#include <boost/math/ccmath/isinf.hpp>

#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
#endif

template <typename T>
constexpr void test()
{
    // Tests are only valid on binary systems
    if constexpr (FLT_RADIX == 2)
    {
        if constexpr (std::numeric_limits<T>::has_quiet_NaN)
        {
            static_assert(boost::math::ccmath::isnan(boost::math::ccmath::scalbln(std::numeric_limits<T>::quiet_NaN(), 1l)), "If x is NaN, NaN is returned");
        }

        static_assert(!boost::math::ccmath::scalbln(T(0), 1l), "If x is +- 0, it is returned, unmodified");
        static_assert(!boost::math::ccmath::scalbln(T(-0), 1l), "If x is +- 0, it is returned, unmodified");

        static_assert(boost::math::ccmath::isinf(boost::math::ccmath::scalbln(std::numeric_limits<T>::infinity(), 1l)), 
                        "If x is +- inf, it is returned, unmodified");
        static_assert(boost::math::ccmath::isinf(boost::math::ccmath::scalbln(-std::numeric_limits<T>::infinity(), 1l)), 
                        "If x is +- inf, it is returned, unmodified");

        static_assert(boost::math::ccmath::scalbln(T(2), 0l) == T(2), "If exp is 0, then x is returned, unmodified");

        // 1 * 2^2 = 4
        static_assert(boost::math::ccmath::scalbln(T(1), 2l) == T(4));

        // 1.2 * 2^10 = 1228.8
        static_assert(boost::math::ccmath::scalbln(T(1.2), 10l) == T(1228.8));

        // 500 * 2^-2 = 125
        static_assert(boost::math::ccmath::scalbln(T(500), -2l) == T(125));
    }
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

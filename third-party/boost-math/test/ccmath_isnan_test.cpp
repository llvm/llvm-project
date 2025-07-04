//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cmath>
#include <cfloat>
#include <cstdint>
#include <limits>
#include <boost/math/ccmath/isnan.hpp>
#include <boost/core/lightweight_test.hpp>
#include <boost/math/tools/config.hpp>

#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
#endif

template <typename T>
void test()
{
    constexpr bool test_val = boost::math::ccmath::isnan(T(0));
    static_assert(!test_val, "Not constexpr");

    if constexpr (std::numeric_limits<T>::has_quiet_NaN)
    {
        static_assert(boost::math::ccmath::isnan(std::numeric_limits<T>::quiet_NaN()), "Quiet NAN failed");
    }
    if constexpr (std::numeric_limits<T>::has_signaling_NaN)
    {
        static_assert(boost::math::ccmath::isnan(std::numeric_limits<T>::signaling_NaN()), "Signaling NAN failed");
    }
    static_assert(!boost::math::ccmath::isnan(std::numeric_limits<T>::infinity()), "Infinity failed");
    static_assert(!boost::math::ccmath::isnan(T(0)), "Real 0 failed");
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
    test<unsigned>();
    test<long>();
    test<std::int32_t>();
    test<std::int64_t>();
    test<std::uint32_t>();
    
    return boost::report_errors();
}
#else
int main()
{
    return 0;
}
#endif

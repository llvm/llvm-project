//  (C) Copyright John Maddock 2008 - 2022.
//  (C) Copyright Matt Borland 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <iomanip>
#include <limits>
#include <boost/math/tools/precision.hpp>
#include <boost/math/special_functions/next.hpp>
#include <boost/math/ccmath/next.hpp>
#include <boost/math/ccmath/fpclassify.hpp>
#include "math_unit_test.hpp"

#if !defined(BOOST_MATH_NO_CONSTEXPR_DETECTION) && !defined(BOOST_MATH_USING_BUILTIN_CONSTANT_P)
template <typename T>
void test_next()
{
    // NaN handling
    static_assert(boost::math::ccmath::isnan(boost::math::ccmath::nextafter(std::numeric_limits<T>::quiet_NaN(), T(0))));
    static_assert(boost::math::ccmath::isnan(boost::math::ccmath::nextafter(T(0), std::numeric_limits<T>::quiet_NaN())));

    // Handling of 0
    static_assert(boost::math::ccmath::nextafter(T(-0.0), T(0.0)) == T(0.0));
    static_assert(boost::math::ccmath::nextafter(T(0.0), T(-0.0)) == T(-0.0));

    // val = 1
    constexpr T test_1 = boost::math::ccmath::nextafter(T(1), T(1.5));
    static_assert(test_1 < 1 + 2*std::numeric_limits<T>::epsilon());
    static_assert(test_1 > 1 - 2*std::numeric_limits<T>::epsilon());

    constexpr T test_1_toward = boost::math::ccmath::nexttoward(T(1), T(1.5));
    
    // For T is long double nextafter is the same as nexttoward
    // For T is not long double the answer will be either greater or equal when from > to depending on loss of precision
    static_assert(test_1 >= test_1_toward);

    // Compare to existing implementation
    // test_1 has already passed through static_asserts so we know it was calculated at compile time
    // rather than farming out to std at run time.
    const T existing_test_1 = boost::math::nextafter(T(1), T(1.5));
    CHECK_EQUAL(test_1, existing_test_1);
}

int main(void)
{
    test_next<float>();
    test_next<double>();

    #ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test_next<long double>();
    #endif

    return boost::math::test::report_errors();
}
#else
int main(void)
{
    return 0;
}
#endif

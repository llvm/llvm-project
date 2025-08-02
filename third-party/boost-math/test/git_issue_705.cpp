// Copyright Matt Borland, 2022
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error

#include "math_unit_test.hpp"
#include <cmath>
#include <boost/math/special_functions/powm1.hpp>

template <typename T>
void test()
{
    CHECK_EQUAL(std::pow(T(0), T(2)) - 1, boost::math::powm1(T(0), T(2)));
    CHECK_EQUAL(std::pow(T(0), T(-2)) - 1, boost::math::powm1(T(0), T(-2)));
    CHECK_EQUAL(std::pow(T(0), T(0.1)) - 1, boost::math::powm1(T(0), T(0.1)));
    CHECK_EQUAL(std::pow(T(0), T(-0.1)) - 1, boost::math::powm1(T(0), T(-0.1)));
}

int main()
{
    test<float>();
    test<double>();
    
    #ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test<long double>();
    #endif

    return boost::math::test::report_errors();
}

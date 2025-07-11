// Copyright Matt Borland, 2022
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "math_unit_test.hpp"
#include <iostream>
#include <boost/math/special_functions/hypergeometric_pFq.hpp>

template <typename T>
void test()
{
    T z = -9;
    T h = boost::math::hypergeometric_pFq({2,3,4}, {5,6,7,8}, z);  // Calculate 3F4

    // https://www.wolframalpha.com/input?i=HypergeometricPFQ%5B%7B2%2C3%2C4%7D%2C+%7B5%2C6%2C7%2C8%7D%2C-9%5D
    CHECK_ULP_CLOSE(h, static_cast<T>(0.8821347263567429637736237739975599147079177547846902085909266074L), 2);
}

int main(void)
{
    test<double>();
    test<long double>();

    return boost::math::test::report_errors();
}

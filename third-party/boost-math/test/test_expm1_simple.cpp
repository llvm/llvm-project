//  Copyright Matt Borland 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <random>
#include <cmath>
#include <boost/math/special_functions/expm1.hpp>
#include "math_unit_test.hpp"

constexpr int N = 50000;

template <typename T>
void test()
{
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<T> dist(0, 0.01);

    for (int n = 0; n < N; ++n)
    {
        const T value (dist(rng));
        CHECK_ULP_CLOSE(std::expm1(value), boost::math::expm1(value), 10);
    }
}

int main()
{
    test<float>();
    test<double>();

    return boost::math::test::report_errors();
}

//  Copyright Matt Borland 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef SYCL_LANGUAGE_VERSION
#include "sycl/sycl.hpp"
#endif

#include <random>
#include <cmath>
#include <boost/math/special_functions/log1p.hpp>
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
        CHECK_ULP_CLOSE(std::log1p(value), boost::math::log1p(value), 10);
    }
}

template <typename T>
void test_log1pmx()
{
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<T> dist(0, 0.01);

    for (int n = 0; n < N; ++n)
    {
        const T value (dist(rng));
        CHECK_ULP_CLOSE(std::log1p(value) - value, boost::math::log1pmx(value), 1e9);
    }

    std::uniform_real_distribution<T> dist2(0, 10);

    for (int n = 0; n < 100; ++n)
    {
        const T value (dist2(rng));
        CHECK_ULP_CLOSE(std::log1p(value) - value, boost::math::log1pmx(value), 1e9);
    }
#ifndef BOOST_MATH_NO_EXCEPTIONS
    CHECK_THROW(boost::math::log1pmx(T(-1.1)), std::domain_error);
    CHECK_THROW(boost::math::log1pmx(T(-1.0)), std::overflow_error);
#endif
}

int main()
{
    test<float>();
    test<double>();

    test_log1pmx<float>();
    test_log1pmx<double>();

    return boost::math::test::report_errors();
}

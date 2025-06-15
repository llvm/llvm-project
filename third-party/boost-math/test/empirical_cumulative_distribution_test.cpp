/*
 * Copyright Nick Thompson, 2019
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "math_unit_test.hpp"
#include <numeric>
#include <utility>
#include <random>
#include <boost/core/demangle.hpp>
#include <boost/math/distributions/empirical_cumulative_distribution_function.hpp>
#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
using boost::multiprecision::float128;
#endif

using boost::math::empirical_cumulative_distribution_function;

template<class Z>
void test_uniform_z()
{
    std::vector<Z> v{6,3,4,1,2,5};

    auto ecdf = empirical_cumulative_distribution_function(std::move(v));

    CHECK_ULP_CLOSE(1.0/6.0, ecdf(1), 1);
    CHECK_ULP_CLOSE(2.0/6.0, ecdf(2), 1);
    CHECK_ULP_CLOSE(3.0/6.0, ecdf(3), 1);
    CHECK_ULP_CLOSE(4.0/6.0, ecdf(4), 1);
    CHECK_ULP_CLOSE(5.0/6.0, ecdf(5), 1);
    CHECK_ULP_CLOSE(6.0/6.0, ecdf(6), 1);

    // Less trivial:

    v = {6,3,4,1,1,1,2,4};
    ecdf = empirical_cumulative_distribution_function(std::move(v));
    CHECK_ULP_CLOSE(3.0/8.0, ecdf(1), 1);
    CHECK_ULP_CLOSE(4.0/8.0, ecdf(2), 1);
    CHECK_ULP_CLOSE(5.0/8.0, ecdf(3), 1);
    CHECK_ULP_CLOSE(7.0/8.0, ecdf(4), 1);
    CHECK_ULP_CLOSE(7.0/8.0, ecdf(5), 1);
    CHECK_ULP_CLOSE(8.0/8.0, ecdf(6), 1);
}

template<class Real>
void test_uniform()
{
    size_t n = 128;
    std::vector<Real> v(n);
    for (size_t i = 0; i < n; ++i) {
      v[i] = Real(i+1)/Real(n);
    }

    auto ecdf = empirical_cumulative_distribution_function(std::move(v));

    for (size_t i = 0; i < n; ++i) {
      CHECK_ULP_CLOSE(Real(i+1)/Real(n), ecdf(Real(i+1)/Real(n)), 1);
    }
}


int main()
{
    test_uniform_z<int>();
    test_uniform<double>();
    return boost::math::test::report_errors();
}

// Copyright Matt Borland, 2022
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0. (See accompanying file
// LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// See: https://github.com/boostorg/math/issues/889
// See: https://github.com/scipy/scipy/issues/17388

#include <array>
#include <numeric>
#include <boost/math/distributions/binomial.hpp>
#include "math_unit_test.hpp"

int main(void)
{
    // Test case 1: Goes against scipy convention but is correct
    const auto binom_dist_1 {boost::math::binomial_distribution<double>(4, 0.5)};
    // https://www.wolframalpha.com/input?i=quantile%28binomial+distribution%284%2C0.5%29%2C0%29
    CHECK_ULP_CLOSE(boost::math::quantile(binom_dist_1, 0.0), 0.0, 1);
    
    // Test case 2 from linked issue
    const auto binom_dist_2 {boost::math::binomial_distribution<double>(4, 1)};
    // https://www.wolframalpha.com/input?i=quantile%28binomial+distribution%284%2C1%29%2C0.3%29
    CHECK_ULP_CLOSE(4.0, boost::math::quantile(binom_dist_2, 0.3), 1);

    return boost::math::test::report_errors();
}

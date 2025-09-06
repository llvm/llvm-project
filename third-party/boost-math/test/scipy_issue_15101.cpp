// Copyright Matt Borland, 2023
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0. (See accompanying file
// LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_MATH_DOMAIN_ERROR_POLICY ignore_error
#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error
#define BOOST_MATH_PROMOTE_DOUBLE_POLICY false

#include <cfenv>
#include <array>
#include <numeric>
#include <boost/math/distributions/binomial.hpp>
#include "math_unit_test.hpp"

#pragma STDC FENV_ACCESS ON

int main()
{
    constexpr double n {2000};
    constexpr double p {0.9999};
    const auto binom_dist = boost::math::binomial_distribution<double>(n, p);
    const auto pdf1 = boost::math::pdf(binom_dist, 3);

    CHECK_ULP_CLOSE(pdf1, 0.0, 1);

    if (std::fetestexcept(FE_DIVBYZERO)) 
    {
        return 1;
    }

    return boost::math::test::report_errors();
}

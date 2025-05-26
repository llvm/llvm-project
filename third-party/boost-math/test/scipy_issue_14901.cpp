// Copyright Matt Borland, 2022
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0. (See accompanying file
// LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// See: https://github.com/scipy/scipy/issues/14901

#include <cfenv>
#include <iostream>
#include <boost/math/distributions/non_central_t.hpp>
#include "math_unit_test.hpp"

#pragma STDC FENV_ACCESS ON

int main() 
{
    auto nct1 = boost::math::non_central_t(2, 2);
    const auto cdf1 = boost::math::cdf(nct1, 0.05);

    CHECK_ULP_CLOSE(cdf1, 0.02528206132724582, 20);

    if (std::fetestexcept(FE_INVALID) || std::fetestexcept(FE_DIVBYZERO)) 
    {
        return 1;
    }

    auto nct2 = boost::math::non_central_t(1, 3);
    const auto cdf2 = boost::math::cdf(nct2, 0.05);

    CHECK_ULP_CLOSE(cdf2, 0.00154456589169420, 20);

    if (std::fetestexcept(FE_INVALID) || std::fetestexcept(FE_DIVBYZERO))
    {
        return 2;
    }

    const auto pdf1 = boost::math::pdf(nct1, 0.05);
    CHECK_ULP_CLOSE(pdf1, 0.05352074159921797, 20);

    if (std::fetestexcept(FE_INVALID) || std::fetestexcept(FE_DIVBYZERO))
    {
        return 3;
    }

    const auto quantile1 = boost::math::quantile(nct1, 0.5);
    CHECK_ULP_CLOSE(quantile1, 2.33039025947198741, 20);

    if (std::fetestexcept(FE_INVALID) || std::fetestexcept(FE_DIVBYZERO))
    {
        return 4;
    }

    return boost::math::test::report_errors();
}

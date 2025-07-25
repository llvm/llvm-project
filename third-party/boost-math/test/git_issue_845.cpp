// Copyright Matt Borland, 2022
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0. (See accompanying file
// LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cfenv>
#include <iostream>
#include <boost/math/distributions/non_central_f.hpp>
#include "math_unit_test.hpp"

#pragma STDC FENV_ACCESS ON

int main() 
{
    auto ncf1 = boost::math::non_central_f(2, 2, 2);
    const auto cdf1 = boost::math::cdf(ncf1, 0.05);

    // https://www.wolframalpha.com/input?i=N%5BCdf%28NonCentralFRatioDistribution%282%2C2%2C2%29%2C+0.05%29%2C+15%5D
    CHECK_ULP_CLOSE(cdf1, 0.0183724431823392, 15);

    if (std::fetestexcept(FE_INVALID) || std::fetestexcept(FE_DIVBYZERO)) 
    {
        return 1;
    }

    auto ncf2 = boost::math::non_central_f(1, 3, 2);
    const auto cdf2 = boost::math::cdf(ncf2, 0.05);

    // https://www.wolframalpha.com/input?i=N%5BCdf%28NonCentralFRatioDistribution%281%2C3%2C2%29%2C+0.05%29%2C+15%5D
    CHECK_ULP_CLOSE(cdf2, 0.0611253404710995, 15);

    if (std::fetestexcept(FE_INVALID) || std::fetestexcept(FE_DIVBYZERO))
    {
        return 2;
    }

    auto ncf3 = boost::math::non_central_f(1, 1, 2);
    const auto cdf3 = boost::math::cdf(ncf3, 0.05);

    // https://www.wolframalpha.com/input?i=N%5BCdf%28NonCentralFRatioDistribution%281%2C1%2C2%29%2C+0.05%29%2C+15%5D
    CHECK_ULP_CLOSE(cdf3, 0.0531991288870987, 15);

    if (std::fetestexcept(FE_INVALID) || std::fetestexcept(FE_DIVBYZERO))
    {
        return 3;
    }

    const auto pdf3 = boost::math::pdf(ncf3, 0.05);

    // https://www.wolframalpha.com/input?i=N%5BPdf%28NonCentralFRatioDistribution%281%2C1%2C2%29%2C+0.05%29%2C+15%5D
    CHECK_ULP_CLOSE(pdf3, 0.547785080683644, 15);

    if (std::fetestexcept(FE_INVALID) || std::fetestexcept(FE_DIVBYZERO))
    {
        return 4;
    }

    return boost::math::test::report_errors();
}

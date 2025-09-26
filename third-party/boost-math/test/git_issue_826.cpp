// Copyright Nicholas McKibben, 2022
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0. (See accompanying file
// LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cfenv>
#include <iostream>
#include <boost/math/distributions/beta.hpp>
#include "math_unit_test.hpp"

#pragma STDC FENV_ACCESS ON

int main() {
    constexpr double q {0.999995};
    constexpr double a {2};
    constexpr double b {99999};
    std::feclearexcept(FE_ALL_EXCEPT);
    boost::math::beta_distribution<double> d {a, b};
    const auto ans = boost::math::quantile(d, q);
    if(std::fetestexcept(FE_OVERFLOW)) {
        std::cout << "overflow reported" << std::endl;
    } else {
        std::cout << "overflow not reported" << std::endl;
    }
    std::cout << std::setprecision(16) << "Ans is: " << ans << std::endl;
    return 0;

    CHECK_ULP_CLOSE(ans, 0.000149761910560502, 1);
}
// overflow reported
// Ans is: 0.000149761910560502

// Copyright Matt Borland, 2023
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0. (See accompanying file
// LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// See: https://github.com/scipy/scipy/issues/17916

#include <boost/math/distributions/non_central_chi_squared.hpp>
#include "math_unit_test.hpp"

int main(void)
{
    auto dist = boost::math::non_central_chi_squared(2.0, 4820232647677555.0);
    double test_pdf;
    double test_cdf;

    try
    {
        test_pdf = boost::math::pdf(dist, 2.0);
        test_cdf = boost::math::cdf(dist, 2.0);
    }
    catch (...)
    {
        return 1;
    }

    CHECK_ULP_CLOSE(test_pdf, 0.0, 1);
    CHECK_ULP_CLOSE(test_cdf, 0.0, 1);

    return boost::math::test::report_errors();
}

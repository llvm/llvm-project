// Copyright Matt Borland, 2023
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0. (See accompanying file
// LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// See: https://github.com/scipy/scipy/issues/17916

#include <boost/math/distributions/beta.hpp>
#include <boost/math/policies/policy.hpp>
#include <limits>
#include "math_unit_test.hpp"

int main(void)
{
    using my_policy = boost::math::policies::policy<boost::math::policies::promote_double<false>>;
    
    // test_beta_dist uses 100 eps as the tolerance
    constexpr double tol = 50;

    auto dist = boost::math::beta_distribution<double, my_policy>(1, 5);

    // https://www.wolframalpha.com/input?i=PDF%28beta+distribution%281%2C+5%29%2C+0%29
    double test_pdf_spot = boost::math::pdf(dist, 0);
    CHECK_ULP_CLOSE(test_pdf_spot, 5.0, tol);

    // https://www.wolframalpha.com/input?i=PDF%28beta+distribution%281%2C+5%29%2C+1e-30%29
    test_pdf_spot = boost::math::pdf(dist, 1e-30);
    CHECK_ULP_CLOSE(test_pdf_spot, 5.0, tol);

    // Appox equal to 5
    test_pdf_spot = boost::math::pdf(dist, 1e-310);
    CHECK_ULP_CLOSE(test_pdf_spot, 5.0, tol);

    test_pdf_spot = boost::math::pdf(dist, 1);
    CHECK_ULP_CLOSE(test_pdf_spot, 0.0, tol);

    return boost::math::test::report_errors();
}

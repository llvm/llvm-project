// Copyright Matt Borland, 2022
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0. (See accompanying file
// LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <array>
#include <numeric>
#include <boost/math/distributions/binomial.hpp>
#include "math_unit_test.hpp"

int main()
{
    constexpr double n {1541096362225563.0};
    constexpr double p {1.0477878413173978e-18};
    const auto binom_dist = boost::math::binomial_distribution<double>(n, p);
    std::array<double, 3> vals {};

    for (size_t i = 0; i < 2; ++i)
    {
        vals[i] = boost::math::pdf(binom_dist, i);
    }

    CHECK_ULP_CLOSE(vals[0], 0.9983865609638467, 10);

    CHECK_LE(std::accumulate(vals.begin(), vals.end(), 0.0), 1.0);

    return boost::math::test::report_errors();
}

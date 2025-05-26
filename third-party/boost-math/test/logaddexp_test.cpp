//  (C) Copyright Matt Borland 2022.
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "math_unit_test.hpp"
#include <limits>
#include <boost/math/special_functions/logaddexp.hpp>
#include <boost/math/constants/constants.hpp>

template <typename Real>
void test()
{
    using boost::math::logaddexp;
    using std::log;
    using std::exp;

    constexpr Real nan_val = std::numeric_limits<Real>::quiet_NaN();
    constexpr Real inf_val = std::numeric_limits<Real>::infinity();

    // NAN
    CHECK_NAN(logaddexp(nan_val, Real(1)));
    CHECK_NAN(logaddexp(Real(1), nan_val));
    CHECK_NAN(logaddexp(nan_val, nan_val));

    // INF
    CHECK_EQUAL(logaddexp(inf_val, Real(1)), inf_val);
    CHECK_EQUAL(logaddexp(Real(1), inf_val), inf_val);
    CHECK_EQUAL(logaddexp(inf_val, inf_val), inf_val);

    // Equal values
    constexpr Real ln2 = boost::math::constants::ln_two<Real>();
    CHECK_ULP_CLOSE(Real(2) + ln2, logaddexp(Real(2), Real(2)), 1);
    CHECK_ULP_CLOSE(Real(1e-50) + ln2, logaddexp(Real(1e-50), Real(1e-50)), 1);

    // Spot check
    // https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html
    // Calculated at higher precision using wolfram alpha
    Real x1 = 1e-50l;
    Real x2 = 2.5e-50l;
    Real spot1 = static_cast<Real>(exp(x1));
    Real spot2 = static_cast<Real>(exp(x2));
    Real spot12 = logaddexp(x1, x2);

    CHECK_ULP_CLOSE(log(spot1 + spot2), spot12, 1);
}

int main (void)
{
    test<float>();
    test<double>();
    test<long double>();
    return boost::math::test::report_errors();
}

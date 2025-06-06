//  (C) Copyright Matt Borland and Nick Thompson 2022.
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "math_unit_test.hpp"
#include <cmath>
#include <vector>
#include <boost/math/special_functions/logsumexp.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/tools/random_vector.hpp>

template <typename Real>
void test()
{
    using boost::math::logsumexp;
    using std::log;
    using std::exp;

    // Spot check 2 values
    // Also validate that 2 values does not attempt to instantiate the iterator version
    // https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html
    // Calculated at higher precision using wolfram alpha
    Real x1 = 1e-50l;
    Real x2 = 2.5e-50l;
    Real spot1 = static_cast<Real>(exp(x1));
    Real spot2 = static_cast<Real>(exp(x2));
    Real spot12 = logsumexp(x1, x2);
    CHECK_ULP_CLOSE(log(spot1 + spot2), spot12, 1);

    // Spot check 3 values and compare result of each different interface
    Real x3 = 5e-50l;
    Real spot3 = static_cast<Real>(exp(x3));
    std::vector<Real> x_vals {x1, x2, x3};

    Real spot123 = logsumexp(x1, x2, x3);
    Real spot123_container = logsumexp(x_vals);
    Real spot123_iter = logsumexp(x_vals.begin(), x_vals.end());

    CHECK_EQUAL(spot123, spot123_container);
    CHECK_EQUAL(spot123_container, spot123_iter);
    CHECK_ULP_CLOSE(log(spot1 + spot2 + spot3), spot123, 1);

    // Spot check 4 values with repeated largest value
    Real x4 = x3;
    Real spot4 = spot3;
    Real spot1234 = logsumexp(x1, x2, x3, x4);
    x_vals.emplace_back(x4);
    Real spot1234_container = logsumexp(x_vals);

    CHECK_EQUAL(spot1234, spot1234_container);
    CHECK_ULP_CLOSE(log(spot1 + spot2 + spot3 + spot4), spot1234, 1);

    // Check with a value of vastly different order of magnitude
    Real x5 = 1.0l;
    Real spot5 = static_cast<Real>(exp(x5));
    x_vals.emplace_back(x5);
    Real spot12345 = logsumexp(x_vals);
    CHECK_ULP_CLOSE(log(spot1 + spot2 + spot3 + spot4 + spot5), spot12345, 1);
}

// The naive method of computation should overflow:
template<typename Real>
void test_overflow() 
{
    using boost::math::logsumexp;
    using std::exp;
    using std::log;

    Real x = ((std::numeric_limits<Real>::max)()/2);

    Real naive_result = log(exp(x) + exp(x));
    CHECK_EQUAL(std::isfinite(naive_result), false);

    Real result = logsumexp(x, x);
    CHECK_EQUAL(std::isfinite(result), true);
    CHECK_ULP_CLOSE(result, x + boost::math::constants::ln_two<Real>(), 1);
}

template <typename Real>
void test_random()
{
    using std::exp;
    using std::log;
    using boost::math::logsumexp;
    using boost::math::generate_random_vector;
    
    std::vector<Real> test_values = generate_random_vector(128, 0, Real(1e-50l), Real(1e-40l));
    Real naive_exp_sum = 0;

    for(const auto& val : test_values)
    {
        naive_exp_sum += exp(val);
    }

    CHECK_ULP_CLOSE(log(naive_exp_sum), logsumexp(test_values), 1);
}

int main (void)
{
    test<float>();
    test<double>();
    test<long double>();

    test_overflow<float>();
    test_overflow<double>();
    test_overflow<long double>();

    test_random<float>();
    test_random<double>();
    test_random<long double>();
    return boost::math::test::report_errors();
}

// Copyright Evan Miller 2020
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#include <pch_light.hpp>
#include <boost/math/concepts/real_concept.hpp>

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // for test_main
#include <boost/test/tools/floating_point_comparison.hpp> // for BOOST_CHECK_CLOSE
#include <boost/math/distributions/kolmogorov_smirnov.hpp>
#include <boost/math/quadrature/exp_sinh.hpp>

template <typename RealType> // Any floating-point type RealType.
void test_spots(RealType)
{
    using namespace boost::math;
    // Test quantiles, CDFs, and complements
    RealType eps = tools::epsilon<RealType>();
    RealType tol = tools::epsilon<RealType>() * 25;
    for (int n=10; n<100; n += 10) {
        kolmogorov_smirnov_distribution<RealType> dist(n);
        for (int i=0; i<1000; i++) {
            RealType p = 1.0 * (i+1) / 1001;
            RealType crit1 = quantile(dist, 1 - p);
            RealType crit2 = quantile(complement(dist, p));
            RealType p1 = cdf(dist, crit1);
            BOOST_CHECK_CLOSE_FRACTION(crit1, crit2, tol);
            BOOST_CHECK_CLOSE_FRACTION(1 - p, p1, tol);
        }

        for (int i=0; i<1000; i++) {
            RealType x = 1.0 * (i+1) / 1001;
            RealType p = cdf(dist, x);
            RealType p1 = cdf(complement(dist, x));
            RealType x1;
            if (p < 0.5)
                x1 = quantile(dist, p);
            else
                x1 = quantile(complement(dist, p1));
            if (p > tol && p1 > tol) // skip the extreme tails
                BOOST_CHECK_CLOSE_FRACTION(x, x1, tol);
        }
    }

    kolmogorov_smirnov_distribution<RealType> dist(100);

    // Basics
    BOOST_CHECK_THROW(pdf(dist, RealType(-1.0)), std::domain_error);
    BOOST_CHECK_THROW(cdf(dist, RealType(-1.0)), std::domain_error);
    BOOST_CHECK_THROW(quantile(dist, RealType(-1.0)), std::domain_error);
    BOOST_CHECK_THROW(quantile(dist, RealType(2.0)), std::domain_error);

    // Confirm mode is at least a local minimum
    RealType mode = boost::math::mode(dist);

    using std::sqrt;
    BOOST_TEST_CHECK(pdf(dist, mode) >= pdf(dist, RealType(mode - sqrt(eps))));
    BOOST_TEST_CHECK(pdf(dist, mode) >= pdf(dist, RealType(mode + sqrt(eps))));

    // Test the moments - each one integrates the entire distribution
    quadrature::exp_sinh<RealType> integrator;

    auto f_one = [&, dist](RealType t) { return pdf(dist, t); };
    BOOST_CHECK_CLOSE_FRACTION(integrator.integrate(f_one, eps), RealType(1), tol);

    RealType mean = boost::math::mean(dist);
    auto f_mean = [&, dist](RealType t) { return pdf(dist, t) * t; };
    BOOST_CHECK_CLOSE_FRACTION(integrator.integrate(f_mean, eps), mean, tol);

    RealType var = variance(dist);
    auto f_var = [&, dist, mean](RealType t) { return pdf(dist, t) * (t - mean) * (t - mean); };
    BOOST_CHECK_CLOSE_FRACTION(integrator.integrate(f_var, eps), var, tol);

    RealType skew = skewness(dist);
    auto f_skew = [&, dist, mean, var](RealType t) { return pdf(dist, t)
        * (t - mean) * (t - mean) * (t - mean) / var / sqrt(var); };
    BOOST_CHECK_CLOSE_FRACTION(integrator.integrate(f_skew, eps), skew, 10*tol);

    RealType kurt = kurtosis(dist);
    auto f_kurt= [&, dist, mean, var](RealType t) { return pdf(dist, t)
        * (t - mean) * (t - mean) * (t - mean) * (t - mean) / var / var; };
    BOOST_CHECK_CLOSE_FRACTION(integrator.integrate(f_kurt, eps), kurt, 5*tol);

    BOOST_CHECK_CLOSE_FRACTION(kurt, kurtosis_excess(dist) + 3, eps);
}

BOOST_AUTO_TEST_CASE( test_main )
{
  BOOST_MATH_CONTROL_FP;

  // (Parameter value, arbitrarily zero, only communicates the floating point type).
     test_spots(0.0F); // Test float.
 test_spots(0.0); // Test double.
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
  test_spots(0.0L); // Test long double.
#if !defined(BOOST_MATH_NO_REAL_CONCEPT_TESTS)
  test_spots(boost::math::concepts::real_concept(0.)); // Test real concept.
#endif
#endif
}

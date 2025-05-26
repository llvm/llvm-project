// Copyright John Maddock 2006, 2012.
// Copyright Paul A. Bristow 2007, 2012.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// test_weibull.cpp

#ifdef _MSC_VER
#  pragma warning (disable : 4127) //  conditional expression is constant.
#endif

#include <boost/math/tools/config.hpp>
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
#include <boost/math/concepts/real_concept.hpp> // for real_concept
#endif
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // Boost.Test
#include <boost/test/tools/floating_point_comparison.hpp>

#include <boost/math/distributions/weibull.hpp>
    using boost::math::weibull_distribution;
#include "../include_private/boost/math/tools/test.hpp"
#include "test_out_of_range.hpp"

#include <iostream>
   using std::cout;
   using std::endl;
   using std::setprecision;
#include <limits>
  using std::numeric_limits;
#include <cmath>
  using std::log;

template <class RealType>
void check_weibull(RealType shape, RealType scale, RealType x, RealType p, RealType q, RealType tol)
{
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         weibull_distribution<RealType>(shape, scale),       // distribution.
         x),                                            // random variable.
         p,                                             // probability.
         tol);                                          // %tolerance.
   BOOST_CHECK_CLOSE(
      ::boost::math::logcdf(
         weibull_distribution<RealType>(shape, scale),       // distribution.
         x),                                            // random variable.
         log(p),                                             // probability.
         tol);   
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         complement(
            weibull_distribution<RealType>(shape, scale),    // distribution.
            x)),                                        // random variable.
         q,                                             // probability complement.
         tol);                                          // %tolerance.
   BOOST_CHECK_CLOSE(
      ::boost::math::logcdf(
         complement(
            weibull_distribution<RealType>(shape, scale),    // distribution.
            x)),                                        // random variable.
         log(q),                                             // probability complement.
         tol);   
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         weibull_distribution<RealType>(shape, scale),       // distribution.
         p),                                            // probability.
         x,                                             // random variable.
         tol);                                          // %tolerance.
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         complement(
            weibull_distribution<RealType>(shape, scale),    // distribution.
            q)),                                        // probability complement.
         x,                                             // random variable.
         tol);                                          // %tolerance.
}

template <class RealType>
void test_spots(RealType)
{
   // Basic sanity checks
   //
   // These test values were generated for the normal distribution
   // using the online calculator at 
   // http://espse.ed.psu.edu/edpsych/faculty/rhale/hale/507Mat/statlets/free/pdist.htm
   //
   // Tolerance is just over 5 decimal digits expressed as a percentage:
   // that's the limit of the test data.
   RealType tolerance = 2e-5f * 100;  
   cout << "Tolerance for type " << typeid(RealType).name()  << " is " << tolerance << " %" << endl;

   using std::exp;

   check_weibull(
      static_cast<RealType>(0.25),     // shape
      static_cast<RealType>(0.5),     // scale
      static_cast<RealType>(0.1),     // x
      static_cast<RealType>(0.487646),   // p
      static_cast<RealType>(1-0.487646),   // q
      tolerance);
   check_weibull(
      static_cast<RealType>(0.25),     // shape
      static_cast<RealType>(0.5),     // scale
      static_cast<RealType>(0.5),     // x
      static_cast<RealType>(1-0.367879),   // p
      static_cast<RealType>(0.367879),   // q
      tolerance);
   check_weibull(
      static_cast<RealType>(0.25),     // shape
      static_cast<RealType>(0.5),     // scale
      static_cast<RealType>(1),     // x
      static_cast<RealType>(1-0.304463),   // p
      static_cast<RealType>(0.304463),   // q
      tolerance);
   check_weibull(
      static_cast<RealType>(0.25),     // shape
      static_cast<RealType>(0.5),     // scale
      static_cast<RealType>(2),     // x
      static_cast<RealType>(1-0.243117),   // p
      static_cast<RealType>(0.243117),   // q
      tolerance);
   check_weibull(
      static_cast<RealType>(0.25),     // shape
      static_cast<RealType>(0.5),     // scale
      static_cast<RealType>(5),     // x
      static_cast<RealType>(1-0.168929),   // p
      static_cast<RealType>(0.168929),   // q
      tolerance);

   check_weibull(
      static_cast<RealType>(0.5),     // shape
      static_cast<RealType>(2),     // scale
      static_cast<RealType>(0.1),     // x
      static_cast<RealType>(0.200371),   // p
      static_cast<RealType>(1-0.200371),   // q
      tolerance);
   check_weibull(
      static_cast<RealType>(0.5),     // shape
      static_cast<RealType>(2),     // scale
      static_cast<RealType>(0.5),     // x
      static_cast<RealType>(0.393469),   // p
      static_cast<RealType>(1-0.393469),   // q
      tolerance);
   check_weibull(
      static_cast<RealType>(0.5),     // shape
      static_cast<RealType>(2),     // scale
      static_cast<RealType>(1),     // x
      static_cast<RealType>(1-0.493069),   // p
      static_cast<RealType>(0.493069),   // q
      tolerance);
   check_weibull(
      static_cast<RealType>(0.5),     // shape
      static_cast<RealType>(2),     // scale
      static_cast<RealType>(2),     // x
      static_cast<RealType>(1-0.367879),   // p
      static_cast<RealType>(0.367879),   // q
      tolerance);
   check_weibull(
      static_cast<RealType>(0.5),     // shape
      static_cast<RealType>(2),     // scale
      static_cast<RealType>(5),     // x
      static_cast<RealType>(1-0.205741),   // p
      static_cast<RealType>(0.205741),   // q
      tolerance);

   check_weibull(
      static_cast<RealType>(2),     // shape
      static_cast<RealType>(0.25),     // scale
      static_cast<RealType>(0.1),     // x
      static_cast<RealType>(0.147856),   // p
      static_cast<RealType>(1-0.147856),   // q
      tolerance);
   check_weibull(
      static_cast<RealType>(2),     // shape
      static_cast<RealType>(0.25),     // scale
      static_cast<RealType>(0.5),     // x
      static_cast<RealType>(1-0.018316),   // p
      static_cast<RealType>(0.018316),   // q
      tolerance);

   /*
   This test value came from 
   http://espse.ed.psu.edu/edpsych/faculty/rhale/hale/507Mat/statlets/free/pdist.htm
   but appears to be grossly incorrect: certainly it does not agree with the values
   I get from pushing numbers into a calculator (0.0001249921878255106610615995196123).   
   Strangely other test values generated for the same shape and scale parameters do look OK.
   check_weibull(
      static_cast<RealType>(3),     // shape
      static_cast<RealType>(2),     // scale
      static_cast<RealType>(0.1),     // x
      static_cast<RealType>(1.25E-40),   // p
      static_cast<RealType>(1-1.25E-40),   // q
      tolerance);
      */
   check_weibull(
      static_cast<RealType>(3),     // shape
      static_cast<RealType>(2),     // scale
      static_cast<RealType>(0.5),     // x
      static_cast<RealType>(0.015504),   // p
      static_cast<RealType>(1-0.015504),   // q
      tolerance * 10); // few digits in test value
   check_weibull(
      static_cast<RealType>(3),     // shape
      static_cast<RealType>(2),     // scale
      static_cast<RealType>(1),     // x
      static_cast<RealType>(0.117503),   // p
      static_cast<RealType>(1-0.117503),   // q
      tolerance);
   check_weibull(
      static_cast<RealType>(3),     // shape
      static_cast<RealType>(2),     // scale
      static_cast<RealType>(2),     // x
      static_cast<RealType>(1-0.367879),   // p
      static_cast<RealType>(0.367879),   // q
      tolerance);

   //
   // Tests for PDF
   //
   BOOST_CHECK_CLOSE(
      pdf(weibull_distribution<RealType>(0.25, 0.5), static_cast<RealType>(0.1)), 
      static_cast<RealType>(0.856579), 
      tolerance);
   BOOST_CHECK_CLOSE(
      pdf(weibull_distribution<RealType>(0.25, 0.5), static_cast<RealType>(0.5)), 
      static_cast<RealType>(0.183940), 
      tolerance);
   BOOST_CHECK_CLOSE(
      pdf(weibull_distribution<RealType>(0.25, 0.5), static_cast<RealType>(5)), 
      static_cast<RealType>(0.015020), 
      tolerance * 10); // fewer digits in test value
   BOOST_CHECK_CLOSE(
      pdf(weibull_distribution<RealType>(0.5, 2), static_cast<RealType>(0.1)), 
      static_cast<RealType>(0.894013), 
      tolerance);
   BOOST_CHECK_CLOSE(
      pdf(weibull_distribution<RealType>(0.5, 2), static_cast<RealType>(0.5)), 
      static_cast<RealType>(0.303265), 
      tolerance);
   BOOST_CHECK_CLOSE(
      pdf(weibull_distribution<RealType>(0.5, 2), static_cast<RealType>(1)), 
      static_cast<RealType>(0.174326), 
      tolerance);
   BOOST_CHECK_CLOSE(
      pdf(weibull_distribution<RealType>(2, 0.25), static_cast<RealType>(0.1)), 
      static_cast<RealType>(2.726860), 
      tolerance);
   BOOST_CHECK_CLOSE(
      pdf(weibull_distribution<RealType>(2, 0.25), static_cast<RealType>(0.5)), 
      static_cast<RealType>(0.293050), 
      tolerance);
   BOOST_CHECK_CLOSE(
      pdf(weibull_distribution<RealType>(3, 2), static_cast<RealType>(1)), 
      static_cast<RealType>(0.330936), 
      tolerance);
   BOOST_CHECK_CLOSE(
      pdf(weibull_distribution<RealType>(3, 2), static_cast<RealType>(2)), 
      static_cast<RealType>(0.551819), 
      tolerance);

   //
   // Tests for logpdf
   //
   BOOST_CHECK_CLOSE(
      logpdf(weibull_distribution<RealType>(0.25, 0.5), static_cast<RealType>(0.1)), 
      log(static_cast<RealType>(0.856579)), 
      tolerance);
   BOOST_CHECK_CLOSE(
      logpdf(weibull_distribution<RealType>(0.25, 0.5), static_cast<RealType>(0.5)), 
      log(static_cast<RealType>(0.183940)), 
      tolerance);
   BOOST_CHECK_CLOSE(
      logpdf(weibull_distribution<RealType>(0.25, 0.5), static_cast<RealType>(5)), 
      log(static_cast<RealType>(0.015020)), 
      tolerance * 10); // fewer digits in test value
   BOOST_CHECK_CLOSE(
      logpdf(weibull_distribution<RealType>(0.5, 2), static_cast<RealType>(0.1)), 
      log(static_cast<RealType>(0.894013)), 
      tolerance);
   BOOST_CHECK_CLOSE(
      logpdf(weibull_distribution<RealType>(0.5, 2), static_cast<RealType>(0.5)), 
      log(static_cast<RealType>(0.303265)), 
      tolerance);
   BOOST_CHECK_CLOSE(
      logpdf(weibull_distribution<RealType>(0.5, 2), static_cast<RealType>(1)), 
      log(static_cast<RealType>(0.174326)), 
      tolerance);
   BOOST_CHECK_CLOSE(
      logpdf(weibull_distribution<RealType>(2, 0.25), static_cast<RealType>(0.1)), 
      log(static_cast<RealType>(2.726860)), 
      tolerance);
   BOOST_CHECK_CLOSE(
      logpdf(weibull_distribution<RealType>(2, 0.25), static_cast<RealType>(0.5)), 
      log(static_cast<RealType>(0.293050)), 
      tolerance);
   BOOST_CHECK_CLOSE(
      logpdf(weibull_distribution<RealType>(3, 2), static_cast<RealType>(1)), 
      log(static_cast<RealType>(0.330936)), 
      tolerance);
   BOOST_CHECK_CLOSE(
      logpdf(weibull_distribution<RealType>(3, 2), static_cast<RealType>(2)), 
      log(static_cast<RealType>(0.551819)), 
      tolerance);

   //
   // These test values were obtained using the formulas at 
   // http://en.wikipedia.org/wiki/Weibull_distribution
   // which are subtly different to (though mathematically
   // the same as) the ones on the Mathworld site
   // http://mathworld.wolfram.com/WeibullDistribution.html
   // which are the ones used in the implementation.
   // The assumption is that if both computation methods
   // agree then the implementation is probably correct...
   // What's not clear is which method is more accurate.
   //
   tolerance = (std::max)(
      boost::math::tools::epsilon<RealType>(),
      static_cast<RealType>(boost::math::tools::epsilon<double>())) * 5 * 100; // 5 eps as a percentage
   cout << "Tolerance for type " << typeid(RealType).name()  << " is " << tolerance << " %" << endl;
   weibull_distribution<RealType> dist(2, 3);
   RealType x = static_cast<RealType>(0.125);

   BOOST_MATH_STD_USING // ADL of std lib math functions

   // mean:
   BOOST_CHECK_CLOSE(
      mean(dist)
      , dist.scale() * boost::math::tgamma(1 + 1 / dist.shape()), tolerance);
   // variance:
   BOOST_CHECK_CLOSE(
      variance(dist)
      , dist.scale() * dist.scale() * boost::math::tgamma(1 + 2 / dist.shape()) - mean(dist) * mean(dist), tolerance);
   // std deviation:
   BOOST_CHECK_CLOSE(
    standard_deviation(dist)
    , sqrt(variance(dist)), tolerance);
   // hazard:
   BOOST_CHECK_CLOSE(
    hazard(dist, x)
    , pdf(dist, x) / cdf(complement(dist, x)), tolerance);
   // cumulative hazard:
   BOOST_CHECK_CLOSE(
    chf(dist, x)
    , -log(cdf(complement(dist, x))), tolerance);
   // coefficient_of_variation:
   BOOST_CHECK_CLOSE(
    coefficient_of_variation(dist)
    , standard_deviation(dist) / mean(dist), tolerance);
   // mode:
   BOOST_CHECK_CLOSE(
    mode(dist)
    , dist.scale() * pow((dist.shape() - 1) / dist.shape(), 1/dist.shape()), tolerance);
   // median:
   BOOST_CHECK_CLOSE(
    median(dist)
    , dist.scale() * pow(log(static_cast<RealType>(2)), 1 / dist.shape()), tolerance);
   // skewness:
   BOOST_CHECK_CLOSE(
    skewness(dist), 
    (boost::math::tgamma(1 + 3/dist.shape()) * pow(dist.scale(), RealType(3)) - 3 * mean(dist) * variance(dist) - pow(mean(dist), RealType(3))) / pow(standard_deviation(dist), RealType(3)), 
    tolerance * 100);
   // kurtosis:
   BOOST_CHECK_CLOSE(
    kurtosis(dist)
    , kurtosis_excess(dist) + 3, tolerance);
   // kurtosis excess:
   BOOST_CHECK_CLOSE(
    kurtosis_excess(dist), 
    (pow(dist.scale(), RealType(4)) * boost::math::tgamma(1 + 4/dist.shape()) 
         - 3 * variance(dist) * variance(dist) 
         - 4 * skewness(dist) * variance(dist) * standard_deviation(dist) * mean(dist)
         - 6 * variance(dist) * mean(dist) * mean(dist) 
         - pow(mean(dist), RealType(4))) / (variance(dist) * variance(dist)), 
    tolerance * 1000);

   RealType expected_entropy = boost::math::constants::euler<RealType>()*(1-1/dist.shape()) + log(dist.scale()/dist.shape()) + 1;
   BOOST_CHECK_CLOSE(
    entropy(dist)
    , expected_entropy, tolerance);

   //
   // Special cases:
   //
   BOOST_CHECK(cdf(dist, 0) == 0);
   BOOST_CHECK(cdf(complement(dist, 0)) == 1);
   BOOST_CHECK(quantile(dist, 0) == 0);
   BOOST_CHECK(quantile(complement(dist, 1)) == 0);

   BOOST_CHECK_EQUAL(pdf(weibull_distribution<RealType>(1, 1), 0), 1);

   //
   // Error checks:
   //
   BOOST_MATH_CHECK_THROW(weibull_distribution<RealType>(1, -1), std::domain_error);
   BOOST_MATH_CHECK_THROW(weibull_distribution<RealType>(-1, 1), std::domain_error);
   BOOST_MATH_CHECK_THROW(weibull_distribution<RealType>(1, 0), std::domain_error);
   BOOST_MATH_CHECK_THROW(weibull_distribution<RealType>(0, 1), std::domain_error);
   BOOST_MATH_CHECK_THROW(pdf(dist, -1), std::domain_error);
   BOOST_MATH_CHECK_THROW(cdf(dist, -1), std::domain_error);
   BOOST_MATH_CHECK_THROW(cdf(complement(dist, -1)), std::domain_error);
   BOOST_MATH_CHECK_THROW(quantile(dist, 1), std::overflow_error);
   BOOST_MATH_CHECK_THROW(quantile(complement(dist, 0)), std::overflow_error);
   BOOST_MATH_CHECK_THROW(quantile(dist, -1), std::domain_error);
   BOOST_MATH_CHECK_THROW(quantile(complement(dist, -1)), std::domain_error);

   BOOST_CHECK_EQUAL(pdf(dist, 0), exp(-pow(RealType(0) / RealType(3), RealType(2))) * pow(RealType(0), RealType(1)) * RealType(2) / RealType(3));
   BOOST_CHECK_EQUAL(pdf(weibull_distribution<RealType>(1, 3), 0), exp(-pow(RealType(0) / RealType(3), RealType(1))) * pow(RealType(0), RealType(0)) * RealType(1) / RealType(3));
   BOOST_MATH_CHECK_THROW(pdf(weibull_distribution<RealType>(0.5, 3), 0), std::overflow_error);

   check_out_of_range<weibull_distribution<RealType> >(1, 1);
} // template <class RealType>void test_spots(RealType)

BOOST_AUTO_TEST_CASE( test_main )
{

  // Check that can construct weibull distribution using the two convenience methods:
  using namespace boost::math;
  weibull myw1(2); // Using typedef
   weibull_distribution<> myw2(2); // Using default RealType double.

    // Basic sanity-check spot values.
   // (Parameter value, arbitrarily zero, only communicates the floating point type).
  test_spots(0.0F); // Test float. OK at decdigits = 0 tolerance = 0.0001 %
  test_spots(0.0); // Test double. OK at decdigits 7, tolerance = 1e07 %
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
  test_spots(0.0L); // Test long double.
#if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x0582)) && !defined(BOOST_MATH_NO_REAL_CONCEPT_TESTS)
  test_spots(boost::math::concepts::real_concept(0.)); // Test real concept.
#endif
#else
   std::cout << "<note>The long double tests have been disabled on this platform "
      "either because the long double overloads of the usual math functions are "
      "not available at all, or because they are too inaccurate for these tests "
      "to pass.</note>" << std::endl;
#endif

   
} // BOOST_AUTO_TEST_CASE( test_main )

/*

Output:

  Description: Autorun "J:\Cpp\MathToolkit\test\Math_test\Debug\test_weibull.exe"
  Running 1 test case...
  Tolerance for type float is 0.002 %
  Tolerance for type float is 5.96046e-005 %
  Tolerance for type double is 0.002 %
  Tolerance for type double is 1.11022e-013 %
  Tolerance for type long double is 0.002 %
  Tolerance for type long double is 1.11022e-013 %
  Tolerance for type class boost::math::concepts::real_concept is 0.002 %
  Tolerance for type class boost::math::concepts::real_concept is 1.11022e-013 %
  
  *** No errors detected


*/





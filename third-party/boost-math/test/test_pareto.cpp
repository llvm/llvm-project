// Copyright Paul A. Bristow 2007, 2009.
// Copyright John Maddock 2006.
// Copyright Matt Borland 2023.
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// test_pareto.cpp

// http://en.wikipedia.org/wiki/pareto_distribution
// http://www.itl.nist.gov/div898/handbook/eda/section3/eda3661.htm
// Also:
// Weisstein, Eric W. "pareto Distribution."
// From MathWorld--A Wolfram Web Resource.
// http://mathworld.wolfram.com/paretoDistribution.html


#ifdef _MSC_VER
#  pragma warning(disable: 4127) // conditional expression is constant.
# pragma warning (disable : 4996) // POSIX name for this item is deprecated
# pragma warning (disable : 4224) // nonstandard extension used : formal parameter 'arg' was previously defined as a type
# pragma warning (disable : 4180) // qualifier applied to function type has no meaning; ignored
#  pragma warning(disable: 4100) // unreferenced formal parameter.
#endif

#include <boost/math/tools/config.hpp>
#include "../include_private/boost/math/tools/test.hpp"

#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
#include <boost/math/concepts/real_concept.hpp> // for real_concept
#endif

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // Boost.Test
#include <boost/test/tools/floating_point_comparison.hpp>

#include <boost/math/distributions/pareto.hpp>
    using boost::math::pareto_distribution;
#include "test_out_of_range.hpp"

#include <iostream>
   using std::cout;
   using std::endl;
   using std::setprecision;
#include <limits>
  using std::numeric_limits;
#include <type_traits>

  template <class RealType>
  void check_pareto(RealType scale, RealType shape, RealType x, RealType p, RealType q, RealType tol)
  {
    RealType logtol = tol * 10;
    #ifndef BOOST_MATH_HAS_GPU_SUPPORT
    BOOST_IF_CONSTEXPR (std::is_same<RealType, long double>::value || 
                        std::is_same<RealType, boost::math::concepts::real_concept>::value)
    {
      logtol *= 100;
    }
    #endif
    
    BOOST_CHECK_CLOSE_FRACTION(
      ::boost::math::cdf(
      pareto_distribution<RealType>(scale, shape),   // distribution.
      x),                                            // random variable.
      p,                                             // probability.
      tol);                                          // tolerance eps.
    BOOST_CHECK_CLOSE_FRACTION(
      ::boost::math::cdf(
      complement(
      pareto_distribution<RealType>(scale, shape),   // distribution.
      x)),                                           // random variable.
      q,                                             // probability complement.
      tol);                                          // tolerance eps.
    BOOST_CHECK_CLOSE_FRACTION(
      ::boost::math::logcdf(
      pareto_distribution<RealType>(scale, shape),   // distribution.
      x),                                            // random variable.
      log(p),                                             // probability.
      logtol);                                          // tolerance eps.
    BOOST_CHECK_CLOSE_FRACTION(
      ::boost::math::logcdf(
      complement(
      pareto_distribution<RealType>(scale, shape),   // distribution.
      x)),                                           // random variable.
      log(q),                                             // probability complement.
      logtol);                                          // tolerance eps.
    BOOST_CHECK_CLOSE_FRACTION(
      ::boost::math::quantile(
      pareto_distribution<RealType>(scale, shape),   // distribution.
      p),                                            // probability.
      x,                                             // random variable.
      tol);                                          // tolerance eps.
    BOOST_CHECK_CLOSE_FRACTION(
      ::boost::math::quantile(
      complement(
      pareto_distribution<RealType>(scale, shape),    // distribution.
      q)),                                        // probability complement.
      x,                                             // random variable.
      tol);                                          // tolerance eps.
  } // check_pareto

template <class RealType>
void test_spots(RealType)
{
   // Basic sanity checks.
   //
   // Tolerance are based on units of epsilon, but capped at
   // double precision, since that's the limit of our test data:
   //
   RealType tol = (std::max)((RealType)boost::math::tools::epsilon<double>(), boost::math::tools::epsilon<RealType>());
   RealType tol5eps = tol * 5;
   RealType tol10eps = tol * 10;
   RealType tol100eps = tol * 100;
   RealType tol1000eps = tol * 1000;

   check_pareto(
      static_cast<RealType>(1.1L), //
      static_cast<RealType>(5.5L),
      static_cast<RealType>(2.2L),
      static_cast<RealType>(0.97790291308792L),
      static_cast<RealType>(0.0220970869120796L),
      tol10eps * 4);

   check_pareto(
      static_cast<RealType>(0.5L),
      static_cast<RealType>(10.1L),
      static_cast<RealType>(1.5L),
      static_cast<RealType>(0.99998482686481L),
      static_cast<RealType>(1.51731351900608e-005L),
      tol100eps * 1000); // Much less accurate as p close to unity.

   check_pareto(
      static_cast<RealType>(0.1L),
      static_cast<RealType>(2.3L),
      static_cast<RealType>(1.5L),
      static_cast<RealType>(0.99802762220697L),
      static_cast<RealType>(0.00197237779302972L),
      tol1000eps);

   // Example from 23.3 page 259
   check_pareto(
      static_cast<RealType>(2.30444301457005L),
      static_cast<RealType>(4),
      static_cast<RealType>(2.4L),
      static_cast<RealType>(0.15L),
      static_cast<RealType>(0.85L),
      tol100eps);

   check_pareto(
      static_cast<RealType>(2),
      static_cast<RealType>(3),
      static_cast<RealType>(3.4L),
      static_cast<RealType>(0.796458375737838L),
      static_cast<RealType>(0.203541624262162L),
      tol10eps);

   check_pareto( // Probability near 0.5
      static_cast<RealType>(2),
      static_cast<RealType>(2),
      static_cast<RealType>(3),
      static_cast<RealType>(0.5555555555555555555555555555555555555556L),
      static_cast<RealType>(0.4444444444444444444444444444444444444444L),
      tol5eps); // accurate.


   // Tests for:

   // pdf for shapes 1, 2 & 3 (exact)
   BOOST_CHECK_CLOSE_FRACTION(
      pdf(pareto_distribution<RealType>(1, 1), 1),
      static_cast<RealType>(1), //
      tol5eps);

    BOOST_CHECK_CLOSE_FRACTION(   pdf(pareto_distribution<RealType>(1, 2), 1),
      static_cast<RealType>(2), //
      tol5eps);

     BOOST_CHECK_CLOSE_FRACTION(   pdf(pareto_distribution<RealType>(1, 3), 1),
      static_cast<RealType>(3), //
      tol5eps);

   // cdf
   BOOST_CHECK_EQUAL( // x = scale
      cdf(pareto_distribution<RealType>(1, 1), 1),
      static_cast<RealType>(0) );

   // Compare with values from StatCalc K. Krishnamoorthy,  ISBN 1-58488-635-8 eq 23.1.3
   BOOST_CHECK_CLOSE_FRACTION( // small x
      cdf(pareto_distribution<RealType>(2, 5), static_cast<RealType>(3.4)),
      static_cast<RealType>(0.929570372227626L), tol5eps);

   BOOST_CHECK_CLOSE_FRACTION( // small x
      cdf(pareto_distribution<RealType>(2, 5), static_cast<RealType>(3.4)),
      static_cast<RealType>(1 - 0.0704296277723743L), tol5eps);

   BOOST_CHECK_CLOSE_FRACTION( // small x
      cdf(complement(pareto_distribution<RealType>(2, 5), static_cast<RealType>(3.4))),
      static_cast<RealType>(0.0704296277723743L), tol5eps);

   // quantile
   BOOST_CHECK_EQUAL( // x = scale
      quantile(pareto_distribution<RealType>(1, 1), 0),
      static_cast<RealType>(1) );

   BOOST_CHECK_EQUAL( // x = scale
      quantile(complement(pareto_distribution<RealType>(1, 1), 1)),
      static_cast<RealType>(1) );

   BOOST_CHECK_CLOSE_FRACTION( // small x
      cdf(complement(pareto_distribution<RealType>(2, 5), static_cast<RealType>(3.4))),
      static_cast<RealType>(0.0704296277723743L), tol5eps);

    using namespace std; // ADL of std names.

    pareto_distribution<RealType> pareto15(1, 5);
    // Note: shape must be big enough (5) that all moments up to kurtosis are defined
    // to allow all functions to be tested.

    // mean:
    BOOST_CHECK_CLOSE_FRACTION(
       mean(pareto15), static_cast<RealType>(1.25), tol5eps); // 1.25 == 5/4
    BOOST_CHECK_EQUAL(
       mean(pareto15), static_cast<RealType>(1.25)); // 1.25 == 5/4 (expect exact so check equal)

    pareto_distribution<RealType> p12(1, 2); //
    BOOST_CHECK_EQUAL(
       mean(p12), static_cast<RealType>(2)); // Exactly two.

    // variance:
   BOOST_CHECK_CLOSE_FRACTION(
       variance(pareto15), static_cast<RealType>(0.10416666666666667L), tol5eps);
    // std deviation:
    BOOST_CHECK_CLOSE_FRACTION(
       standard_deviation(pareto15), static_cast<RealType>(0.32274861218395140L), tol5eps);
    // hazard:   No independent test values found yet.
    //BOOST_CHECK_CLOSE_FRACTION(
    //   hazard(pareto15, x), pdf(pareto15, x) / cdf(complement(pareto15, x)), tol5eps);
    //// cumulative hazard:
    //BOOST_CHECK_CLOSE_FRACTION(
    //   chf(pareto15, x), -log(cdf(complement(pareto15, x))), tol5eps);
    //// coefficient_of_variation:
    BOOST_CHECK_CLOSE_FRACTION(
       coefficient_of_variation(pareto15), static_cast<RealType>(0.25819888974716110L), tol5eps);
    // mode:
    BOOST_CHECK_CLOSE_FRACTION(
       mode(pareto15), static_cast<RealType>(1), tol5eps);

    BOOST_CHECK_CLOSE_FRACTION(
       median(pareto15), static_cast<RealType>(1.1486983549970351L), tol5eps);

    // skewness:
    BOOST_CHECK_CLOSE_FRACTION(
       skewness(pareto15), static_cast<RealType>(4.6475800154489004L), tol5eps);
    // kurtosis:
    BOOST_CHECK_CLOSE_FRACTION(
       kurtosis(pareto15), static_cast<RealType>(73.8L), tol5eps);
    // kurtosis excess:
    BOOST_CHECK_CLOSE_FRACTION(
       kurtosis_excess(pareto15), static_cast<RealType>(70.8L), tol5eps);
    // Check difference between kurtosis and excess:
    BOOST_CHECK_CLOSE_FRACTION(
      kurtosis_excess(pareto15), kurtosis(pareto15) - static_cast<RealType>(3L), tol5eps);
    // Check kurtosis excess = kurtosis - 3;

    RealType expected_entropy = 1 + RealType(1)/RealType(5) + log(RealType(1)/RealType(5));
    BOOST_CHECK_CLOSE_FRACTION(
      entropy(pareto15), expected_entropy, tol5eps);

    // Error condition checks:
    check_out_of_range<pareto_distribution<RealType> >(1, 1);
    BOOST_MATH_CHECK_THROW(pdf(pareto_distribution<RealType>(0, 1), 0), std::domain_error);
    BOOST_MATH_CHECK_THROW(pdf(pareto_distribution<RealType>(1, 0), 0), std::domain_error);
    BOOST_MATH_CHECK_THROW(pdf(pareto_distribution<RealType>(-1, 1), 0), std::domain_error);
    BOOST_MATH_CHECK_THROW(pdf(pareto_distribution<RealType>(1, -1), 0), std::domain_error);

    BOOST_MATH_CHECK_THROW(pdf(pareto_distribution<RealType>(1, 1), 0), std::domain_error);
    BOOST_MATH_CHECK_THROW(cdf(pareto_distribution<RealType>(1, 1), 0), std::domain_error);

    BOOST_MATH_CHECK_THROW(quantile(pareto_distribution<RealType>(1, 1), -1), std::domain_error);
    BOOST_MATH_CHECK_THROW(quantile(pareto_distribution<RealType>(1, 1), 2), std::domain_error);
} // template <class RealType>void test_spots(RealType)

BOOST_AUTO_TEST_CASE( test_main )
{
  // Check that can generate pareto distribution using the two convenience methods:
   boost::math::pareto myp1(1., 1); // Using typedef
   pareto_distribution<> myp2(1., 1); // Using default RealType double.
  boost::math::pareto pareto11; // Use default values (scale = 1, shape = 1).
  // Note NOT pareto11() as the compiler will interpret as a function!
   // Basic sanity-check spot values.

  BOOST_CHECK_EQUAL(pareto11.scale(), 1); // Check defaults again.
  BOOST_CHECK_EQUAL(pareto11.shape(), 1);
  BOOST_CHECK_EQUAL(myp1.scale(), 1);
  BOOST_CHECK_EQUAL(myp1.shape(), 1);
  BOOST_CHECK_EQUAL(myp2.scale(), 1);
  BOOST_CHECK_EQUAL(myp2.shape(), 1);

  // Test range and support using double only,
  // because it supports numeric_limits max for pseudo-infinity.
  BOOST_CHECK_EQUAL(range(myp2).first, 0); // range 0 to +infinity
  BOOST_CHECK_EQUAL(range(myp2).second, (numeric_limits<double>::max)());
  BOOST_CHECK_EQUAL(support(myp2).first, myp2.scale()); // support scale to + infinity.
  BOOST_CHECK_EQUAL(support(myp2).second, (numeric_limits<double>::max)());

  // Check some bad parameters to the distribution.
#ifndef BOOST_NO_EXCEPTIONS
   BOOST_MATH_CHECK_THROW(boost::math::pareto mypm1(-1, 1), std::domain_error); // Using typedef
   BOOST_MATH_CHECK_THROW(boost::math::pareto myp0(0, 1), std::domain_error); // Using typedef
   BOOST_MATH_CHECK_THROW(boost::math::pareto myp1m1(1, -1), std::domain_error); // Using typedef
   BOOST_MATH_CHECK_THROW(boost::math::pareto myp10(1, 0), std::domain_error); // Using typedef
#else
   BOOST_MATH_CHECK_THROW(boost::math::pareto(-1, 1), std::domain_error); // Using typedef
   BOOST_MATH_CHECK_THROW(boost::math::pareto(0, 1), std::domain_error); // Using typedef
   BOOST_MATH_CHECK_THROW(boost::math::pareto(1, -1), std::domain_error); // Using typedef
   BOOST_MATH_CHECK_THROW(boost::math::pareto(1, 0), std::domain_error); // Using typedef
#endif

  // Check some moments that should fail because shape not big enough.
  BOOST_MATH_CHECK_THROW(variance(myp2), std::domain_error);
  BOOST_MATH_CHECK_THROW(standard_deviation(myp2), std::domain_error);
  BOOST_MATH_CHECK_THROW(skewness(myp2), std::domain_error);
  BOOST_MATH_CHECK_THROW(kurtosis(myp2), std::domain_error);
  BOOST_MATH_CHECK_THROW(kurtosis_excess(myp2), std::domain_error);

  // Test on extreme values of distribution parameters,
  // using just double because it has numeric_limit infinity etc.
#ifndef BOOST_NO_EXCEPTIONS
  BOOST_MATH_CHECK_THROW(boost::math::pareto mypinf1(+std::numeric_limits<double>::infinity(), 1), std::domain_error); // Using typedef
   BOOST_MATH_CHECK_THROW(boost::math::pareto myp1inf(1, +std::numeric_limits<double>::infinity()), std::domain_error); // Using typedef
   BOOST_MATH_CHECK_THROW(boost::math::pareto mypinf1(+std::numeric_limits<double>::infinity(), +std::numeric_limits<double>::infinity()), std::domain_error); // Using typedef
#else
  BOOST_MATH_CHECK_THROW(boost::math::pareto(+std::numeric_limits<double>::infinity(), 1), std::domain_error); // Using typedef
   BOOST_MATH_CHECK_THROW(boost::math::pareto(1, +std::numeric_limits<double>::infinity()), std::domain_error); // Using typedef
   BOOST_MATH_CHECK_THROW(boost::math::pareto(+std::numeric_limits<double>::infinity(), +std::numeric_limits<double>::infinity()), std::domain_error); // Using typedef
#endif

  // Test on extreme values of random variate x, using just double because it has numeric_limit infinity etc..
  // No longer allow x to be + or - infinity, then these tests should throw.
  BOOST_MATH_CHECK_THROW(pdf(pareto11, +std::numeric_limits<double>::infinity()), std::domain_error); // x = + infinity
  BOOST_MATH_CHECK_THROW(pdf(pareto11, -std::numeric_limits<double>::infinity()), std::domain_error); // x = - infinity
  BOOST_MATH_CHECK_THROW(cdf(pareto11, +std::numeric_limits<double>::infinity()), std::domain_error); // x = + infinity
  BOOST_MATH_CHECK_THROW(cdf(pareto11, -std::numeric_limits<double>::infinity()), std::domain_error); // x = - infinity

  BOOST_CHECK_EQUAL(pdf(pareto11, 0.5), 0); // x < scale but > 0
  BOOST_CHECK_EQUAL(pdf(pareto11, (std::numeric_limits<double>::min)()), 0); // x almost zero but > 0
  BOOST_CHECK_EQUAL(pdf(pareto11, 1), 1); // x == scale, result == shape == 1
  BOOST_CHECK_EQUAL(pdf(pareto11, +(std::numeric_limits<double>::max)()), 0); // x = +max, pdf has fallen to zero.

  BOOST_MATH_CHECK_THROW(pdf(pareto11, 0), std::domain_error); // x == 0
  BOOST_MATH_CHECK_THROW(pdf(pareto11, -1), std::domain_error); // x = -1
  BOOST_MATH_CHECK_THROW(pdf(pareto11, -(std::numeric_limits<double>::max)()), std::domain_error); // x = - max
  BOOST_MATH_CHECK_THROW(pdf(pareto11, -(std::numeric_limits<double>::min)()), std::domain_error); // x = - min

  BOOST_CHECK_EQUAL(cdf(pareto11, 1), 0); // x == scale, cdf = zero.
  BOOST_CHECK_EQUAL(cdf(pareto11, +(std::numeric_limits<double>::max)()), 1); // x = + max, cdf = unity.

  BOOST_MATH_CHECK_THROW(cdf(pareto11, 0), std::domain_error); // x == 0
  BOOST_MATH_CHECK_THROW(cdf(pareto11, -(std::numeric_limits<double>::min)()), std::domain_error); // x = - min,
  BOOST_MATH_CHECK_THROW(cdf(pareto11, -(std::numeric_limits<double>::max)()), std::domain_error); // x = - max,

   // (Parameter value, arbitrarily zero, only communicates the floating point type).
  test_spots(0.0F); // Test float. OK at decdigits = 0 tol5eps = 0.0001 %
  test_spots(0.0); // Test double. OK at decdigits 7, tol5eps = 1e07 %
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

Compiling...
test_pareto.cpp
Linking...
Embedding manifest...
Autorun "i:\boost-06-05-03-1300\libs\math\test\Math_test\debug\test_pareto.exe"
Running 1 test case...
*** No errors detected



*/



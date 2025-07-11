// Copyright Paul A. Bristow 2012.
// Copyright John Maddock 2012.
// Copyright Benjamin Sobotta 2012

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef _MSC_VER
#  pragma warning (disable : 4127) // conditional expression is constant.
#  pragma warning (disable : 4305) // 'initializing' : truncation from 'double' to 'const float'.
#  pragma warning (disable : 4310) // cast truncates constant value.
#  pragma warning (disable : 4512) // assignment operator could not be generated.
#endif

//#include <pch.hpp> // include directory libs/math/src/tr1/ is needed.

#include <boost/math/concepts/real_concept.hpp> // for real_concept
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // Boost.Test
#include <boost/test/tools/floating_point_comparison.hpp>

#include <boost/math/distributions/skew_normal.hpp>
using boost::math::skew_normal_distribution;
using boost::math::skew_normal;
#include <boost/math/tools/test.hpp> 

#include <iostream>
#include <iomanip>
using std::cout;
using std::endl;
using std::setprecision;
#include <limits>
using std::numeric_limits;
#include "test_out_of_range.hpp"

template <class RealType>
void check_skew_normal(RealType mean, RealType scale, RealType shape, RealType x, RealType p, RealType q, RealType tol)
{
 using boost::math::skew_normal_distribution;

  BOOST_CHECK_CLOSE_FRACTION(
    ::boost::math::cdf(   // Check cdf
    skew_normal_distribution<RealType>(mean, scale, shape),      // distribution.
    x),    // random variable.
    p,     // probability.
    tol);   // tolerance.
  BOOST_CHECK_CLOSE_FRACTION(
    ::boost::math::cdf( // Check cdf complement
    complement( 
    skew_normal_distribution<RealType>(mean, scale, shape),   // distribution.
    x)),   // random variable.
    q,      // probability complement.
    tol);    // %tolerance.
  BOOST_CHECK_CLOSE_FRACTION(
    ::boost::math::quantile( // Check quantile
    skew_normal_distribution<RealType>(mean, scale, shape),    // distribution.
    p),   // probability.
    x,   // random variable.
    tol);   // tolerance.
  BOOST_CHECK_CLOSE_FRACTION(
    ::boost::math::quantile( // Check quantile complement
    complement(
    skew_normal_distribution<RealType>(mean, scale, shape),   // distribution.
    q)),   // probability complement.
    x,     // random variable.
    tol);  // tolerance.

   skew_normal_distribution<RealType> dist (mean, scale, shape);

   if((p < 0.999) && (q < 0.999))
   {  // We can only check this if P is not too close to 1,
      // so that we can guarantee Q is accurate:
      BOOST_CHECK_CLOSE_FRACTION(
        cdf(complement(dist, x)), q, tol); // 1 - cdf
      BOOST_CHECK_CLOSE_FRACTION(
        quantile(dist, p), x, tol); // quantile(cdf) = x
      BOOST_CHECK_CLOSE_FRACTION(
        quantile(complement(dist, q)), x, tol); // quantile(complement(1 - cdf)) = x
   }
} // template <class RealType>void check_skew_normal()


template <class RealType>
void test_spots(RealType)
{
   // Basic sanity checks
   RealType tolerance = 1e-4f; // 1e-4 (as %)

  // Check some bad parameters to the distribution,
#ifndef BOOST_NO_EXCEPTIONS
   BOOST_MATH_CHECK_THROW(boost::math::skew_normal_distribution<RealType> nbad1(0, 0), std::domain_error); // zero sd
   BOOST_MATH_CHECK_THROW(boost::math::skew_normal_distribution<RealType> nbad1(0, -1), std::domain_error); // negative sd
#else
   BOOST_MATH_CHECK_THROW(boost::math::skew_normal_distribution<RealType>(0, 0), std::domain_error); // zero sd
   BOOST_MATH_CHECK_THROW(boost::math::skew_normal_distribution<RealType>(0, -1), std::domain_error); // negative sd
#endif
  // Tests on extreme values of random variate x, if has numeric_limit infinity etc.
    skew_normal_distribution<RealType> N01;
  if(std::numeric_limits<RealType>::has_infinity)
  {
    BOOST_CHECK_EQUAL(pdf(N01, +std::numeric_limits<RealType>::infinity()), 0); // x = + infinity, pdf = 0
    BOOST_CHECK_EQUAL(pdf(N01, -std::numeric_limits<RealType>::infinity()), 0); // x = - infinity, pdf = 0
    BOOST_CHECK_EQUAL(cdf(N01, +std::numeric_limits<RealType>::infinity()), 1); // x = + infinity, cdf = 1
    BOOST_CHECK_EQUAL(cdf(N01, -std::numeric_limits<RealType>::infinity()), 0); // x = - infinity, cdf = 0
    BOOST_CHECK_EQUAL(cdf(complement(N01, +std::numeric_limits<RealType>::infinity())), 0); // x = + infinity, c cdf = 0
    BOOST_CHECK_EQUAL(cdf(complement(N01, -std::numeric_limits<RealType>::infinity())), 1); // x = - infinity, c cdf = 1
#ifndef BOOST_NO_EXCEPTIONS
    BOOST_MATH_CHECK_THROW(boost::math::skew_normal_distribution<RealType> nbad1(std::numeric_limits<RealType>::infinity(), static_cast<RealType>(1)), std::domain_error); // +infinite mean
    BOOST_MATH_CHECK_THROW(boost::math::skew_normal_distribution<RealType> nbad1(-std::numeric_limits<RealType>::infinity(),  static_cast<RealType>(1)), std::domain_error); // -infinite mean
    BOOST_MATH_CHECK_THROW(boost::math::skew_normal_distribution<RealType> nbad1(static_cast<RealType>(0), std::numeric_limits<RealType>::infinity()), std::domain_error); // infinite sd
#else
    BOOST_MATH_CHECK_THROW(boost::math::skew_normal_distribution<RealType>(std::numeric_limits<RealType>::infinity(), static_cast<RealType>(1)), std::domain_error); // +infinite mean
    BOOST_MATH_CHECK_THROW(boost::math::skew_normal_distribution<RealType>(-std::numeric_limits<RealType>::infinity(),  static_cast<RealType>(1)), std::domain_error); // -infinite mean
    BOOST_MATH_CHECK_THROW(boost::math::skew_normal_distribution<RealType>(static_cast<RealType>(0), std::numeric_limits<RealType>::infinity()), std::domain_error); // infinite sd
#endif
  }

  if (std::numeric_limits<RealType>::has_quiet_NaN)
  {
    // No longer allow x to be NaN, then these tests should throw.
    BOOST_MATH_CHECK_THROW(pdf(N01, +std::numeric_limits<RealType>::quiet_NaN()), std::domain_error); // x = NaN
    BOOST_MATH_CHECK_THROW(cdf(N01, +std::numeric_limits<RealType>::quiet_NaN()), std::domain_error); // x = NaN
    BOOST_MATH_CHECK_THROW(cdf(complement(N01, +std::numeric_limits<RealType>::quiet_NaN())), std::domain_error); // x = + infinity
    BOOST_MATH_CHECK_THROW(quantile(N01, +std::numeric_limits<RealType>::quiet_NaN()), std::domain_error); // p = + infinity
    BOOST_MATH_CHECK_THROW(quantile(complement(N01, +std::numeric_limits<RealType>::quiet_NaN())), std::domain_error); // p = + infinity
  }

  BOOST_CHECK_EQUAL(mean(N01), 0);
  BOOST_CHECK_EQUAL(mode(N01), 0);
  BOOST_CHECK_EQUAL(variance(N01), 1);
  BOOST_CHECK_EQUAL(skewness(N01), 0);
  BOOST_CHECK_EQUAL(kurtosis_excess(N01), 0);

   cout << "Tolerance for type " << typeid(RealType).name()  << " is " << tolerance << " %" << endl;

   // Tests where shape = 0, so same as normal tests.
   // (These might be removed later).
   check_skew_normal(
      static_cast<RealType>(5),
      static_cast<RealType>(2),
      static_cast<RealType>(0),
      static_cast<RealType>(4.8),
      static_cast<RealType>(0.46017),
      static_cast<RealType>(1 - 0.46017),
      tolerance);

   check_skew_normal(
      static_cast<RealType>(5),
      static_cast<RealType>(2),
      static_cast<RealType>(0),
      static_cast<RealType>(5.2),
      static_cast<RealType>(1 - 0.46017),
      static_cast<RealType>(0.46017),
      tolerance);

   check_skew_normal(
      static_cast<RealType>(5),
      static_cast<RealType>(2),
      static_cast<RealType>(0),
      static_cast<RealType>(2.2),
      static_cast<RealType>(0.08076),
      static_cast<RealType>(1 - 0.08076),
      tolerance);

   check_skew_normal(
      static_cast<RealType>(5),
      static_cast<RealType>(2),
      static_cast<RealType>(0),
      static_cast<RealType>(7.8),
      static_cast<RealType>(1 - 0.08076),
      static_cast<RealType>(0.08076),
      tolerance);

   check_skew_normal(
      static_cast<RealType>(-3),
      static_cast<RealType>(5),
      static_cast<RealType>(0),
      static_cast<RealType>(-4.5),
      static_cast<RealType>(0.38209),
      static_cast<RealType>(1 - 0.38209),
      tolerance);

   check_skew_normal(
      static_cast<RealType>(-3),
      static_cast<RealType>(5),
      static_cast<RealType>(0),
      static_cast<RealType>(-1.5),
      static_cast<RealType>(1 - 0.38209),
      static_cast<RealType>(0.38209),
      tolerance);

   check_skew_normal(
      static_cast<RealType>(-3),
      static_cast<RealType>(5),
      static_cast<RealType>(0),
      static_cast<RealType>(-8.5),
      static_cast<RealType>(0.13567),
      static_cast<RealType>(1 - 0.13567),
      tolerance);

   check_skew_normal(
      static_cast<RealType>(-3),
      static_cast<RealType>(5),
      static_cast<RealType>(0),
      static_cast<RealType>(2.5),
      static_cast<RealType>(1 - 0.13567),
      static_cast<RealType>(0.13567),
      tolerance);

   // Tests where shape != 0, specific to skew_normal distribution.
   //void check_skew_normal(RealType mean, RealType scale, RealType shape, RealType x, RealType p, RealType q, RealType tol)
      check_skew_normal( // 1st R example.
      static_cast<RealType>(1.1),
      static_cast<RealType>(2.2),
      static_cast<RealType>(-3.3),
      static_cast<RealType>(0.4), // x
      static_cast<RealType>(0.733918618927874), // p == psn
      static_cast<RealType>(1 - 0.733918618927874), // q 
      tolerance);

   // Not sure about these yet.
      //check_skew_normal( // 2nd R example.
      //static_cast<RealType>(1.1),
      //static_cast<RealType>(0.02),
      //static_cast<RealType>(0.03),
      //static_cast<RealType>(1.3), // x
      //static_cast<RealType>(0.01), // p
      //static_cast<RealType>(0.09), // q
      //tolerance);
      //check_skew_normal( // 3nd R example.
      //static_cast<RealType>(10.1),
      //static_cast<RealType>(5.),
      //static_cast<RealType>(-0.03),
      //static_cast<RealType>(-1.3), // x
      //static_cast<RealType>(0.01201290665838824), // p
      //static_cast<RealType>(1. - 0.01201290665838824), // q 0.987987101
      //tolerance);

    // Tests for PDF: we know that the normal peak value is at 1/sqrt(2*pi)
   //
   tolerance = boost::math::tools::epsilon<RealType>() * 5; // 5 eps as a fraction
   BOOST_CHECK_CLOSE_FRACTION(
      pdf(skew_normal_distribution<RealType>(), static_cast<RealType>(0)),
      static_cast<RealType>(0.3989422804014326779399460599343818684759L), // 1/sqrt(2*pi)
      tolerance);
   BOOST_CHECK_CLOSE_FRACTION(
      pdf(skew_normal_distribution<RealType>(3), static_cast<RealType>(3)),
      static_cast<RealType>(0.3989422804014326779399460599343818684759L),
      tolerance);
   BOOST_CHECK_CLOSE_FRACTION(
      pdf(skew_normal_distribution<RealType>(3, 5), static_cast<RealType>(3)),
      static_cast<RealType>(0.3989422804014326779399460599343818684759L / 5),
      tolerance);

   // Shape != 0.
   BOOST_CHECK_CLOSE_FRACTION(
      pdf(skew_normal_distribution<RealType>(3,5,1e-6), static_cast<RealType>(3)),
      static_cast<RealType>(0.3989422804014326779399460599343818684759L / 5),
      tolerance);


   // Checks on mean, variance cumulants etc.
   // Checks on shape ==0

    RealType tol5 = boost::math::tools::epsilon<RealType>() * 5;
    skew_normal_distribution<RealType> dist(8, 3);
    RealType x = static_cast<RealType>(0.125);

    BOOST_MATH_STD_USING // ADL of std math lib names

    // mean:
    BOOST_CHECK_CLOSE(
       mean(dist)
       , static_cast<RealType>(8), tol5);
    // variance:
    BOOST_CHECK_CLOSE(
       variance(dist)
       , static_cast<RealType>(9), tol5);
    // std deviation:
    BOOST_CHECK_CLOSE(
       standard_deviation(dist)
       , static_cast<RealType>(3), tol5);
    // hazard:
    BOOST_CHECK_CLOSE(
       hazard(dist, x)
       , pdf(dist, x) / cdf(complement(dist, x)), tol5);
    // cumulative hazard:
    BOOST_CHECK_CLOSE(
       chf(dist, x)
       , -log(cdf(complement(dist, x))), tol5);
    // coefficient_of_variation:
    BOOST_CHECK_CLOSE(
       coefficient_of_variation(dist)
       , standard_deviation(dist) / mean(dist), tol5);
    // mode: 
    BOOST_CHECK_CLOSE_FRACTION(mode(dist), static_cast<RealType>(8), 0.001f);

    BOOST_CHECK_CLOSE(
       median(dist)
       , static_cast<RealType>(8), tol5);

    // skewness:
    BOOST_CHECK_CLOSE(
       skewness(dist)
       , static_cast<RealType>(0), tol5);
    // kurtosis:
    BOOST_CHECK_CLOSE(
       kurtosis(dist)
       , static_cast<RealType>(3), tol5);
    // kurtosis excess:
    BOOST_CHECK_CLOSE(
       kurtosis_excess(dist)
       , static_cast<RealType>(0), tol5);

    skew_normal_distribution<RealType> norm01(0, 1); // Test default (0, 1)
    BOOST_CHECK_CLOSE(
       mean(norm01),
       static_cast<RealType>(0), 0); // Mean == zero

    skew_normal_distribution<RealType> defsd_norm01(0); // Test default (0, sd = 1)
    BOOST_CHECK_CLOSE(
       mean(defsd_norm01),
       static_cast<RealType>(0), 0); // Mean == zero

    skew_normal_distribution<RealType> def_norm01; // Test default (0, sd = 1)
    BOOST_CHECK_CLOSE(
       mean(def_norm01),
       static_cast<RealType>(0), 0); // Mean == zero

    BOOST_CHECK_CLOSE(
       standard_deviation(def_norm01),
       static_cast<RealType>(1), 0);  // 

    BOOST_CHECK_CLOSE(
       mode(def_norm01),
       static_cast<RealType>(0), 0); // Mode == zero


    // Skew_normal tests with shape != 0.
    {
      // Note these tolerances are expressed as percentages, hence the extra * 100 on the end:
      RealType tol10 = boost::math::tools::epsilon<RealType>() * 10 * 100;
      RealType tol100 = boost::math::tools::epsilon<RealType>() * 100 * 100;

      //skew_normal_distribution<RealType> dist(1.1, 0.02, 0.03);

      BOOST_MATH_STD_USING // ADL of std math lib names.

      // Test values from R = see skew_normal_drv.cpp which included the R code used.
      // Note test values have limited precision.
      if(boost::math::tools::digits<RealType>() <= 64)
      {
        dist = skew_normal_distribution<RealType>(static_cast<RealType>(1.1l), static_cast<RealType>(2.2l), static_cast<RealType>(-3.3l));

        BOOST_CHECK_CLOSE(      // mean:
           mean(dist)
           , static_cast<RealType>(-0.5799089925398568258625490172876619L), tol10 * 2);

        std::cout << std::setprecision(17) << "Variance = " << variance(dist) << std::endl;
         BOOST_CHECK_CLOSE(      // variance: N[variance[skewnormaldistribution[1.1, 2.2, -3.3]], 50]
          variance(dist)
          , static_cast<RealType>(2.0179057767837232633904061072049998357047989154484L), tol10);

        BOOST_CHECK_CLOSE(      // skewness:
           skewness(dist)
           , static_cast<RealType>(-0.709854548171537509192897824663027155L), tol100);
        BOOST_CHECK_CLOSE(      // kurtosis:
           kurtosis(dist)
           , static_cast<RealType>(3.55387526252417906013770535120683805L), tol100);
        BOOST_CHECK_CLOSE(      // kurtosis excess:
           kurtosis_excess(dist)
           , static_cast<RealType>(0.553875262524179060137705351206838143L), tol100);

        BOOST_CHECK_CLOSE(
          pdf(dist, static_cast<RealType>(0.4L)),
          static_cast<RealType>(0.294140110156599539564571034730246656L),
          tol10);

        BOOST_CHECK_CLOSE(
          cdf(dist, static_cast<RealType>(0.4L)),
          static_cast<RealType>(0.733918618927873797632667645226588243L),
          tol100);

        BOOST_CHECK_CLOSE(
          quantile(dist, static_cast<RealType>(0.3L)),
          static_cast<RealType>(-1.18010406808687531441924729956233392L),
          tol100);


      { // mode tests

           dist = skew_normal_distribution<RealType>(static_cast<RealType>(0.l), static_cast<RealType>(1.l), static_cast<RealType>(4.l));

       // cout << "pdf(dist, 0) = " << pdf(dist, 0) <<  ", pdf(dist, 0.45) = " << pdf(dist, 0.45) << endl;
       // BOOST_CHECK_CLOSE(mode(dist), boost::math::constants::root_two<RealType>() / 2, tol5);
        BOOST_CHECK_CLOSE(mode(dist), static_cast<RealType>(0.416972994973888639318345129445233074L), tol100);
      }


      }
      dist = skew_normal_distribution<RealType>(static_cast<RealType>(1.1l), static_cast<RealType>(0.02l), static_cast<RealType>(0.03l));

      BOOST_CHECK_CLOSE(      // mean:
           mean(dist)
           , static_cast<RealType>(1.1004785154529557886162056250600829L), tol10);
      BOOST_CHECK_CLOSE(      // variance:
          variance(dist)
           , static_cast<RealType>(0.000399771022961282516451686289719995601L), tol10);

      BOOST_CHECK_CLOSE(      // skewness:
           skewness(dist)
           , static_cast<RealType>(5.88348112598903597820852388986073439e-006L), tol100);
      BOOST_CHECK_CLOSE(      // kurtosis:
           kurtosis(dist)
           , static_cast<RealType>(3.L + 9.290347581213780023900209941e-008L), tol100);
      BOOST_CHECK_CLOSE(      // kurtosis excess:
           kurtosis_excess(dist)
           , static_cast<RealType>(9.29034758121378002390020993765449518e-008L), tol100);
      dist = skew_normal_distribution<RealType>(static_cast<RealType>(10.1l), static_cast<RealType>(5.l), static_cast<RealType>(-0.03l));
      BOOST_CHECK_CLOSE(      // mean:
           mean(dist)
           , static_cast<RealType>(9.98037113676105284594859373497928476L), tol10);
      BOOST_CHECK_CLOSE(      // variance:
          variance(dist)
           , static_cast<RealType>(24.9856889350801572782303931074997234L), tol10);

      BOOST_CHECK_CLOSE(      // skewness:
           skewness(dist)
           , static_cast<RealType>(-5.88348112598903597820852388986073439e-006L), tol100);
      BOOST_CHECK_CLOSE(      // kurtosis:
           kurtosis(dist)
           , static_cast<RealType>(3.L + 9.290347581213780023900209941e-008L), tol100);
      BOOST_CHECK_CLOSE(      // kurtosis excess:
           kurtosis_excess(dist)
           , static_cast<RealType>(9.29034758121378002390020993765449518e-008L), tol100);
      dist = skew_normal_distribution<RealType>(static_cast<RealType>(-10.1l), static_cast<RealType>(5.l), static_cast<RealType>(30.l));
      BOOST_CHECK_CLOSE(      // mean:
           mean(dist)
           , static_cast<RealType>(-6.11279169674138408531365149047090859L), 2 * tol10);
      BOOST_CHECK_CLOSE(      // variance:
          variance(dist)
          , static_cast<RealType>(9.10216994642554914628242097277880642L), tol10 * 2);

      BOOST_CHECK_CLOSE(      // skewness:
           skewness(dist)
           , static_cast<RealType>(0.990724254436869044244695246354219556L), tol100);
      BOOST_CHECK_CLOSE(      // kurtosis:
           kurtosis(dist)
           , static_cast<RealType>(3.L + 0.8638862008406084244563090239530549L), tol100);
      BOOST_CHECK_CLOSE(      // kurtosis excess:
           kurtosis_excess(dist)
           , static_cast<RealType>(0.863886200840608424456309023953054896L), tol100);

      BOOST_MATH_CHECK_THROW(cdf(skew_normal_distribution<RealType>(0, 0, 0), 0), std::domain_error);
      BOOST_MATH_CHECK_THROW(cdf(skew_normal_distribution<RealType>(0, -1, 0), 0), std::domain_error);
      BOOST_MATH_CHECK_THROW(quantile(skew_normal_distribution<RealType>(0, 1, 0), -1), std::domain_error);
      BOOST_MATH_CHECK_THROW(quantile(skew_normal_distribution<RealType>(0, 1, 0), 2), std::domain_error);
      check_out_of_range<skew_normal_distribution<RealType> >(1, 1, 1);
    }


} // template <class RealType>void test_spots(RealType)

BOOST_AUTO_TEST_CASE( test_main )
{


  using boost::math::skew_normal;
  using boost::math::skew_normal_distribution;

  //int precision = 17; // std::numeric_limits<double::max_digits10;
  double tolfeweps = numeric_limits<double>::epsilon() * 5;
  //double tol6decdigits = numeric_limits<float>::epsilon() * 2;
  // Check that can generate skew_normal distribution using the two convenience methods:
  boost::math::skew_normal w12(1., 2); // Using typedef.
  boost::math::skew_normal_distribution<> w01; // Use default unity values for mean and scale.
  // Note NOT myn01() as the compiler will interpret as a function!

  // Checks on constructors.
  // Default parameters.
  BOOST_CHECK_EQUAL(w01.location(), 0);
  BOOST_CHECK_EQUAL(w01.scale(), 1);
  BOOST_CHECK_EQUAL(w01.shape(), 0);

  skew_normal_distribution<> w23(2., 3); // Using default RealType double.
  BOOST_CHECK_EQUAL(w23.scale(), 3);
  BOOST_CHECK_EQUAL(w23.shape(), 0);

  skew_normal_distribution<> w123(1., 2., 3.); // Using default RealType double.
  BOOST_CHECK_EQUAL(w123.location(), 1.);
  BOOST_CHECK_EQUAL(w123.scale(), 2.);
  BOOST_CHECK_EQUAL(w123.shape(), 3.);

  BOOST_CHECK_CLOSE_FRACTION(mean(w01), static_cast<double>(0), tolfeweps); // Default mean == zero
  BOOST_CHECK_CLOSE_FRACTION(scale(w01), static_cast<double>(1), tolfeweps); // Default scale == unity

  // Basic sanity-check spot values for all floating-point types..
  // (Parameter value, arbitrarily zero, only communicates the floating point type).
  test_spots(0.0F); // Test float. OK at decdigits = 0 tolerance = 0.0001 %
  test_spots(0.0); // Test double. OK at decdigits 7, tolerance = 1e07 %
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
  test_spots(0.0L); // Test long double.
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
  test_spots(boost::math::concepts::real_concept(0.)); // Test real concept.
#endif
#else
  std::cout << "<note>The long double tests have been disabled on this platform "
    "either because the long double overloads of the usual math functions are "
    "not available at all, or because they are too inaccurate for these tests "
    "to pass.</note>" << std::endl;
#endif
  /*      */
  
} // BOOST_AUTO_TEST_CASE( test_main )

/*

Output:


*/



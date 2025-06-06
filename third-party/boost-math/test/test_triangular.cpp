// Copyright Paul Bristow 2006, 2007.
// Copyright John Maddock 2006, 2007.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// test_triangular.cpp

#ifndef SYCL_LANGUAGE_VERSION
#include <pch.hpp>
#endif

#ifdef _MSC_VER
#  pragma warning(disable: 4127) // conditional expression is constant.
#  pragma warning(disable: 4305) // truncation from 'long double' to 'float'
#endif

#include <boost/math/tools/config.hpp>

#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
#include <boost/math/concepts/real_concept.hpp> // for real_concept
#endif

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // Boost.Test
#include <boost/test/tools/floating_point_comparison.hpp>

#include <boost/math/distributions/triangular.hpp>
using boost::math::triangular_distribution;
#include "../include_private/boost/math/tools/test.hpp"
#include <boost/math/special_functions/fpclassify.hpp>
#include "test_out_of_range.hpp"

#include <iostream>
#include <iomanip>
using std::cout;
using std::endl;
using std::scientific;
using std::fixed;
using std::left;
using std::right;
using std::setw;
using std::setprecision;
using std::showpos;
#include <limits>
using std::numeric_limits;

template <class RealType>
void check_triangular(RealType lower, RealType mode, RealType upper, RealType x, RealType p, RealType q, RealType tol)
{
  BOOST_CHECK_CLOSE_FRACTION(
    ::boost::math::cdf(
    triangular_distribution<RealType>(lower, mode, upper),   // distribution.
    x),  // random variable.
    p,    // probability.
    tol);   // tolerance.
  BOOST_CHECK_CLOSE_FRACTION(
    ::boost::math::cdf(
    complement(
    triangular_distribution<RealType>(lower, mode, upper), // distribution.
    x)),    // random variable.
    q,    // probability complement.
    tol);  // tolerance.
  BOOST_CHECK_CLOSE_FRACTION(
    ::boost::math::quantile(
    triangular_distribution<RealType>(lower,mode, upper),  // distribution.
    p),   // probability.
    x,  // random variable.
    tol);  // tolerance.
  BOOST_CHECK_CLOSE_FRACTION(
    ::boost::math::quantile(
    complement(
    triangular_distribution<RealType>(lower, mode, upper),  // distribution.
    q)),     // probability complement.
    x,                                             // random variable.
    tol);  // tolerance.
} // void check_triangular

template <class RealType>
void test_spots(RealType)
{
  // Basic sanity checks:
  //
  // Some test values were generated for the triangular distribution
  // using the online calculator at
  // http://espse.ed.psu.edu/edpsych/faculty/rhale/hale/507Mat/statlets/free/pdist.htm
  //
  // Tolerance is just over 5 epsilon expressed as a fraction:
  RealType tolerance = boost::math::tools::epsilon<RealType>() * 5; // 5 eps as a fraction.
  RealType tol5eps = boost::math::tools::epsilon<RealType>() * 5; // 5 eps as a fraction.

  cout << "Tolerance for type " << typeid(RealType).name()  << " is " << tolerance << "." << endl;

  using namespace std; // for ADL of std::exp;

  // Tests on construction
  // Default should be 0, 0, 1
  BOOST_CHECK_EQUAL(triangular_distribution<RealType>().lower(), -1);
  BOOST_CHECK_EQUAL(triangular_distribution<RealType>().mode(), 0);
  BOOST_CHECK_EQUAL(triangular_distribution<RealType>().upper(), 1);
  BOOST_CHECK_EQUAL(support(triangular_distribution<RealType>()).first, triangular_distribution<RealType>().lower());
  BOOST_CHECK_EQUAL(support(triangular_distribution<RealType>()).second, triangular_distribution<RealType>().upper());

  if (std::numeric_limits<RealType>::has_quiet_NaN == true)
  {
  BOOST_MATH_CHECK_THROW( // duff parameter lower.
    triangular_distribution<RealType>(static_cast<RealType>(std::numeric_limits<RealType>::quiet_NaN()), 0, 0),
    std::domain_error);

  BOOST_MATH_CHECK_THROW( // duff parameter mode.
    triangular_distribution<RealType>(0, static_cast<RealType>(std::numeric_limits<RealType>::quiet_NaN()), 0),
    std::domain_error);

  BOOST_MATH_CHECK_THROW( // duff parameter upper.
    triangular_distribution<RealType>(0, 0, static_cast<RealType>(std::numeric_limits<RealType>::quiet_NaN())),
    std::domain_error);
  } // quiet_NaN tests.

  BOOST_MATH_CHECK_THROW( // duff parameters upper < lower.
    triangular_distribution<RealType>(1, 0, -1),
    std::domain_error);

  BOOST_MATH_CHECK_THROW( // duff parameters upper == lower.
    triangular_distribution<RealType>(0, 0, 0),
    std::domain_error);
  BOOST_MATH_CHECK_THROW( // duff parameters mode < lower.
    triangular_distribution<RealType>(0, -1, 1),
    std::domain_error);

  BOOST_MATH_CHECK_THROW( // duff parameters mode > upper.
    triangular_distribution<RealType>(0, 2, 1),
    std::domain_error);

  // Tests for PDF
  // // triangular_distribution<RealType>() default is (0, 0, 1), mode == lower.
  BOOST_CHECK_CLOSE_FRACTION( // x == lower == mode
    pdf(triangular_distribution<RealType>(0, 0, 1), static_cast<RealType>(0)),
    static_cast<RealType>(2),
    tolerance);

  BOOST_CHECK_CLOSE_FRACTION( // x == upper
    pdf(triangular_distribution<RealType>(0, 0, 1), static_cast<RealType>(1)),
    static_cast<RealType>(0),
    tolerance);

  BOOST_CHECK_CLOSE_FRACTION( // x > upper
    pdf(triangular_distribution<RealType>(0, 0, 1), static_cast<RealType>(-1)),
    static_cast<RealType>(0),
    tolerance);
  BOOST_CHECK_CLOSE_FRACTION( // x < lower
    pdf(triangular_distribution<RealType>(0, 0, 1), static_cast<RealType>(2)),
    static_cast<RealType>(0),
    tolerance);

  BOOST_CHECK_CLOSE_FRACTION( // x < lower
    pdf(triangular_distribution<RealType>(0, 0, 1), static_cast<RealType>(2)),
    static_cast<RealType>(0),
    tolerance);

  // triangular_distribution<RealType>() (0, 1, 1) mode == upper
  BOOST_CHECK_CLOSE_FRACTION( // x == lower
    pdf(triangular_distribution<RealType>(0, 1, 1), static_cast<RealType>(0)),
    static_cast<RealType>(0),
    tolerance);

  BOOST_CHECK_CLOSE_FRACTION( // x == upper
    pdf(triangular_distribution<RealType>(0, 1, 1), static_cast<RealType>(1)),
    static_cast<RealType>(2),
    tolerance);

  BOOST_CHECK_CLOSE_FRACTION( // x > upper
    pdf(triangular_distribution<RealType>(0, 1, 1), static_cast<RealType>(-1)),
    static_cast<RealType>(0),
    tolerance);
  BOOST_CHECK_CLOSE_FRACTION( // x < lower
    pdf(triangular_distribution<RealType>(0, 1, 1), static_cast<RealType>(2)),
    static_cast<RealType>(0),
    tolerance);

  BOOST_CHECK_CLOSE_FRACTION( // x < middle so Wiki says special case pdf = 2 * x
    pdf(triangular_distribution<RealType>(0, 1, 1), static_cast<RealType>(0.25)),
    static_cast<RealType>(0.5),
    tolerance);

  BOOST_CHECK_CLOSE_FRACTION( // x < middle so Wiki says special case cdf = x * x
    cdf(triangular_distribution<RealType>(0, 1, 1), static_cast<RealType>(0.25)),
    static_cast<RealType>(0.25 * 0.25),
    tolerance);

  // triangular_distribution<RealType>() (0, 0.5, 1) mode == middle.
  BOOST_CHECK_CLOSE_FRACTION( // x == lower
    pdf(triangular_distribution<RealType>(0, 0.5, 1), static_cast<RealType>(0)),
    static_cast<RealType>(0),
    tolerance);

  BOOST_CHECK_CLOSE_FRACTION( // x == upper
    pdf(triangular_distribution<RealType>(0, 0.5, 1), static_cast<RealType>(1)),
    static_cast<RealType>(0),
    tolerance);

  BOOST_CHECK_CLOSE_FRACTION( // x > upper
    pdf(triangular_distribution<RealType>(0, 0.5, 1), static_cast<RealType>(-1)),
    static_cast<RealType>(0),
    tolerance);
  BOOST_CHECK_CLOSE_FRACTION( // x < lower
    pdf(triangular_distribution<RealType>(0, 0.5, 1), static_cast<RealType>(2)),
    static_cast<RealType>(0),
    tolerance);

  BOOST_CHECK_CLOSE_FRACTION( // x == mode
    pdf(triangular_distribution<RealType>(0, 0.5, 1), static_cast<RealType>(0.5)),
    static_cast<RealType>(2),
    tolerance);

  BOOST_CHECK_CLOSE_FRACTION( // x == half mode
    pdf(triangular_distribution<RealType>(0, 0.5, 1), static_cast<RealType>(0.25)),
    static_cast<RealType>(1),
    tolerance);
  BOOST_CHECK_CLOSE_FRACTION( // x == half mode
    pdf(triangular_distribution<RealType>(0, 0.5, 1), static_cast<RealType>(0.75)),
    static_cast<RealType>(1),
    tolerance);

  if(std::numeric_limits<RealType>::has_infinity)
  { // BOOST_CHECK tests for infinity using std::numeric_limits<>::infinity()
    // Note that infinity is not implemented for real_concept, so these tests
    // are only done for types, like built-in float, double.. that have infinity.
    // Note that these assume that  BOOST_MATH_OVERFLOW_ERROR_POLICY is NOT throw_on_error.
    // #define BOOST_MATH_OVERFLOW_ERROR_POLICY == throw_on_error would give a throw here.
    // #define BOOST_MATH_DOMAIN_ERROR_POLICY == throw_on_error IS defined, so the throw path
    // of error handling is tested below with BOOST_MATH_CHECK_THROW tests.

    BOOST_MATH_CHECK_THROW( // x == infinity NOT OK.
      pdf(triangular_distribution<RealType>(0, 0, 1), static_cast<RealType>(std::numeric_limits<RealType>::infinity())),
      std::domain_error);

    BOOST_MATH_CHECK_THROW( // x == minus infinity not OK too.
      pdf(triangular_distribution<RealType>(0, 0, 1), static_cast<RealType>(-std::numeric_limits<RealType>::infinity())),
      std::domain_error);
  }
  if(std::numeric_limits<RealType>::has_quiet_NaN)
  { // BOOST_CHECK tests for NaN using std::numeric_limits<>::has_quiet_NaN() - should throw.
    BOOST_MATH_CHECK_THROW(
      pdf(triangular_distribution<RealType>(0, 0, 1), static_cast<RealType>(std::numeric_limits<RealType>::quiet_NaN())),
      std::domain_error);
    BOOST_MATH_CHECK_THROW(
      pdf(triangular_distribution<RealType>(0, 0, 1), static_cast<RealType>(-std::numeric_limits<RealType>::quiet_NaN())),
      std::domain_error);
  } // test for x = NaN using std::numeric_limits<>::quiet_NaN()

  // cdf
  BOOST_CHECK_EQUAL( // x < lower
    cdf(triangular_distribution<RealType>(0, 1, 1), static_cast<RealType>(-1)),
    static_cast<RealType>(0) );
  BOOST_CHECK_CLOSE_FRACTION( // x == lower
    cdf(triangular_distribution<RealType>(0, 1, 1), static_cast<RealType>(0)),
    static_cast<RealType>(0),
    tolerance);
  BOOST_CHECK_CLOSE_FRACTION( // x == upper
    cdf(triangular_distribution<RealType>(0, 1, 1), static_cast<RealType>(1)),
    static_cast<RealType>(1),
    tolerance);
   BOOST_CHECK_EQUAL( // x > upper
    cdf(triangular_distribution<RealType>(0, 1, 1), static_cast<RealType>(2)),
    static_cast<RealType>(1));

  BOOST_CHECK_CLOSE_FRACTION( // x == mode
    cdf(triangular_distribution<RealType>(-1, 0, 1), static_cast<RealType>(0)),
    //static_cast<RealType>((mode - lower) / (upper - lower)),
    static_cast<RealType>(0.5),    // (0 --1) / (1 -- 1) = 0.5
    tolerance);
  BOOST_CHECK_CLOSE_FRACTION(
    cdf(triangular_distribution<RealType>(0, 1, 1), static_cast<RealType>(0.9L)),
    static_cast<RealType>(0.81L),
    tolerance);

  BOOST_CHECK_CLOSE_FRACTION(
    cdf(triangular_distribution<RealType>(-1, 0, 1), static_cast<RealType>(-1)),
    static_cast<RealType>(0),
    tolerance);
  BOOST_CHECK_CLOSE_FRACTION(
    cdf(triangular_distribution<RealType>(-1, 0, 1), static_cast<RealType>(-0.5L)),
    static_cast<RealType>(0.125L),
    tolerance);
  BOOST_CHECK_CLOSE_FRACTION(
    cdf(triangular_distribution<RealType>(-1, 0, 1), static_cast<RealType>(0)),
    static_cast<RealType>(0.5),
    tolerance);
  BOOST_CHECK_CLOSE_FRACTION(
    cdf(triangular_distribution<RealType>(-1, 0, 1), static_cast<RealType>(+0.5L)),
    static_cast<RealType>(0.875L),
    tolerance);
  BOOST_CHECK_CLOSE_FRACTION(
    cdf(triangular_distribution<RealType>(-1, 0, 1), static_cast<RealType>(1)),
    static_cast<RealType>(1),
    tolerance);

   // cdf complement
  BOOST_CHECK_EQUAL( // x < lower
    cdf(complement(triangular_distribution<RealType>(0, 0, 1), static_cast<RealType>(-1))),
    static_cast<RealType>(1));
  BOOST_CHECK_EQUAL( // x == lower
    cdf(complement(triangular_distribution<RealType>(0, 0, 1), static_cast<RealType>(0))),
    static_cast<RealType>(1));

  BOOST_CHECK_EQUAL( // x == mode
    cdf(complement(triangular_distribution<RealType>(-1, 0, 1), static_cast<RealType>(0))),
    static_cast<RealType>(0.5));

  BOOST_CHECK_EQUAL( // x == mode
    cdf(complement(triangular_distribution<RealType>(0, 0, 1), static_cast<RealType>(0))),
    static_cast<RealType>(1));
  BOOST_CHECK_EQUAL( // x == mode
    cdf(complement(triangular_distribution<RealType>(0, 1, 1), static_cast<RealType>(1))),
    static_cast<RealType>(0));

  BOOST_CHECK_EQUAL( // x > upper
    cdf(complement(triangular_distribution<RealType>(0, 0, 1), static_cast<RealType>(2))),
    static_cast<RealType>(0));
  BOOST_CHECK_EQUAL( // x == upper
    cdf(complement(triangular_distribution<RealType>(0, 0, 1), static_cast<RealType>(1))),
    static_cast<RealType>(0));

  BOOST_CHECK_CLOSE_FRACTION( // x = -0.5
    cdf(complement(triangular_distribution<RealType>(-1, 0, 1), static_cast<RealType>(-0.5))),
    static_cast<RealType>(0.875L),
    tolerance);

  BOOST_CHECK_CLOSE_FRACTION( // x = +0.5
    cdf(complement(triangular_distribution<RealType>(-1, 0, 1), static_cast<RealType>(0.5))),
    static_cast<RealType>(0.125),
    tolerance);

  triangular_distribution<RealType> triang; // Using typedef == triangular_distribution<double> tristd;
  triangular_distribution<RealType> tristd(0, 0.5, 1); // 'Standard' triangular distribution.

  BOOST_CHECK_CLOSE_FRACTION( // median of Standard triangular is sqrt(mode/2) if c > 1/2 else 1 - sqrt((1-c)/2)
    median(tristd),
    static_cast<RealType>(0.5),
    tolerance);
  triangular_distribution<RealType> tri011(0, 1, 1); // Using default RealType double.
  triangular_distribution<RealType> tri0q1(0, 0.25, 1); // mode is near bottom.
  triangular_distribution<RealType> tri0h1(0, 0.5, 1); // Equilateral triangle - mode is the middle.
  triangular_distribution<RealType> trim12(-1, -0.5, 2); // mode is negative.

  BOOST_CHECK_CLOSE_FRACTION(cdf(tri0q1, 0.02L), static_cast<RealType>(0.0016L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cdf(tri0q1, 0.5L), static_cast<RealType>(0.66666666666666666666666666666666666666666666667L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cdf(tri0q1, 0.98L), static_cast<RealType>(0.9994666666666666666666666666666666666666666666L), tolerance);

  // quantile
  BOOST_CHECK_CLOSE_FRACTION(quantile(tri0q1, static_cast<RealType>(0.0016L)), static_cast<RealType>(0.02L), tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(quantile(tri0q1, static_cast<RealType>(0.66666666666666666666666666666666666666666666667L)), static_cast<RealType>(0.5), tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(quantile(complement(tri0q1, static_cast<RealType>(0.3333333333333333333333333333333333333333333333333L))), static_cast<RealType>(0.5), tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(quantile(tri0q1, static_cast<RealType>(0.999466666666666666666666666666666666666666666666666L)), static_cast<RealType>(98) / 100, 10 * tol5eps);

  BOOST_CHECK_CLOSE_FRACTION(pdf(trim12, 0), static_cast<RealType>(0.533333333333333333333333333333333333333333333L), tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(cdf(trim12, 0), static_cast<RealType>(0.466666666666666666666666666666666666666666667L), tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(cdf(complement(trim12, 0)), static_cast<RealType>(1 - 0.466666666666666666666666666666666666666666667L), tol5eps);

  BOOST_CHECK_CLOSE_FRACTION(quantile(complement(tri0q1, static_cast<RealType>(1 - 0.999466666666666666666666666666666666666666666666L))), static_cast<RealType>(0.98L), 10 * tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(quantile(complement(tri0h1, static_cast<RealType>(1))), static_cast<RealType>(0), tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(quantile(complement(tri0h1, static_cast<RealType>(0.5))), static_cast<RealType>(0.5), tol5eps); // OK
  BOOST_CHECK_CLOSE_FRACTION(quantile(complement(tri0h1, static_cast<RealType>(1 - 0.02L))), static_cast<RealType>(0.1L), tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(quantile(complement(tri0h1, static_cast<RealType>(1 - 0.98L))), static_cast<RealType>(0.9L), tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(quantile(complement(tri0h1, 0)), static_cast<RealType>(1), tol5eps);

  RealType xs [] = {0, 0.01L, 0.02L, 0.05L, 0.1L, 0.2L, 0.3L, 0.4L, 0.5L, 0.6L, 0.7L, 0.8L, 0.9L, 0.95L, 0.98L, 0.99L, 1};

  const triangular_distribution<RealType>& distr = triang;
  BOOST_CHECK_CLOSE_FRACTION(quantile(complement(distr, 1.)), static_cast<RealType>(-1), tol5eps);
  const triangular_distribution<RealType>* distp = &triang;
  BOOST_CHECK_CLOSE_FRACTION(quantile(complement(*distp, 1.)), static_cast<RealType>(-1), tol5eps);

  const triangular_distribution<RealType>* dists [] = {&tristd, &tri011, &tri0q1, &tri0h1, &trim12};
  BOOST_CHECK_CLOSE_FRACTION(quantile(complement(*dists[1], 1.)), static_cast<RealType>(0), tol5eps);

   for (int i = 0; i < 5; i++)
  {
    const triangular_distribution<RealType>* const dist = dists[i];
    // cout << "Distribution " << i << endl;
    BOOST_CHECK_CLOSE_FRACTION(quantile(*dists[i], 0.5L), quantile(complement(*dist, 0.5L)),  tol5eps);
    BOOST_CHECK_CLOSE_FRACTION(quantile(*dists[i], 0.98L), quantile(complement(*dist, 1.L - 0.98L)),tol5eps);
    BOOST_CHECK_CLOSE_FRACTION(quantile(*dists[i], 0.98L), quantile(complement(*dist, 1.L - 0.98L)),tol5eps);
  } // for i

   // quantile complement
  for (int i = 0; i < 5; i++)
  {
    const triangular_distribution<RealType>* const dist = dists[i];
    //cout << "Distribution " << i << endl;
    BOOST_CHECK_EQUAL(quantile(complement(*dists[i], 1.)), quantile(*dists[i], 0.));
    for (unsigned j = 0; j < sizeof(xs) /sizeof(RealType); j++)
    {
      RealType x = xs[j];
      BOOST_CHECK_CLOSE_FRACTION(quantile(*dists[i], x), quantile(complement(*dist, 1 - x)),  tol5eps);
    } // for j
  } // for i


  check_triangular(
    static_cast<RealType>(0),       // lower
    static_cast<RealType>(0.5),     // mode
    static_cast<RealType>(1),       // upper
    static_cast<RealType>(0.5),     // x
    static_cast<RealType>(0.5),     // p
    static_cast<RealType>(1 - 0.5), // q
    tolerance);

  // Some Not-standard triangular tests.
  check_triangular(
    static_cast<RealType>(-1),    // lower
    static_cast<RealType>(0),     // mode
    static_cast<RealType>(1),     // upper
    static_cast<RealType>(0),     // x
    static_cast<RealType>(0.5),   // p
    static_cast<RealType>(1 - 0.5), // q = 1 - p
    tolerance);

  check_triangular(
    static_cast<RealType>(1),       // lower
    static_cast<RealType>(1),       // mode
    static_cast<RealType>(3),       // upper
    static_cast<RealType>(2),    // x
    static_cast<RealType>(0.75),     // p
    static_cast<RealType>(1 - 0.75), // q = 1 - p
    tolerance);

  check_triangular(
    static_cast<RealType>(-1),    // lower
    static_cast<RealType>(1),       // mode
    static_cast<RealType>(2),     // upper
    static_cast<RealType>(1),     // x
    static_cast<RealType>(0.66666666666666666666666666666666666666666667L),   // p
    static_cast<RealType>(0.33333333333333333333333333333333333333333333L), // q = 1 - p
    tolerance);
  tolerance = (std::max)(
    boost::math::tools::epsilon<RealType>(),
    static_cast<RealType>(boost::math::tools::epsilon<double>())) * 10; // 10 eps as a fraction.
  cout << "Tolerance (as fraction) for type " << typeid(RealType).name()  << " is " << tolerance << "." << endl;
  
    triangular_distribution<RealType> tridef; // (-1, 0, 1) // Default distribution.
    RealType x = static_cast<RealType>(0.5);
    using namespace std; // ADL of std names.
    // mean:
    BOOST_CHECK_CLOSE_FRACTION(
      mean(tridef), static_cast<RealType>(0), tolerance);
    // variance:
    BOOST_CHECK_CLOSE_FRACTION(
      variance(tridef), static_cast<RealType>(0.16666666666666666666666666666666666666666667L), tolerance);
    // was 0.0833333333333333333333333333333333333333333L

    // std deviation:
    BOOST_CHECK_CLOSE_FRACTION(
      standard_deviation(tridef), sqrt(variance(tridef)), tolerance);
    // hazard:
    BOOST_CHECK_CLOSE_FRACTION(
      hazard(tridef, x), pdf(tridef, x) / cdf(complement(tridef, x)), tolerance);
    // cumulative hazard:
    BOOST_CHECK_CLOSE_FRACTION(
      chf(tridef, x), -log(cdf(complement(tridef, x))), tolerance);
    // coefficient_of_variation:
    if (mean(tridef) != 0)
    {
      BOOST_CHECK_CLOSE_FRACTION(
        coefficient_of_variation(tridef), standard_deviation(tridef) / mean(tridef), tolerance);
    }
    // mode:
    BOOST_CHECK_CLOSE_FRACTION(
      mode(tridef), static_cast<RealType>(0), tolerance);
    // skewness:
    // On device the result does not get flushed exactly to zero so the eps difference is by default huge
    #ifndef BOOST_MATH_HAS_GPU_SUPPORT
    BOOST_CHECK_CLOSE_FRACTION(
      median(tridef), static_cast<RealType>(0), tolerance);
    #endif
    // https://reference.wolfram.com/language/ref/Skewness.html  skewness{-1, 0, +1} = 0
    // skewness[triangulardistribution{-1, 0, +1}] does not compute a result.
    // skewness[triangulardistribution{0, +1}] result == 0
    // skewness[normaldistribution{0,1}] result == 0

    BOOST_CHECK_EQUAL(
      skewness(tridef), static_cast<RealType>(0));
    // kurtosis:
    BOOST_CHECK_CLOSE_FRACTION(
      kurtosis_excess(tridef), kurtosis(tridef) - static_cast<RealType>(3L), tolerance);
    // kurtosis excess = kurtosis - 3;
    BOOST_CHECK_CLOSE_FRACTION(
      kurtosis_excess(tridef), static_cast<RealType>(-0.6), tolerance); // Constant value of -3/5 for all distributions.

    BOOST_CHECK_CLOSE_FRACTION(
      entropy(tridef), static_cast<RealType>(1)/2, tolerance);

  {
    triangular_distribution<RealType> tri01(0, 1, 1); //  Asymmetric 0, 1, 1 distribution.
    RealType x = static_cast<RealType>(0.5);
    using namespace std; // ADL of std names.
                         // mean:
    BOOST_CHECK_CLOSE_FRACTION(
      mean(tri01), static_cast<RealType>(0.66666666666666666666666666666666666666666666666667L), tolerance);
    // variance: N[variance[triangulardistribution{0, 1}, 1], 50]
    BOOST_CHECK_CLOSE_FRACTION(
      variance(tri01), static_cast<RealType>(0.055555555555555555555555555555555555555555555555556L), tolerance);
    // std deviation:
    BOOST_CHECK_CLOSE_FRACTION(
      standard_deviation(tri01), sqrt(variance(tri01)), tolerance);
    // hazard:
    BOOST_CHECK_CLOSE_FRACTION(
      hazard(tri01, x), pdf(tri01, x) / cdf(complement(tri01, x)), tolerance);
    // cumulative hazard:
    BOOST_CHECK_CLOSE_FRACTION(
      chf(tri01, x), -log(cdf(complement(tri01, x))), tolerance);
    // coefficient_of_variation:
    if (mean(tri01) != 0)
    {
      BOOST_CHECK_CLOSE_FRACTION(
        coefficient_of_variation(tri01), standard_deviation(tri01) / mean(tri01), tolerance);
    }
    // mode:
    BOOST_CHECK_CLOSE_FRACTION(
      mode(tri01), static_cast<RealType>(1), tolerance);
    // skewness:
    BOOST_CHECK_CLOSE_FRACTION(
      median(tri01), static_cast<RealType>(0.70710678118654752440084436210484903928483593768847L), tolerance);

    // https://reference.wolfram.com/language/ref/Skewness.html
    // N[skewness[triangulardistribution{0, 1}, 1], 50]
    BOOST_CHECK_CLOSE_FRACTION(
      skewness(tri01), static_cast<RealType>(-0.56568542494923801952067548968387923142786875015078L), tolerance);
    // kurtosis:
    BOOST_CHECK_CLOSE_FRACTION(
      kurtosis_excess(tri01), kurtosis(tri01) - static_cast<RealType>(3L), tolerance);
    // kurtosis excess = kurtosis - 3;
    BOOST_CHECK_CLOSE_FRACTION(
      kurtosis_excess(tri01), static_cast<RealType>(-0.6), tolerance); // Constant value of -3/5 for all distributions.
  } // tri01 tests

  if(std::numeric_limits<RealType>::has_infinity)
  { // BOOST_CHECK tests for infinity using std::numeric_limits<>::infinity()
    // Note that infinity is not implemented for real_concept, so these tests
    // are only done for types, like built-in float, double.. that have infinity.
    // Note that these assume that BOOST_MATH_OVERFLOW_ERROR_POLICY is NOT throw_on_error.
    // #define BOOST_MATH_OVERFLOW_ERROR_POLICY == throw_on_error would give a throw here.
    // #define BOOST_MATH_DOMAIN_ERROR_POLICY == throw_on_error IS defined, so the throw path
    // of error handling is tested below with BOOST_MATH_CHECK_THROW tests.

    using boost::math::policies::policy;
    using boost::math::policies::domain_error;
    using boost::math::policies::ignore_error;

    // Define a (bad?) policy to ignore domain errors ('bad' arguments):
    typedef policy<domain_error<ignore_error> > inf_policy; // domain error returns infinity.
    triangular_distribution<RealType, inf_policy> tridef_inf(-1, 0., 1);
    // But can't use BOOST_CHECK_EQUAL(?, quiet_NaN)
    using boost::math::isnan;
    BOOST_CHECK((isnan)(pdf(tridef_inf, std::numeric_limits<RealType>::infinity())));
  } // test for infinity using std::numeric_limits<>::infinity()
  else
  { // real_concept case, does has_infinfity == false, so can't check it throws.
    // cout << std::numeric_limits<RealType>::infinity() << ' '
    // << (boost::math::fpclassify)(std::numeric_limits<RealType>::infinity()) << endl;
    // value of std::numeric_limits<RealType>::infinity() is zero, so FPclassify is zero,
    // so (boost::math::isfinite)(std::numeric_limits<RealType>::infinity()) does not detect infinity.
    // so these tests would never throw.
    //BOOST_MATH_CHECK_THROW(pdf(tridef, std::numeric_limits<RealType>::infinity()),  std::domain_error);
    //BOOST_MATH_CHECK_THROW(pdf(tridef, std::numeric_limits<RealType>::quiet_NaN()),  std::domain_error);
    // BOOST_MATH_CHECK_THROW(pdf(tridef, boost::math::tools::max_value<RealType>() * 2),  std::domain_error); // Doesn't throw.
    BOOST_CHECK_EQUAL(pdf(tridef, boost::math::tools::max_value<RealType>()), 0);
  }
  // Special cases:
  BOOST_CHECK(pdf(tridef, -1) == 0);
  BOOST_CHECK(pdf(tridef, 1) == 0);
  BOOST_CHECK(cdf(tridef, 0) == 0.5);
  BOOST_CHECK(pdf(tridef, 1) == 0);
  BOOST_CHECK(cdf(tridef, 1) == 1);
  BOOST_CHECK(cdf(complement(tridef, -1)) == 1);
  BOOST_CHECK(cdf(complement(tridef, 1)) == 0);
  BOOST_CHECK(quantile(tridef, 1) == 1);
  BOOST_CHECK(quantile(complement(tridef, 1)) == -1);

  BOOST_CHECK_EQUAL(support(trim12).first, trim12.lower());
  BOOST_CHECK_EQUAL(support(trim12).second, trim12.upper());

  // Error checks:
  if(std::numeric_limits<RealType>::has_quiet_NaN)
  { // BOOST_CHECK tests for quiet_NaN (not for real_concept, for example - see notes above).
    BOOST_MATH_CHECK_THROW(triangular_distribution<RealType>(0, std::numeric_limits<RealType>::quiet_NaN()), std::domain_error);
    BOOST_MATH_CHECK_THROW(triangular_distribution<RealType>(0, -std::numeric_limits<RealType>::quiet_NaN()), std::domain_error);
  }
  BOOST_MATH_CHECK_THROW(triangular_distribution<RealType>(1, 0), std::domain_error); // lower > upper!

  check_out_of_range<triangular_distribution<RealType> >(-1, 0, 1);
} // template <class RealType>void test_spots(RealType)

BOOST_AUTO_TEST_CASE( test_main )
{
  //  double toleps = std::numeric_limits<double>::epsilon(); // 5 eps as a fraction.
  double tol5eps = std::numeric_limits<double>::epsilon() * 5; // 5 eps as a fraction.
  // double tol50eps = std::numeric_limits<double>::epsilon() * 50; // 50 eps as a fraction.
  double tol500eps = std::numeric_limits<double>::epsilon() * 500; // 500 eps as a fraction.

  // Check that can construct triangular distribution using the two convenience methods:
  using namespace boost::math;
  triangular triang; // Using typedef
  // == triangular_distribution<double> triang;

  BOOST_CHECK_EQUAL(triang.lower(), -1); // Check default.
  BOOST_CHECK_EQUAL(triang.mode(), 0);
  BOOST_CHECK_EQUAL(triang.upper(), 1);

  triangular tristd (0, 0.5, 1); // Using typedef

  BOOST_CHECK_EQUAL(tristd.lower(), 0);
  BOOST_CHECK_EQUAL(tristd.mode(), 0.5);
  BOOST_CHECK_EQUAL(tristd.upper(), 1);

  //cout << "X range from " << range(tristd).first << " to " << range(tristd).second << endl;
  //cout << "Supported from "<< support(tristd).first << ' ' << support(tristd).second << endl;

  BOOST_CHECK_EQUAL(support(tristd).first, tristd.lower());
  BOOST_CHECK_EQUAL(support(tristd).second, tristd.upper());

  triangular_distribution<> tri011(0, 1, 1); // Using default RealType double.
  // mode is upper
  BOOST_CHECK_EQUAL(tri011.lower(), 0); // Check defaults again.
  BOOST_CHECK_EQUAL(tri011.mode(), 1); // Check defaults again.
  BOOST_CHECK_EQUAL(tri011.upper(), 1);
  BOOST_CHECK_EQUAL(mode(tri011), 1);

  BOOST_CHECK_EQUAL(pdf(tri011, 0), 0);
  BOOST_CHECK_EQUAL(pdf(tri011, 0.1), 0.2);
  BOOST_CHECK_EQUAL(pdf(tri011, 0.5), 1);
  BOOST_CHECK_EQUAL(pdf(tri011, 0.9), 1.8);
  BOOST_CHECK_EQUAL(pdf(tri011, 1), 2);

  BOOST_CHECK_EQUAL(cdf(tri011, 0), 0);
  BOOST_CHECK_CLOSE_FRACTION(cdf(tri011, 0.1), 0.01, tol5eps);
  BOOST_CHECK_EQUAL(cdf(tri011, 0.5), 0.25);
  BOOST_CHECK_EQUAL(cdf(tri011, 0.9), 0.81);
  BOOST_CHECK_EQUAL(cdf(tri011, 1), 1);
  BOOST_CHECK_EQUAL(cdf(tri011, 9), 1);
  BOOST_CHECK_EQUAL(mean(tri011), 0.666666666666666666666666666666666666666666666666667);
  BOOST_CHECK_EQUAL(variance(tri011), 1./18.);

  triangular tri0h1(0, 0.5, 1); // Equilateral triangle - mode is the middle.
  BOOST_CHECK_EQUAL(tri0h1.lower(), 0);
  BOOST_CHECK_EQUAL(tri0h1.mode(), 0.5);
  BOOST_CHECK_EQUAL(tri0h1.upper(), 1);
  BOOST_CHECK_EQUAL(mean(tri0h1), 0.5);
  BOOST_CHECK_EQUAL(mode(tri0h1), 0.5);
  BOOST_CHECK_EQUAL(pdf(tri0h1, -1), 0);
  BOOST_CHECK_EQUAL(cdf(tri0h1, -1), 0);
  BOOST_CHECK_EQUAL(pdf(tri0h1, 1), 0);
  BOOST_CHECK_EQUAL(pdf(tri0h1, 999), 0);
  BOOST_CHECK_EQUAL(cdf(tri0h1, 999), 1);
  BOOST_CHECK_EQUAL(cdf(tri0h1, 1), 1);
  BOOST_CHECK_CLOSE_FRACTION(cdf(tri0h1, 0.1), 0.02, tol5eps);
  BOOST_CHECK_EQUAL(cdf(tri0h1, 0.5), 0.5);
  BOOST_CHECK_CLOSE_FRACTION(cdf(tri0h1, 0.9), 0.98, tol5eps);

  BOOST_CHECK_CLOSE_FRACTION(quantile(tri0h1, 0.), 0., tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(quantile(tri0h1, 0.02), 0.1, tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(quantile(tri0h1, 0.5), 0.5, tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(quantile(tri0h1, 0.98), 0.9, tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(quantile(tri0h1, 1.), 1., tol5eps);

  triangular tri0q1(0, 0.25, 1); // mode is near bottom.
  BOOST_CHECK_CLOSE_FRACTION(cdf(tri0q1, 0.02), 0.0016, tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(cdf(tri0q1, 0.5), 0.66666666666666666666666666666666666666666666667, tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(cdf(tri0q1, 0.98), 0.99946666666666661, tol5eps);

  BOOST_CHECK_CLOSE_FRACTION(quantile(tri0q1, 0.0016), 0.02, tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(quantile(tri0q1, 0.66666666666666666666666666666666666666666666667), 0.5, tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(quantile(complement(tri0q1, 0.3333333333333333333333333333333333333333333333333)), 0.5, tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(quantile(tri0q1, 0.99946666666666661), 0.98, 10 * tol5eps);

  triangular trim12(-1, -0.5, 2); // mode is negative.
  BOOST_CHECK_CLOSE_FRACTION(pdf(trim12, 0), 0.533333333333333333333333333333333333333333333, tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(cdf(trim12, 0), 0.466666666666666666666666666666666666666666667, tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(cdf(complement(trim12, 0)), 1 - 0.466666666666666666666666666666666666666666667, tol5eps);

  BOOST_CHECK_CLOSE_FRACTION(quantile(complement(tri0q1, 1 - 0.99946666666666661)), 0.98, 10 * tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(quantile(complement(tri0h1, 1.)), 0., tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(quantile(complement(tri0h1, 0.5)), 0.5, tol5eps); // OK
  BOOST_CHECK_CLOSE_FRACTION(quantile(complement(tri0h1, 1. - 0.02)), 0.1, tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(quantile(complement(tri0h1, 1. - 0.98)), 0.9, tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(quantile(complement(tri0h1, 0)), 1., tol5eps);

  double xs [] = {0., 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.};

  const triangular_distribution<double>& distr = tristd;
  BOOST_CHECK_CLOSE_FRACTION(quantile(complement(distr, 1.)), 0., tol5eps);
  const triangular_distribution<double>* distp = &tristd;
  BOOST_CHECK_CLOSE_FRACTION(quantile(complement(*distp, 1.)), 0., tol5eps);

  const triangular_distribution<double>* dists [] = {&tristd, &tri011, &tri0q1, &tri0h1, &trim12};
  BOOST_CHECK_CLOSE_FRACTION(quantile(complement(*dists[1], 1.)), 0., tol5eps);

  for (int i = 0; i < 5; i++)
  {
    const triangular_distribution<double>* const dist = dists[i];
    cout << "Distribution " << i << endl;
    BOOST_CHECK_EQUAL(quantile(complement(*dists[i], 1.)), quantile(*dists[i], 0.));
    BOOST_CHECK_CLOSE_FRACTION(quantile(*dists[i], 0.5), quantile(complement(*dist, 0.5)),  tol5eps); // OK
    BOOST_CHECK_CLOSE_FRACTION(quantile(*dists[i], 0.98), quantile(complement(*dist, 1. - 0.98)),tol5eps);
    // cout << setprecision(17) <<  median(*dist) << endl;
  }

  cout << showpos << setprecision(2) << endl;

  //triangular_distribution<double>& dist = trim12;
  for (unsigned i = 0; i < sizeof(xs) /sizeof(double); i++)
  {
    double x = xs[i] * (trim12.upper() - trim12.lower()) + trim12.lower();
    double dx = cdf(trim12, x);
    double cx = cdf(complement(trim12, x));
    //cout << fixed << showpos << setprecision(3)
    //  << xs[i] << ", " << x << ",  " << pdf(trim12, x) << ",  " << dx << ",  " << cx << ",, " ;

    BOOST_CHECK_CLOSE_FRACTION(cx, 1 - dx, tol500eps); // cx == 1 - dx

    // << setprecision(2) << scientific << cr - x << ", " // difference x - quan(cdf)
    // << setprecision(3) << fixed
    // << quantile(trim12, dx) << ", "
    // << quantile(complement(trim12, 1 - dx)) << ", "
    // << quantile(complement(trim12, cx)) << ", "
    // << endl;
    BOOST_CHECK_CLOSE_FRACTION(quantile(trim12, dx), quantile(complement(trim12, 1 - dx)), tol500eps);
  }
  cout << endl;
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

Autorun "i:\boost-06-05-03-1300\libs\math\test\Math_test\debug\test_triangular.exe"
Running 1 test case...
Distribution 0
Distribution 1
Distribution 2
Distribution 3
Distribution 4
Tolerance for type float is 5.96046e-007.
Tolerance for type double is 1.11022e-015.
Tolerance for type long double is 1.11022e-015.
Tolerance for type class boost::math::concepts::real_concept is 1.11022e-015.
*** No errors detected



*/





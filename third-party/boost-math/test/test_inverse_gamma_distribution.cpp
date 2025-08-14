// test_inverse_gamma.cpp

// Copyright Paul A. Bristow 2010.
// Copyright John Maddock 2010.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef _MSC_VER
#  pragma warning (disable : 4224) // nonstandard extension used : formal parameter 'type' was previously defined as a type
// in Boost.test and lexical_cast
#  pragma warning (disable : 4310) // cast truncates constant value
#endif

#include <boost/math/tools/config.hpp>
#include "../include_private/boost/math/tools/test.hpp"

#ifndef BOOST_MATH_HAS_GPU_SUPPORT
#include <boost/math/concepts/real_concept.hpp> // for real_concept
using ::boost::math::concepts::real_concept;
#endif

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // for test_main
#include <boost/test/tools/floating_point_comparison.hpp> // for BOOST_CHECK_CLOSE_FRACTION
#include "test_out_of_range.hpp"

#include <boost/math/distributions/inverse_gamma.hpp> // for inverse_gamma_distribution
using boost::math::inverse_gamma_distribution;
using  ::boost::math::inverse_gamma;
//  using  ::boost::math::cdf;
//  using  ::boost::math::pdf;

#include <boost/math/special_functions/gamma.hpp> 
using boost::math::tgamma; // for naive pdf.

#include <iostream>
using std::cout;
using std::endl;
#include <limits>
using std::numeric_limits;

template <class RealType>
RealType naive_pdf(RealType shape, RealType scale, RealType x)
{ // Formula from Wikipedia
   using namespace std; // For ADL of std functions.
   using boost::math::tgamma;
   RealType result = (pow(scale, shape) * pow(x, (-shape -1)) * exp(-scale/x) ) / tgamma(shape);
   return result;
}

// Test using a spot value from some other reference source,
// in this case test values from output from R provided by Thomas Mang.

template <class RealType>
void test_spot(
     RealType shape, // shape,
     RealType scale, // scale,
     RealType x, // random variate x,
     RealType pd, // expected pdf,
     RealType P, // expected CDF,
     RealType Q, // expected complement of CDF,
     RealType tol) // test tolerance.
{
   boost::math::inverse_gamma_distribution<RealType> dist(shape, scale);

   BOOST_CHECK_CLOSE_FRACTION
      ( // Compare to expected PDF.
      pdf(dist, x), // calculated.
      pd, // expected
      tol);

    BOOST_CHECK_CLOSE_FRACTION( // Compare to naive formula (might be less accurate).
      pdf(dist, x), naive_pdf(dist.shape(), dist.scale(), x), tol);

    BOOST_CHECK_CLOSE_FRACTION( // Compare direct logpdf to naive log(pdf())
      logpdf(dist, x), log(pdf(dist,x)), tol);

   BOOST_CHECK_CLOSE_FRACTION( // Compare to expected CDF.
      cdf(dist, x), P, tol);

   if((P < 0.999) && (Q < 0.999))
   {  // We can only check this if P is not too close to 1,
      // so that we can guarantee Q is accurate:
      BOOST_CHECK_CLOSE_FRACTION(
        cdf(complement(dist, x)), Q, tol);
      BOOST_CHECK_CLOSE_FRACTION(
        quantile(dist, P), x, tol); // quantile(pdf) = x
      BOOST_CHECK_CLOSE_FRACTION(
        quantile(complement(dist, Q)), x, tol);
   }
} // test_spot

// Test using a spot value from some other reference source.

template <class RealType> // Any floating-point type RealType.
void test_spots(RealType)
{
  // Basic sanity checks, test data is to six decimal places only
  // so set tolerance to 0.000001 expressed as a percentage = 0.0001%.

  RealType tolerance = 0.000001f; // as fraction.
  cout << "Tolerance = " << tolerance * 100 << "%." << endl;

// This test values from output from R provided by Thomas Mang.
  test_spot(static_cast<RealType>(2), static_cast<RealType>(1), // shape, scale
  static_cast<RealType>(2.L), // x
  static_cast<RealType>(0.075816332464079136L), // pdf
  static_cast<RealType>(0.90979598956895047L), // cdf
  static_cast<RealType>(1 - 0.90979598956895047L), // cdf complement
  tolerance  // tol
  );

  test_spot(static_cast<RealType>(1.593), static_cast<RealType>( 0.5), // shape, scale
  static_cast<RealType>( 0.5), // x
  static_cast<RealType>(0.82415241749687074L), // pdf
  static_cast<RealType>(0.60648042700409865L), // cdf
  static_cast<RealType>(1 - 0.60648042700409865L), // cdf complement
  tolerance  // tol
  );

  test_spot(static_cast<RealType>(13.319), static_cast<RealType>(0.5), // shape, scale
  static_cast<RealType>(0.5), // x
  static_cast<RealType>(0.00000000068343206235379223), // pdf
  static_cast<RealType>(0.99999999997242739L), // cdf
  static_cast<RealType>(1 - 0.99999999997242739L), // cdf complement
  tolerance  // tol
  );

  test_spot(static_cast<RealType>(1.593), static_cast<RealType>(1), // shape, scale
  static_cast<RealType>(1.977), // x
  static_cast<RealType>(0.11535946773398653L), // pdf
  static_cast<RealType>(0.82449794420341549L), // cdf
  static_cast<RealType>(1 - 0.82449794420341549L), // cdf complement
  tolerance  // tol
  );
  
  test_spot(static_cast<RealType>(6.666), static_cast<RealType>(1.411), // shape, scale
  static_cast<RealType>(5), // x
  static_cast<RealType>(0.000000084415758206386872), // pdf
  static_cast<RealType>(0.99999993427280998L), // cdf
  static_cast<RealType>(1 - 0.99999993427280998L), // cdf complement
  tolerance  // tol
  );

  // Check some bad parameters to the distribution,
#ifndef BOOST_NO_EXCEPTIONS
  BOOST_MATH_CHECK_THROW(boost::math::inverse_gamma_distribution<RealType> igbad1(-1, 0), std::domain_error); // negative shape.
  BOOST_MATH_CHECK_THROW(boost::math::inverse_gamma_distribution<RealType> igbad2(0, -1), std::domain_error); // negative scale.
  BOOST_MATH_CHECK_THROW(boost::math::inverse_gamma_distribution<RealType> igbad2(-1, -1), std::domain_error); // negative scale and shape.
#else
  BOOST_MATH_CHECK_THROW(boost::math::inverse_gamma_distribution<RealType>(-1, 0), std::domain_error); // negative shape.
  BOOST_MATH_CHECK_THROW(boost::math::inverse_gamma_distribution<RealType>(0, -1), std::domain_error); // negative scale.
  BOOST_MATH_CHECK_THROW(boost::math::inverse_gamma_distribution<RealType>(-1, -1), std::domain_error); // negative scale and shape.
#endif

  inverse_gamma_distribution<RealType> ig21(2, 1);

  if(std::numeric_limits<RealType>::has_infinity)
  {
    BOOST_MATH_CHECK_THROW(pdf(ig21, +std::numeric_limits<RealType>::infinity()), std::domain_error); // x = + infinity, pdf = 0
    BOOST_MATH_CHECK_THROW(pdf(ig21, -std::numeric_limits<RealType>::infinity()),  std::domain_error); // x = - infinity, pdf = 0
    BOOST_MATH_CHECK_THROW(cdf(ig21, +std::numeric_limits<RealType>::infinity()),std::domain_error ); // x = + infinity, cdf = 1
    BOOST_MATH_CHECK_THROW(cdf(ig21, -std::numeric_limits<RealType>::infinity()), std::domain_error); // x = - infinity, cdf = 0
    BOOST_MATH_CHECK_THROW(cdf(complement(ig21, +std::numeric_limits<RealType>::infinity())), std::domain_error); // x = + infinity, c cdf = 0
    BOOST_MATH_CHECK_THROW(cdf(complement(ig21, -std::numeric_limits<RealType>::infinity())), std::domain_error); // x = - infinity, c cdf = 1
#ifndef BOOST_NO_EXCEPTIONS
    BOOST_MATH_CHECK_THROW(boost::math::inverse_gamma_distribution<RealType> nbad1(std::numeric_limits<RealType>::infinity(), static_cast<RealType>(1)), std::domain_error); // +infinite mean
    BOOST_MATH_CHECK_THROW(boost::math::inverse_gamma_distribution<RealType> nbad1(-std::numeric_limits<RealType>::infinity(),  static_cast<RealType>(1)), std::domain_error); // -infinite mean
    BOOST_MATH_CHECK_THROW(boost::math::inverse_gamma_distribution<RealType> nbad1(static_cast<RealType>(0), std::numeric_limits<RealType>::infinity()), std::domain_error); // infinite sd
#else
    BOOST_MATH_CHECK_THROW(boost::math::inverse_gamma_distribution<RealType>(std::numeric_limits<RealType>::infinity(), static_cast<RealType>(1)), std::domain_error); // +infinite mean
    BOOST_MATH_CHECK_THROW(boost::math::inverse_gamma_distribution<RealType>(-std::numeric_limits<RealType>::infinity(),  static_cast<RealType>(1)), std::domain_error); // -infinite mean
    BOOST_MATH_CHECK_THROW(boost::math::inverse_gamma_distribution<RealType>(static_cast<RealType>(0), std::numeric_limits<RealType>::infinity()), std::domain_error); // infinite sd
#endif
  }

  if (std::numeric_limits<RealType>::has_quiet_NaN)
  {
    // No longer allow x to be NaN, then these tests should throw.
    BOOST_MATH_CHECK_THROW(pdf(ig21, +std::numeric_limits<RealType>::quiet_NaN()), std::domain_error); // x = NaN
    BOOST_MATH_CHECK_THROW(cdf(ig21, +std::numeric_limits<RealType>::quiet_NaN()), std::domain_error); // x = NaN
    BOOST_MATH_CHECK_THROW(cdf(complement(ig21, +std::numeric_limits<RealType>::quiet_NaN())), std::domain_error); // x = + infinity
    BOOST_MATH_CHECK_THROW(quantile(ig21, +std::numeric_limits<RealType>::quiet_NaN()), std::domain_error); // p = + infinity
    BOOST_MATH_CHECK_THROW(quantile(complement(ig21, +std::numeric_limits<RealType>::quiet_NaN())), std::domain_error); // p = + infinity
  }
    // Spot check for pdf using 'naive pdf' function
  for(RealType x = 0.5; x < 5; x += 0.5)
  {
    BOOST_CHECK_CLOSE_FRACTION(
      pdf(inverse_gamma_distribution<RealType>(5, 6), x),
      naive_pdf(RealType(5), RealType(6), x),
      tolerance);
  }   // Spot checks for parameters:

  RealType tol_few_eps = boost::math::tools::epsilon<RealType>() * 5; // 5 eps as a fraction.
  inverse_gamma_distribution<RealType> dist51(5, 1);
  inverse_gamma_distribution<RealType> dist52(5, 2);
  inverse_gamma_distribution<RealType> dist31(3, 1);
  inverse_gamma_distribution<RealType> dist111(11, 1);
  // 11 mean 0.10000000000000001, variance  0.0011111111111111111, sd 0.033333333333333333

  RealType x = static_cast<RealType>(0.125);
  using namespace std; // ADL of std names.
  using namespace boost::math;

  //  mean, variance etc
  BOOST_CHECK_CLOSE_FRACTION(mean(dist52), static_cast<RealType>(0.5), tol_few_eps);
  BOOST_CHECK_CLOSE_FRACTION(mean(dist111), static_cast<RealType>(0.1L), tol_few_eps);
  inverse_gamma_distribution<RealType> igamma41(static_cast<RealType>(4.), static_cast<RealType>(1.) );
  BOOST_CHECK_CLOSE_FRACTION(mean(igamma41), static_cast<RealType>(0.3333333333333333333333333333333333333333333333333333333L), tol_few_eps);
  // variance:
  BOOST_CHECK_CLOSE_FRACTION(variance(dist51), static_cast<RealType>(0.0208333333333333333333333333333333333333333333333333L), tol_few_eps);
  BOOST_CHECK_CLOSE_FRACTION(variance(dist31), static_cast<RealType>(0.25), tol_few_eps);
  BOOST_CHECK_CLOSE_FRACTION(variance(dist111), static_cast<RealType>(0.001111111111111111111111111111111111111111111111111L), tol_few_eps);
  // std deviation:
  BOOST_CHECK_CLOSE_FRACTION(standard_deviation(dist31), static_cast<RealType>(0.5), tol_few_eps);
  BOOST_CHECK_CLOSE_FRACTION(standard_deviation(dist111), static_cast<RealType>(0.0333333333333333333333333333333333333333333333333L), tol_few_eps);
  // hazard:
  BOOST_CHECK_CLOSE_FRACTION(hazard(dist51, x), pdf(dist51, x) / cdf(complement(dist51, x)), tol_few_eps);
 //  cumulative hazard:
  BOOST_CHECK_CLOSE_FRACTION(chf(dist51, x), -log(cdf(complement(dist51, x))), tol_few_eps);
  // coefficient_of_variation:
  BOOST_CHECK_CLOSE_FRACTION(coefficient_of_variation(dist51), standard_deviation(dist51) / mean(dist51), tol_few_eps);
  // mode:
  BOOST_CHECK_CLOSE_FRACTION(mode(dist51), static_cast<RealType>(0.166666666666666666666666666666666666666666666666666L), tol_few_eps);
  // median
  //BOOST_CHECK_CLOSE_FRACTION(median(dist52), static_cast<RealType>(0), tol_few_eps);
  // Useful to have an exact median?  Failing that use a loop back test.
   BOOST_CHECK_CLOSE_FRACTION(cdf(dist111, median(dist111)), 0.5, tol_few_eps);
  // skewness:
  BOOST_CHECK_CLOSE_FRACTION(skewness(dist111), static_cast<RealType>(1.5), tol_few_eps);
   //kurtosis:
  BOOST_CHECK_CLOSE_FRACTION(kurtosis(dist51), static_cast<RealType>(42 + 3), tol_few_eps);
  // kurtosis excess:
  BOOST_CHECK_CLOSE_FRACTION(kurtosis_excess(dist51), static_cast<RealType>(42), tol_few_eps);

  tol_few_eps = boost::math::tools::epsilon<RealType>() * 3; // 3 eps as a percentage.

  // Special and limit cases:

  if(std::numeric_limits<RealType>::is_specialized)
  {
    RealType mx = (std::numeric_limits<RealType>::max)();
    RealType mi = (std::numeric_limits<RealType>::min)();

     BOOST_CHECK_EQUAL(
     pdf(inverse_gamma_distribution<RealType>(1),
       static_cast<RealType>(mx)), // max()
       static_cast<RealType>(0)
       );

     BOOST_CHECK_EQUAL(
     pdf(inverse_gamma_distribution<RealType>(1),
       static_cast<RealType>(mi)), // min()
       static_cast<RealType>(0)
       );

  }

  BOOST_CHECK_EQUAL(
    pdf(inverse_gamma_distribution<RealType>(1), static_cast<RealType>(0)), static_cast<RealType>(0));
  BOOST_CHECK_EQUAL(
    pdf(inverse_gamma_distribution<RealType>(3), static_cast<RealType>(0))
    , static_cast<RealType>(0.0f));
  BOOST_CHECK_EQUAL(
    cdf(inverse_gamma_distribution<RealType>(1), static_cast<RealType>(0))
    , static_cast<RealType>(0.0f));
  BOOST_CHECK_EQUAL(
    cdf(inverse_gamma_distribution<RealType>(2), static_cast<RealType>(0))
    , static_cast<RealType>(0.0f));
  BOOST_CHECK_EQUAL(
    cdf(inverse_gamma_distribution<RealType>(3), static_cast<RealType>(0))
    , static_cast<RealType>(0.0f));
  BOOST_CHECK_EQUAL(
    cdf(complement(inverse_gamma_distribution<RealType>(1), static_cast<RealType>(0)))
    , static_cast<RealType>(1));
  BOOST_CHECK_EQUAL(
    cdf(complement(inverse_gamma_distribution<RealType>(2), static_cast<RealType>(0)))
    , static_cast<RealType>(1));
  BOOST_CHECK_EQUAL(
    cdf(complement(inverse_gamma_distribution<RealType>(3), static_cast<RealType>(0)))
    , static_cast<RealType>(1));

  BOOST_MATH_CHECK_THROW(
    pdf(
    inverse_gamma_distribution<RealType>(static_cast<RealType>(-1)), // shape negative.
    static_cast<RealType>(1)), std::domain_error
    );
  BOOST_MATH_CHECK_THROW(
    pdf(
    inverse_gamma_distribution<RealType>(static_cast<RealType>(8)),
    static_cast<RealType>(-1)), std::domain_error
    );
  BOOST_MATH_CHECK_THROW(
    cdf(
    inverse_gamma_distribution<RealType>(static_cast<RealType>(-1)),
    static_cast<RealType>(1)), std::domain_error
    );
  BOOST_MATH_CHECK_THROW(
    cdf(
    inverse_gamma_distribution<RealType>(static_cast<RealType>(8)),
    static_cast<RealType>(-1)), std::domain_error
    );
  BOOST_MATH_CHECK_THROW(
    cdf(complement(
    inverse_gamma_distribution<RealType>(static_cast<RealType>(-1)),
    static_cast<RealType>(1))), std::domain_error
    );
  BOOST_MATH_CHECK_THROW(
    cdf(complement(
    inverse_gamma_distribution<RealType>(static_cast<RealType>(8)),
    static_cast<RealType>(-1))), std::domain_error
    );
  BOOST_MATH_CHECK_THROW(
    quantile(
    inverse_gamma_distribution<RealType>(static_cast<RealType>(-1)),
    static_cast<RealType>(0.5)), std::domain_error
    );
  BOOST_MATH_CHECK_THROW(
    quantile(
    inverse_gamma_distribution<RealType>(static_cast<RealType>(8)),
    static_cast<RealType>(-1)), std::domain_error
    );
  BOOST_MATH_CHECK_THROW(
    quantile(
    inverse_gamma_distribution<RealType>(static_cast<RealType>(8)),
    static_cast<RealType>(1.1)), std::domain_error
    );
  BOOST_MATH_CHECK_THROW(
    quantile(complement(
    inverse_gamma_distribution<RealType>(static_cast<RealType>(-1)),
    static_cast<RealType>(0.5))), std::domain_error
    );
  BOOST_MATH_CHECK_THROW(
    quantile(complement(
    inverse_gamma_distribution<RealType>(static_cast<RealType>(8)),
    static_cast<RealType>(-1))), std::domain_error
    );
  BOOST_MATH_CHECK_THROW(
    quantile(complement(
    inverse_gamma_distribution<RealType>(static_cast<RealType>(8)),
    static_cast<RealType>(1.1))), std::domain_error
    );
   check_out_of_range<inverse_gamma_distribution<RealType> >(1, 1);
} // template <class RealType>void test_spots(RealType)

BOOST_AUTO_TEST_CASE( test_main )
{
  BOOST_MATH_CONTROL_FP;

  // Check that can generate inverse_gamma distribution using the two convenience methods:
  // inverse_gamma_distribution; // with default parameters, shape = 1, scale - 1
  using boost::math::inverse_gamma;
  inverse_gamma ig2(2.); // Using typedef and shape parameter (and default scale = 1).
  BOOST_CHECK_EQUAL(ig2.shape(), 2.); // scale  == 2.
  BOOST_CHECK_EQUAL(ig2.scale(), 1.); // scale  == 1 (default).
  inverse_gamma ig; // Using typedef, type double and default values, shape = 1 and scale = 1
  // check default is (1, 1)
  BOOST_CHECK_EQUAL(ig.shape(), 1.); // shape == 1
  BOOST_CHECK_EQUAL(ig.scale(), 1.); // scale  == 1
  BOOST_CHECK_EQUAL(mode(ig), 0.5); // mode = 1/2

  // Used to find some 'exact' values for testing mean, variance ...
 //for (int shape = 4; shape < 30; shape++)
 // {
 //   inverse_gamma ig(shape, 1);
 //   cout.precision(17);
 //   cout << shape << ' ' << mean(ig) << ' ' << variance(ig) << ' ' << standard_deviation(ig)
 //     << ' ' << median(ig) << endl;
 // }

  // and "using boost::math::inverse_gamma_distribution;".
  inverse_gamma_distribution<> ig23(2., 3.); // Using default RealType double.
  BOOST_CHECK_EQUAL(ig23.shape(), 2.); //
  BOOST_CHECK_EQUAL(ig23.scale(), 3.); //

  inverse_gamma_distribution<float> igf23(1.f, 2.f); // Using explicit RealType float.
  BOOST_CHECK_EQUAL(igf23.shape(), 1.f); //
  BOOST_CHECK_EQUAL(igf23.scale(), 2.f); //
  // Some tests using default double.
  double tol5eps = boost::math::tools::epsilon<double>() * 5; // 5 eps as a fraction.
  inverse_gamma_distribution<double> ig102(10., 2.); //
  BOOST_CHECK_EQUAL(ig102.shape(), 10.); //
  BOOST_CHECK_EQUAL(ig102.scale(), 2.); //
  // formatC(SuppDists::dinvGauss(10, 1, 0.5), digits=17)[1] "0.0011774669940754754"
  BOOST_CHECK_CLOSE_FRACTION(pdf(ig102, 0.5), 0.1058495335284024, tol5eps);
  // formatC(SuppDists::pinvGauss(10, 1, 0.5), digits=17) [1] "0.99681494462166653"
  BOOST_CHECK_CLOSE_FRACTION(cdf(ig102, 0.5), 0.99186775720306608, tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(quantile(ig102, 0.05), 0.12734622346137681, tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(quantile(ig102, 0.5), 0.20685272858879727, tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(quantile(ig102, 0.95),  0.36863602680851204, tol5eps);
  // Check mean, etc spot values.
  inverse_gamma_distribution<double> ig51(5., 1.); // shape = 5, scale = 1
  BOOST_CHECK_CLOSE_FRACTION(mean(ig51), 0.25, tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(variance(ig51), 0.0208333333333333333333333333333333333333333, tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(skewness(ig51), 2 * std::sqrt(3.), tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(kurtosis_excess(ig51), 42, tol5eps);
  // mode and median
  inverse_gamma_distribution<double> ig21(1., 2.);
  BOOST_CHECK_CLOSE_FRACTION(mode(ig21), 1, tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(median(ig21), 2.8853900817779268, tol5eps);

  BOOST_CHECK_CLOSE_FRACTION(quantile(ig21, 0.5), 2.8853900817779268, tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(cdf(ig21, median(ig21)), 0.5, tol5eps);

    // Check throws from bad parameters.
  inverse_gamma ig051(0.5, 1.); // shape < 1, so wrong for mean.
  BOOST_MATH_CHECK_THROW(mean(ig051), std::domain_error);
  inverse_gamma ig191(1.9999, 1.); // shape < 2, so wrong for variance.
  BOOST_MATH_CHECK_THROW(variance(ig191), std::domain_error);
  inverse_gamma ig291(2.9999, 1.); // shape < 3, so wrong for skewness.
  BOOST_MATH_CHECK_THROW(skewness(ig291), std::domain_error);
  inverse_gamma ig391(3.9999, 1.); // shape < 1, so wrong for kurtosis and kurtosis_excess.
  BOOST_MATH_CHECK_THROW(kurtosis(ig391), std::domain_error);
  BOOST_MATH_CHECK_THROW(kurtosis_excess(ig391), std::domain_error);

  // Basic sanity-check spot values.
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
  
} // BOOST_AUTO_TEST_CASE( test_main )

/*

Output:

------ Build started: Project: test_inverse_gamma_distribution, Configuration: Release Win32 ------
  test_inverse_gamma_distribution.cpp
  Generating code
  Finished generating code
  test_inverse_gamma_distribution.vcxproj -> J:\Cpp\MathToolkit\test\Math_test\Release\test_inverse_gamma_distribution.exe
  Running 1 test case...
  Tolerance = 0.0001%.
  Tolerance = 0.0001%.
  Tolerance = 0.0001%.
  Tolerance = 0.0001%.

  *** No errors detected
========== Build: 1 succeeded, 0 failed, 0 up-to-date, 0 skipped ==========


*/




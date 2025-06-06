// test_negative_binomial.cpp

// Copyright Paul A. Bristow 2007.
// Copyright John Maddock 2006.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Tests for Negative Binomial Distribution.

// Note that these defines must be placed BEFORE #includes.
#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error
// because several tests overflow & underflow by design.
#define BOOST_MATH_DISCRETE_QUANTILE_POLICY real

#ifdef _MSC_VER
#  pragma warning(disable: 4127) // conditional expression is constant.
#endif

#if !defined(TEST_FLOAT) && !defined(TEST_DOUBLE) && !defined(TEST_LDOUBLE) && !defined(TEST_REAL_CONCEPT)
#  define TEST_FLOAT
#  define TEST_DOUBLE
#  define TEST_LDOUBLE
#  define TEST_REAL_CONCEPT
#endif

#include <boost/math/tools/config.hpp>
#include "../include_private/boost/math/tools/test.hpp"

#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
#include <boost/math/concepts/real_concept.hpp> // for real_concept
using ::boost::math::concepts::real_concept;
#endif

#include <boost/math/distributions/negative_binomial.hpp> // for negative_binomial_distribution
using boost::math::negative_binomial_distribution;

#include <boost/math/special_functions/gamma.hpp>
  using boost::math::lgamma;  // log gamma

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // for test_main
#include <boost/test/tools/floating_point_comparison.hpp> // for BOOST_CHECK_CLOSE
#include "table_type.hpp"
#include "test_out_of_range.hpp"

#include <iostream>
using std::cout;
using std::endl;
using std::setprecision;
using std::showpoint;
#include <limits>
using std::numeric_limits;

template <class RealType>
void test_spot( // Test a single spot value against 'known good' values.
               RealType N,    // Number of successes.
               RealType k,    // Number of failures.
               RealType p,    // Probability of success_fraction.
               RealType P,    // CDF probability.
               RealType Q,    // Complement of CDF.
               RealType tol)  // Test tolerance.
{
   boost::math::negative_binomial_distribution<RealType> bn(N, p);
   BOOST_CHECK_EQUAL(N, bn.successes());
   BOOST_CHECK_EQUAL(p, bn.success_fraction());
   BOOST_CHECK_CLOSE(
     cdf(bn, k), P, tol);

  if((P < 0.99) && (Q < 0.99))
  {
    // We can only check this if P is not too close to 1,
    // so that we can guarantee that Q is free of error:
    //
    BOOST_CHECK_CLOSE(
      cdf(complement(bn, k)), Q, tol);
    if(k != 0)
    {
      BOOST_CHECK_CLOSE(
        quantile(bn, P), k, tol);
    }
    else
    {
      // Just check quantile is very small:
      if((std::numeric_limits<RealType>::max_exponent <= std::numeric_limits<double>::max_exponent)
        && (boost::is_floating_point<RealType>::value))
      {
        // Limit where this is checked: if exponent range is very large we may
        // run out of iterations in our root finding algorithm.
        BOOST_CHECK(quantile(bn, P) < boost::math::tools::epsilon<RealType>() * 10);
      }
    }
    if(k != 0)
    {
      BOOST_CHECK_CLOSE(
        quantile(complement(bn, Q)), k, tol);
    }
    else
    {
      // Just check quantile is very small:
      if((std::numeric_limits<RealType>::max_exponent <= std::numeric_limits<double>::max_exponent)
        && (boost::is_floating_point<RealType>::value))
      {
        // Limit where this is checked: if exponent range is very large we may
        // run out of iterations in our root finding algorithm.
        BOOST_CHECK(quantile(complement(bn, Q)) < boost::math::tools::epsilon<RealType>() * 10);
      }
    }
    // estimate success ratio:
    BOOST_CHECK_CLOSE(
      negative_binomial_distribution<RealType>::find_lower_bound_on_p(
      N+k, N, P),
      p, tol);
    // Note we bump up the sample size here, purely for the sake of the test,
    // internally the function has to adjust the sample size so that we get
    // the right upper bound, our test undoes this, so we can verify the result.
    BOOST_CHECK_CLOSE(
      negative_binomial_distribution<RealType>::find_upper_bound_on_p(
      N+k+1, N, Q),
      p, tol);

    if(Q < P)
    {
       //
       // We check two things here, that the upper and lower bounds
       // are the right way around, and that they do actually bracket
       // the naive estimate of p = successes / (sample size)
       //
      BOOST_CHECK(
        negative_binomial_distribution<RealType>::find_lower_bound_on_p(
        N+k, N, Q)
        <=
        negative_binomial_distribution<RealType>::find_upper_bound_on_p(
        N+k, N, Q)
        );
      BOOST_CHECK(
        negative_binomial_distribution<RealType>::find_lower_bound_on_p(
        N+k, N, Q)
        <=
        N / (N+k)
        );
      BOOST_CHECK(
        N / (N+k)
        <=
        negative_binomial_distribution<RealType>::find_upper_bound_on_p(
        N+k, N, Q)
        );
    }
    else
    {
       // As above but when P is small.
      BOOST_CHECK(
        negative_binomial_distribution<RealType>::find_lower_bound_on_p(
        N+k, N, P)
        <=
        negative_binomial_distribution<RealType>::find_upper_bound_on_p(
        N+k, N, P)
        );
      BOOST_CHECK(
        negative_binomial_distribution<RealType>::find_lower_bound_on_p(
        N+k, N, P)
        <=
        N / (N+k)
        );
      BOOST_CHECK(
        N / (N+k)
        <=
        negative_binomial_distribution<RealType>::find_upper_bound_on_p(
        N+k, N, P)
        );
    }

    // Estimate sample size:
    BOOST_CHECK_CLOSE(
      negative_binomial_distribution<RealType>::find_minimum_number_of_trials(
      k, p, P),
      N+k, tol);
    BOOST_CHECK_CLOSE(
      negative_binomial_distribution<RealType>::find_maximum_number_of_trials(
         k, p, Q),
      N+k, tol);

    // Double check consistency of CDF and PDF by computing the finite sum:
    RealType sum = 0;
    for(unsigned i = 0; i <= k; ++i)
    {
      sum += pdf(bn, RealType(i));
    }
    BOOST_CHECK_CLOSE(sum, P, tol);

    // Complement is not possible since sum is to infinity.
  } //
} // test_spot

template <class RealType> // Any floating-point type RealType.
void test_spots(RealType)
{
  // Basic sanity checks, test data is to double precision only
  // so set tolerance to 1000 eps expressed as a percent, or
  // 1000 eps of type double expressed as a percent, whichever
  // is the larger.

  RealType tolerance = (std::max)
    (boost::math::tools::epsilon<RealType>(),
    static_cast<RealType>(std::numeric_limits<double>::epsilon()));
  tolerance *= 100 * 100000.0f;

  cout << "Tolerance = " << tolerance << "%." << endl;

  RealType tol1eps = boost::math::tools::epsilon<RealType>() * 2; // Very tight, suit exact values.
  //RealType tol2eps = boost::math::tools::epsilon<RealType>() * 2; // Tight, suit exact values.
  RealType tol5eps = boost::math::tools::epsilon<RealType>() * 5; // Wider 5 epsilon.
  cout << "Tolerance 5 eps = " << tol5eps << "%." << endl;

  // Sources of spot test values:

  // MathCAD defines pbinom(k, r, p) (at about 64-bit double precision, about 16 decimal digits)
  // returns pr(X , k) when random variable X has the binomial distribution with parameters r and p.
  // 0 <= k
  // r > 0
  // 0 <= p <= 1
  // P = pbinom(30, 500, 0.05) = 0.869147702104609

  // And functions.wolfram.com

  using boost::math::negative_binomial_distribution;
  using  ::boost::math::negative_binomial;
  using  ::boost::math::cdf;
  using  ::boost::math::pdf;

  // Test negative binomial using cdf spot values from MathCAD cdf = pnbinom(k, r, p).
  // These test quantiles and complements as well.

  test_spot(  // pnbinom(1,2,0.5) = 0.5
  static_cast<RealType>(2),   // successes r
  static_cast<RealType>(1),   // Number of failures, k
  static_cast<RealType>(0.5), // Probability of success as fraction, p
  static_cast<RealType>(0.5), // Probability of result (CDF), P
  static_cast<RealType>(0.5),  // complement CCDF Q = 1 - P
  tolerance);

  test_spot( // pbinom(0, 2, 0.25)
  static_cast<RealType>(2),    // successes r
  static_cast<RealType>(0),    // Number of failures, k
  static_cast<RealType>(0.25),
  static_cast<RealType>(0.0625),                    // Probability of result (CDF), P
  static_cast<RealType>(0.9375),                    // Q = 1 - P
  tolerance);

  test_spot(  // pbinom(48,8,0.25)
  static_cast<RealType>(8),     // successes r
  static_cast<RealType>(48),    // Number of failures, k
  static_cast<RealType>(0.25),                    // Probability of success, p
  static_cast<RealType>(9.826582228110670E-1),     // Probability of result (CDF), P
  static_cast<RealType>(1 - 9.826582228110670E-1),   // Q = 1 - P
  tolerance);

  test_spot(  // pbinom(2,5,0.4)
  static_cast<RealType>(5),     // successes r
  static_cast<RealType>(2),     // Number of failures, k
  static_cast<RealType>(0.4),                    // Probability of success, p
  static_cast<RealType>(9.625600000000020E-2),     // Probability of result (CDF), P
  static_cast<RealType>(1 - 9.625600000000020E-2),   // Q = 1 - P
  tolerance);

  test_spot(  // pbinom(10,100,0.9)
  static_cast<RealType>(100),     // successes r
  static_cast<RealType>(10),     // Number of failures, k
  static_cast<RealType>(0.9),                    // Probability of success, p
  static_cast<RealType>(4.535522887695670E-1),     // Probability of result (CDF), P
  static_cast<RealType>(1 - 4.535522887695670E-1),   // Q = 1 - P
  tolerance);

  test_spot(  // pbinom(1,100,0.991)
  static_cast<RealType>(100),     // successes r
  static_cast<RealType>(1),     // Number of failures, k
  static_cast<RealType>(0.991),                    // Probability of success, p
  static_cast<RealType>(7.693413044217000E-1),     // Probability of result (CDF), P
  static_cast<RealType>(1 - 7.693413044217000E-1),   // Q = 1 - P
  tolerance);

  test_spot(  // pbinom(10,100,0.991)
  static_cast<RealType>(100),     // successes r
  static_cast<RealType>(10),     // Number of failures, k
  static_cast<RealType>(0.991),                    // Probability of success, p
  static_cast<RealType>(9.999999940939000E-1),     // Probability of result (CDF), P
  static_cast<RealType>(1 - 9.999999940939000E-1),   // Q = 1 - P
  tolerance);

if(std::numeric_limits<RealType>::is_specialized)
{ // An extreme value test that takes 3 minutes using the real concept type
  // for which numeric_limits<RealType>::is_specialized == false, deliberately
  // and for which there is no Lanczos approximation defined (also deliberately)
  // giving a very slow computation, but with acceptable accuracy.
  // A possible enhancement might be to use a normal approximation for
  // extreme values, but this is not implemented.
  test_spot(  // pbinom(100000,100,0.001)
  static_cast<RealType>(100),     // successes r
  static_cast<RealType>(100000),     // Number of failures, k
  static_cast<RealType>(0.001),                    // Probability of success, p
  static_cast<RealType>(5.173047534260320E-1),     // Probability of result (CDF), P
  static_cast<RealType>(1 - 5.173047534260320E-1),   // Q = 1 - P
  tolerance*1000); // *1000 is OK 0.51730475350664229  versus

  // functions.wolfram.com
  //   for I[0.001](100, 100000+1) gives:
  // Wolfram       0.517304753506834882009032744488738352004003696396461766326713
  // JM nonLanczos 0.51730475350664229 differs at the 13th decimal digit.
  // MathCAD       0.51730475342603199 differs at 10th decimal digit.

  // Error tests:
  check_out_of_range<negative_binomial_distribution<RealType> >(20, 0.5);
  BOOST_MATH_CHECK_THROW(negative_binomial_distribution<RealType>(0, 0.5), std::domain_error);
  BOOST_MATH_CHECK_THROW(negative_binomial_distribution<RealType>(-2, 0.5), std::domain_error);
  BOOST_MATH_CHECK_THROW(negative_binomial_distribution<RealType>(20, -0.5), std::domain_error);
  BOOST_MATH_CHECK_THROW(negative_binomial_distribution<RealType>(20, 1.5), std::domain_error);
}
 // End of single spot tests using RealType


  // Tests on PDF:
  BOOST_CHECK_CLOSE(
  pdf(negative_binomial_distribution<RealType>(static_cast<RealType>(2), static_cast<RealType>(0.5)),
  static_cast<RealType>(0) ),  // k = 0.
  static_cast<RealType>(0.25), // 0
  tolerance);

  BOOST_CHECK_CLOSE(
  pdf(negative_binomial_distribution<RealType>(static_cast<RealType>(4), static_cast<RealType>(0.5)),
  static_cast<RealType>(0)),  // k = 0.
  static_cast<RealType>(0.0625), // exact 1/16
  tolerance);

  BOOST_CHECK_CLOSE(
  pdf(negative_binomial_distribution<RealType>(static_cast<RealType>(20), static_cast<RealType>(0.25)),
  static_cast<RealType>(0)),  // k = 0
  static_cast<RealType>(9.094947017729270E-13), // pbinom(0,20,0.25) = 9.094947017729270E-13
  tolerance);

  BOOST_CHECK_CLOSE(
  pdf(negative_binomial_distribution<RealType>(static_cast<RealType>(20), static_cast<RealType>(0.2)),
  static_cast<RealType>(0)),  // k = 0
  static_cast<RealType>(1.0485760000000003e-014), // MathCAD 1.048576000000000E-14
  tolerance);

  BOOST_CHECK_CLOSE(
  pdf(negative_binomial_distribution<RealType>(static_cast<RealType>(10), static_cast<RealType>(0.1)),
  static_cast<RealType>(0)),  // k = 0.
  static_cast<RealType>(1e-10), // MathCAD says zero, but suffers cancellation error?
  tolerance);

  BOOST_CHECK_CLOSE(
  pdf(negative_binomial_distribution<RealType>(static_cast<RealType>(20), static_cast<RealType>(0.1)),
  static_cast<RealType>(0)),  // k = 0.
  static_cast<RealType>(1e-20), // MathCAD says zero, but suffers cancellation error?
  tolerance);


  BOOST_CHECK_CLOSE( // .
  pdf(negative_binomial_distribution<RealType>(static_cast<RealType>(20), static_cast<RealType>(0.9)),
  static_cast<RealType>(0)),  // k.
  static_cast<RealType>(1.215766545905690E-1), // k=20  p = 0.9
  tolerance);

  // Tests on cdf:
  // MathCAD pbinom k, r, p) == failures, successes, probability.

  BOOST_CHECK_CLOSE(cdf(
    negative_binomial_distribution<RealType>(static_cast<RealType>(2), static_cast<RealType>(0.5)), // successes = 2,prob 0.25
    static_cast<RealType>(0) ), // k = 0
    static_cast<RealType>(0.25), // probability 1/4
    tolerance);

  BOOST_CHECK_CLOSE(cdf(complement(
    negative_binomial_distribution<RealType>(static_cast<RealType>(2), static_cast<RealType>(0.5)), // successes = 2,prob 0.25
    static_cast<RealType>(0) )), // k = 0
    static_cast<RealType>(0.75), // probability 3/4
    tolerance);
  BOOST_CHECK_CLOSE( // k = 1.
  cdf(negative_binomial_distribution<RealType>(static_cast<RealType>(20), static_cast<RealType>(0.25)),
  static_cast<RealType>(1)),  // k =1.
  static_cast<RealType>(1.455191522836700E-11),
  tolerance);

  BOOST_CHECK_SMALL( // Check within an epsilon with CHECK_SMALL
  cdf(negative_binomial_distribution<RealType>(static_cast<RealType>(20), static_cast<RealType>(0.25)),
  static_cast<RealType>(1)) -
  static_cast<RealType>(1.455191522836700E-11),
  tolerance );

  // Some exact (probably - judging by trailing zeros) values.
  BOOST_CHECK_CLOSE(
  cdf(negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
  static_cast<RealType>(0)),  // k.
  static_cast<RealType>(1.525878906250000E-5),
  tolerance);

  BOOST_CHECK_CLOSE(
  cdf(negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
  static_cast<RealType>(0)),  // k.
  static_cast<RealType>(1.525878906250000E-5),
  tolerance);

  BOOST_CHECK_SMALL(
  cdf(negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
  static_cast<RealType>(0)) -
  static_cast<RealType>(1.525878906250000E-5),
  tolerance );

  BOOST_CHECK_CLOSE( // k = 1.
  cdf(negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
  static_cast<RealType>(1)),  // k.
  static_cast<RealType>(1.068115234375010E-4),
  tolerance);

  BOOST_CHECK_CLOSE( // k = 2.
  cdf(negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
  static_cast<RealType>(2)),  // k.
  static_cast<RealType>(4.158020019531300E-4),
  tolerance);

  BOOST_CHECK_CLOSE( // k = 3.
  cdf(negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
  static_cast<RealType>(3)),  // k.bristow
  static_cast<RealType>(1.188278198242200E-3),
  tolerance);

  BOOST_CHECK_CLOSE( // k = 4.
  cdf(negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
  static_cast<RealType>(4)),  // k.
  static_cast<RealType>(2.781510353088410E-3),
  tolerance);

  BOOST_CHECK_CLOSE( // k = 5.
  cdf(negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
  static_cast<RealType>(5)),  // k.
  static_cast<RealType>(5.649328231811500E-3),
  tolerance);

  BOOST_CHECK_CLOSE( // k = 6.
  cdf(negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
  static_cast<RealType>(6)),  // k.
  static_cast<RealType>(1.030953228473680E-2),
  tolerance);

  BOOST_CHECK_CLOSE( // k = 7.
  cdf(negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
  static_cast<RealType>(7)),  // k.
  static_cast<RealType>(1.729983836412430E-2),
  tolerance);

  BOOST_CHECK_CLOSE( // k = 8.
  cdf(negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
  static_cast<RealType>(8)),  // k = n.
  static_cast<RealType>(2.712995628826370E-2),
  tolerance);

  BOOST_CHECK_CLOSE( //
  cdf(negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
  static_cast<RealType>(48)),  // k
  static_cast<RealType>(9.826582228110670E-1),
  tolerance);

  BOOST_CHECK_CLOSE( //
  cdf(negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
  static_cast<RealType>(64)),  // k
  static_cast<RealType>(9.990295004935590E-1),
  tolerance);

  BOOST_CHECK_CLOSE( //
  cdf(negative_binomial_distribution<RealType>(static_cast<RealType>(5), static_cast<RealType>(0.4)),
  static_cast<RealType>(26)),  // k
  static_cast<RealType>(9.989686246611190E-1),
  tolerance);

  BOOST_CHECK_CLOSE( //
  cdf(negative_binomial_distribution<RealType>(static_cast<RealType>(5), static_cast<RealType>(0.4)),
  static_cast<RealType>(2)),  // k failures
  static_cast<RealType>(9.625600000000020E-2),
  tolerance);

  BOOST_CHECK_CLOSE( //
  cdf(negative_binomial_distribution<RealType>(static_cast<RealType>(50), static_cast<RealType>(0.9)),
  static_cast<RealType>(20)),  // k
  static_cast<RealType>(9.999970854144170E-1),
  tolerance);

  BOOST_CHECK_CLOSE( //
  cdf(negative_binomial_distribution<RealType>(static_cast<RealType>(500), static_cast<RealType>(0.7)),
  static_cast<RealType>(200)),  // k
  static_cast<RealType>(2.172846379930550E-1),
  tolerance* 2);

  BOOST_CHECK_CLOSE( //
  cdf(negative_binomial_distribution<RealType>(static_cast<RealType>(50), static_cast<RealType>(0.7)),
  static_cast<RealType>(20)),  // k
  static_cast<RealType>(4.550203671301790E-1),
  tolerance);

  // Tests of other functions, mean and other moments ...

  negative_binomial_distribution<RealType> dist(static_cast<RealType>(8), static_cast<RealType>(0.25));
  using namespace std; // ADL of std names.
  // mean:
  BOOST_CHECK_CLOSE(
    mean(dist), static_cast<RealType>(8 * (1 - 0.25) /0.25), tol5eps);
  BOOST_CHECK_CLOSE(
    mode(dist), static_cast<RealType>(21), tol1eps);
  // variance:
  BOOST_CHECK_CLOSE(
    variance(dist), static_cast<RealType>(8 * (1 - 0.25) / (0.25 * 0.25)), tol5eps);
  // std deviation:
  BOOST_CHECK_CLOSE(
    standard_deviation(dist), // 9.79795897113271239270
    static_cast<RealType>(9.797958971132712392789136298823565567864L), // using functions.wolfram.com
    //                              9.79795897113271152534  == sqrt(8 * (1 - 0.25) / (0.25 * 0.25)))
    tol5eps * 100);
  BOOST_CHECK_CLOSE(
    skewness(dist), //
    static_cast<RealType>(0.71443450831176036),
    // using http://mathworld.wolfram.com/skewness.html
    tolerance);
  BOOST_CHECK_CLOSE(
    kurtosis_excess(dist), //
    static_cast<RealType>(0.7604166666666666666666666666666666666666L), // using Wikipedia Kurtosis(excess) formula
    tol5eps * 100);
  BOOST_CHECK_CLOSE(
    kurtosis(dist), // true 
    static_cast<RealType>(3.76041666666666666666666666666666666666666L), // 
    tol5eps * 100);
  // hazard:
  RealType x = static_cast<RealType>(0.125);
  BOOST_CHECK_CLOSE(
  hazard(dist, x)
  , pdf(dist, x) / cdf(complement(dist, x)), tol5eps);
  // cumulative hazard:
  BOOST_CHECK_CLOSE(
  chf(dist, x), -log(cdf(complement(dist, x))), tol5eps);
  // coefficient_of_variation:
  BOOST_CHECK_CLOSE(
  coefficient_of_variation(dist)
  , standard_deviation(dist) / mean(dist), tol5eps);

  // Special cases for PDF:
  BOOST_CHECK_EQUAL(
  pdf(
  negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0)), //
  static_cast<RealType>(0)),
  static_cast<RealType>(0) );

  BOOST_CHECK_EQUAL(
  pdf(
  negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0)),
  static_cast<RealType>(0.0001)),
  static_cast<RealType>(0) );

  BOOST_CHECK_EQUAL(
  pdf(
  negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(1)),
  static_cast<RealType>(0.001)),
  static_cast<RealType>(0) );

  BOOST_CHECK_EQUAL(
  pdf(
  negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(1)),
  static_cast<RealType>(8)),
  static_cast<RealType>(0) );

  BOOST_CHECK_SMALL(
  pdf(
   negative_binomial_distribution<RealType>(static_cast<RealType>(2), static_cast<RealType>(0.25)),
  static_cast<RealType>(0))-
  static_cast<RealType>(0.0625),
  2 * boost::math::tools::epsilon<RealType>() ); // Expect exact, but not quite.
  // numeric_limits<RealType>::epsilon()); // Not suitable for real concept!

  // Quantile boundary cases checks:
  BOOST_CHECK_EQUAL(
  quantile(  // zero P < cdf(0) so should be exactly zero.
  negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
  static_cast<RealType>(0)),
  static_cast<RealType>(0));

  BOOST_CHECK_EQUAL(
  quantile(  // min P < cdf(0) so should be exactly zero.
  negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
  static_cast<RealType>(boost::math::tools::min_value<RealType>())),
  static_cast<RealType>(0));

  BOOST_CHECK_CLOSE_FRACTION(
  quantile(  // Small P < cdf(0) so should be near zero.
  negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
  static_cast<RealType>(boost::math::tools::epsilon<RealType>())), // 
  static_cast<RealType>(0),
    tol5eps);

  BOOST_CHECK_CLOSE(
  quantile(  // Small P < cdf(0) so should be exactly zero.
  negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
  static_cast<RealType>(0.0001)),
  static_cast<RealType>(0.95854156929288470),
    tolerance);

  //BOOST_CHECK(  // Fails with overflow for real_concept
  //quantile(  // Small P near 1 so k failures should be big.
  //negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
  //static_cast<RealType>(1 - boost::math::tools::epsilon<RealType>())) <=
  //static_cast<RealType>(189.56999032670058)  // 106.462769 for float
  //);

  if(std::numeric_limits<RealType>::has_infinity)
  { // BOOST_CHECK tests for infinity using std::numeric_limits<>::infinity()
    // Note that infinity is not implemented for real_concept, so these tests
    // are only done for types, like built-in float, double.. that have infinity.
    // Note that these assume that  BOOST_MATH_OVERFLOW_ERROR_POLICY is NOT throw_on_error.
    // #define BOOST_MATH_THROW_ON_OVERFLOW_POLICY ==  throw_on_error would throw here.
    // #define BOOST_MAT_DOMAIN_ERROR_POLICY IS defined throw_on_error,
    //  so the throw path of error handling is tested below with BOOST_MATH_CHECK_THROW tests.

    BOOST_CHECK(
    quantile(  // At P == 1 so k failures should be infinite.
    negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
    static_cast<RealType>(1)) ==
    //static_cast<RealType>(boost::math::tools::infinity<RealType>())
    static_cast<RealType>(std::numeric_limits<RealType>::infinity()) );

    BOOST_CHECK_EQUAL(
    quantile(  // At 1 == P  so should be infinite.
    negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
    static_cast<RealType>(1)), //
    std::numeric_limits<RealType>::infinity() );

    BOOST_CHECK_EQUAL(
    quantile(complement(  // Q zero 1 so P == 1 < cdf(0) so should be exactly infinity.
    negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
    static_cast<RealType>(0))),
    std::numeric_limits<RealType>::infinity() );
   } // test for infinity using std::numeric_limits<>::infinity()
  else
  { // real_concept case, so check it throws rather than returning infinity.
    BOOST_CHECK_EQUAL(
    quantile(  // At P == 1 so k failures should be infinite.
    negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
    static_cast<RealType>(1)),
    boost::math::tools::max_value<RealType>() );

    BOOST_CHECK_EQUAL(
    quantile(complement(  // Q zero 1 so P == 1 < cdf(0) so should be exactly infinity.
    negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
    static_cast<RealType>(0))),
    boost::math::tools::max_value<RealType>());
  }
  BOOST_CHECK( // Should work for built-in and real_concept.
  quantile(complement(  // Q very near to 1 so P nearly 1  < so should be large > 384.
  negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
  static_cast<RealType>(boost::math::tools::min_value<RealType>())))
   >= static_cast<RealType>(384) );

  BOOST_CHECK_EQUAL(
  quantile(  //  P ==  0 < cdf(0) so should be zero.
  negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
  static_cast<RealType>(0)),
  static_cast<RealType>(0));

  // Quantile Complement boundary cases:

  BOOST_CHECK_EQUAL(
  quantile(complement(  // Q = 1 so P = 0 < cdf(0) so should be exactly zero.
  negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
  static_cast<RealType>(1))),
  static_cast<RealType>(0)
  );

  BOOST_CHECK_EQUAL(
  quantile(complement(  // Q very near 1 so P == epsilon < cdf(0) so should be exactly zero.
  negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
  static_cast<RealType>(1 - boost::math::tools::epsilon<RealType>()))),
  static_cast<RealType>(0)
  );

  // Check that duff arguments throw domain_error:
  BOOST_MATH_CHECK_THROW(
  pdf( // Negative successes!
  negative_binomial_distribution<RealType>(static_cast<RealType>(-1), static_cast<RealType>(0.25)),
  static_cast<RealType>(0)), std::domain_error
  );
  BOOST_MATH_CHECK_THROW(
  pdf( // Negative success_fraction!
  negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(-0.25)),
  static_cast<RealType>(0)), std::domain_error
  );
  BOOST_MATH_CHECK_THROW(
  pdf( // Success_fraction > 1!
  negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(1.25)),
  static_cast<RealType>(0)),
  std::domain_error
  );
  BOOST_MATH_CHECK_THROW(
  pdf( // Negative k argument !
  negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
  static_cast<RealType>(-1)),
  std::domain_error
  );
  //BOOST_MATH_CHECK_THROW(
  //pdf( // Unlike binomial there is NO limit on k (failures)
  //negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
  //static_cast<RealType>(9)), std::domain_error
  //);
  BOOST_MATH_CHECK_THROW(
  cdf(  // Negative k argument !
  negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
  static_cast<RealType>(-1)),
  std::domain_error
  );
  BOOST_MATH_CHECK_THROW(
  cdf( // Negative success_fraction!
  negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(-0.25)),
  static_cast<RealType>(0)), std::domain_error
  );
  BOOST_MATH_CHECK_THROW(
  cdf( // Success_fraction > 1!
  negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(1.25)),
  static_cast<RealType>(0)), std::domain_error
  );
  BOOST_MATH_CHECK_THROW(
  quantile(  // Negative success_fraction!
  negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(-0.25)),
  static_cast<RealType>(0)), std::domain_error
  );
  BOOST_MATH_CHECK_THROW(
  quantile( // Success_fraction > 1!
  negative_binomial_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(1.25)),
  static_cast<RealType>(0)), std::domain_error
  );
  // End of check throwing 'duff' out-of-domain values.

#define T RealType
#include "negative_binomial_quantile.ipp"

  for(unsigned i = 0; i < negative_binomial_quantile_data.size(); ++i)
  {
     using namespace boost::math::policies;
     typedef policy<discrete_quantile<boost::math::policies::real> > P1;
     typedef policy<discrete_quantile<integer_round_down> > P2;
     typedef policy<discrete_quantile<integer_round_up> > P3;
     typedef policy<discrete_quantile<integer_round_outwards> > P4;
     typedef policy<discrete_quantile<integer_round_inwards> > P5;
     typedef policy<discrete_quantile<integer_round_nearest> > P6;
     RealType tol = boost::math::tools::epsilon<RealType>() * 700;
     if(!boost::is_floating_point<RealType>::value)
        tol *= 10;  // no lanczos approximation implies less accuracy
     //
     // Check full real value first:
     //
     negative_binomial_distribution<RealType, P1> p1(negative_binomial_quantile_data[i][0], negative_binomial_quantile_data[i][1]);
     RealType x = quantile(p1, negative_binomial_quantile_data[i][2]);
     BOOST_CHECK_CLOSE_FRACTION(x, negative_binomial_quantile_data[i][3], tol);
     x = quantile(complement(p1, negative_binomial_quantile_data[i][2]));
     BOOST_CHECK_CLOSE_FRACTION(x, negative_binomial_quantile_data[i][4], tol);
     //
     // Now with round down to integer:
     //
     negative_binomial_distribution<RealType, P2> p2(negative_binomial_quantile_data[i][0], negative_binomial_quantile_data[i][1]);
     x = quantile(p2, negative_binomial_quantile_data[i][2]);
     BOOST_CHECK_EQUAL(x, floor(negative_binomial_quantile_data[i][3]));
     x = quantile(complement(p2, negative_binomial_quantile_data[i][2]));
     BOOST_CHECK_EQUAL(x, floor(negative_binomial_quantile_data[i][4]));
     //
     // Now with round up to integer:
     //
     negative_binomial_distribution<RealType, P3> p3(negative_binomial_quantile_data[i][0], negative_binomial_quantile_data[i][1]);
     x = quantile(p3, negative_binomial_quantile_data[i][2]);
     BOOST_CHECK_EQUAL(x, ceil(negative_binomial_quantile_data[i][3]));
     x = quantile(complement(p3, negative_binomial_quantile_data[i][2]));
     BOOST_CHECK_EQUAL(x, ceil(negative_binomial_quantile_data[i][4]));
     //
     // Now with round to integer "outside":
     //
     negative_binomial_distribution<RealType, P4> p4(negative_binomial_quantile_data[i][0], negative_binomial_quantile_data[i][1]);
     x = quantile(p4, negative_binomial_quantile_data[i][2]);
     BOOST_CHECK_EQUAL(x, negative_binomial_quantile_data[i][2] < 0.5f ? floor(negative_binomial_quantile_data[i][3]) : ceil(negative_binomial_quantile_data[i][3]));
     x = quantile(complement(p4, negative_binomial_quantile_data[i][2]));
     BOOST_CHECK_EQUAL(x, negative_binomial_quantile_data[i][2] < 0.5f ? ceil(negative_binomial_quantile_data[i][4]) : floor(negative_binomial_quantile_data[i][4]));
     //
     // Now with round to integer "inside":
     //
     negative_binomial_distribution<RealType, P5> p5(negative_binomial_quantile_data[i][0], negative_binomial_quantile_data[i][1]);
     x = quantile(p5, negative_binomial_quantile_data[i][2]);
     BOOST_CHECK_EQUAL(x, negative_binomial_quantile_data[i][2] < 0.5f ? ceil(negative_binomial_quantile_data[i][3]) : floor(negative_binomial_quantile_data[i][3]));
     x = quantile(complement(p5, negative_binomial_quantile_data[i][2]));
     BOOST_CHECK_EQUAL(x, negative_binomial_quantile_data[i][2] < 0.5f ? floor(negative_binomial_quantile_data[i][4]) : ceil(negative_binomial_quantile_data[i][4]));
     //
     // Now with round to nearest integer:
     //
     negative_binomial_distribution<RealType, P6> p6(negative_binomial_quantile_data[i][0], negative_binomial_quantile_data[i][1]);
     x = quantile(p6, negative_binomial_quantile_data[i][2]);
     BOOST_CHECK_EQUAL(x, floor(negative_binomial_quantile_data[i][3] + 0.5f));
     x = quantile(complement(p6, negative_binomial_quantile_data[i][2]));
     BOOST_CHECK_EQUAL(x, floor(negative_binomial_quantile_data[i][4] + 0.5f));
  }

  return;
} // template <class RealType> void test_spots(RealType) // Any floating-point type RealType.

BOOST_AUTO_TEST_CASE( test_main )
{
  // Check that can generate negative_binomial distribution using the two convenience methods:
  using namespace boost::math;
   negative_binomial mynb1(2., 0.5); // Using typedef - default type is double.
   negative_binomial_distribution<> myf2(2., 0.5); // Using default RealType double.

  // Basic sanity-check spot values.

  // Test some simple double only examples.
  negative_binomial_distribution<double> my8dist(8., 0.25);
  // 8 successes (r), 0.25 success fraction = 35% or 1 in 4 successes.
  // Note: double values (matching the distribution definition) avoid the need for any casting.

  // Check accessor functions return exact values for double at least.
  BOOST_CHECK_EQUAL(my8dist.successes(), static_cast<double>(8));
  BOOST_CHECK_EQUAL(my8dist.success_fraction(), static_cast<double>(1./4.));

  // (Parameter value, arbitrarily zero, only communicates the floating point type).
#ifdef TEST_FLOAT
  test_spots(0.0F); // Test float.
#endif
#ifdef TEST_DOUBLE
  test_spots(0.0); // Test double.
#endif
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
#ifdef TEST_LDOUBLE
  test_spots(0.0L); // Test long double.
#endif
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
#ifdef TEST_REAL_CONCEPT
    test_spots(boost::math::concepts::real_concept(0.)); // Test real concept.
#endif
  #endif
#else
   std::cout << "<note>The long double tests have been disabled on this platform "
      "either because the long double overloads of the usual math functions are "
      "not available at all, or because they are too inaccurate for these tests "
      "to pass.</note>" << std::endl;
#endif

  
} // BOOST_AUTO_TEST_CASE( test_main )

/*

Autorun "i:\boost-06-05-03-1300\libs\math\test\Math_test\debug\test_negative_binomial.exe"
Running 1 test case...
Tolerance = 0.0119209%.
Tolerance 5 eps = 5.96046e-007%.
Tolerance = 2.22045e-011%.
Tolerance 5 eps = 1.11022e-015%.
Tolerance = 2.22045e-011%.
Tolerance 5 eps = 1.11022e-015%.
Tolerance = 2.22045e-011%.
Tolerance 5 eps = 1.11022e-015%.
*** No errors detected

*/

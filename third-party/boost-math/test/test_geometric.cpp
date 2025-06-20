// test_geometric.cpp

// Copyright Paul A. Bristow 2010.
// Copyright John Maddock 2010.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Tests for Geometric Distribution.

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

#include <boost/math/distributions/geometric.hpp> // for geometric_distribution
using boost::math::geometric_distribution;
using boost::math::geometric; // using typedef for geometric_distribution<double>

#include <boost/math/distributions/negative_binomial.hpp> // for some comparisons.

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // for test_main
#include <boost/test/tools/floating_point_comparison.hpp> // for BOOST_CHECK_CLOSE_FRACTION
#include "test_out_of_range.hpp"

#include <iostream>
using std::cout;
using std::endl;
using std::setprecision;
using std::showpoint;
#include <limits>
using std::numeric_limits;
#include <cmath>
using std::log;
using std::abs;
#include <type_traits>

template <class RealType>
void test_spot( // Test a single spot value against 'known good' values.
               RealType k,       // Number of failures.
               RealType p,       // Probability of success_fraction.
               RealType P,       // CDF probability.
               RealType Q,       // Complement of CDF.
               RealType logP,    // Logcdf probability
               RealType logQ,    // Complement of logcdf
               RealType tol,     // Test tolerance
               RealType logtol)  // Logcdf Test tolerance.
{
   BOOST_IF_CONSTEXPR (std::is_same<RealType, long double>::value 
                       #ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
                       || std::is_same<RealType, real_concept>::value
                       #endif
                       )
   {
     logtol *= 100;
   }
   
   boost::math::geometric_distribution<RealType> g(p);
   BOOST_CHECK_EQUAL(p, g.success_fraction());
   BOOST_CHECK_CLOSE_FRACTION(cdf(g, k), P, tol);
   BOOST_CHECK_CLOSE_FRACTION(logcdf(g, k), logP, logtol);

  if((P < 0.99) && (Q < 0.99))
  {
    // We can only check this if P is not too close to 1,
    // so that we can guarantee that Q is free of error:
    //
    BOOST_CHECK_CLOSE_FRACTION(
      cdf(complement(g, k)), Q, tol);
    BOOST_CHECK_CLOSE_FRACTION(
      logcdf(complement(g, k)), logQ, logtol);
    if(k != 0)
    {
      BOOST_CHECK_CLOSE_FRACTION(
        quantile(g, P), k, tol);
    }
    else
    {
      // Just check quantile is very small:
      if((std::numeric_limits<RealType>::max_exponent <= std::numeric_limits<double>::max_exponent)
        && (boost::is_floating_point<RealType>::value))
      {
        // Limit where this is checked: if exponent range is very large we may
        // run out of iterations in our root finding algorithm.
        BOOST_CHECK(quantile(g, P) < boost::math::tools::epsilon<RealType>() * 10);
      }
    }
    if(k != 0)
    {
      BOOST_CHECK_CLOSE_FRACTION(
        quantile(complement(g, Q)), k, tol);
    }
    else
    {
      // Just check quantile is very small:
      if((std::numeric_limits<RealType>::max_exponent <= std::numeric_limits<double>::max_exponent)
        && (boost::is_floating_point<RealType>::value))
      {
        // Limit where this is checked: if exponent range is very large we may
        // run out of iterations in our root finding algorithm.
        BOOST_CHECK(quantile(complement(g, Q)) < boost::math::tools::epsilon<RealType>() * 10);
      }
    }
  } //   if((P < 0.99) && (Q < 0.99))

    // Parameter estimation test:  estimate success ratio:
    BOOST_CHECK_CLOSE_FRACTION(
      geometric_distribution<RealType>::find_lower_bound_on_p(
      1+k, P),
      p, 0.02); // Wide tolerance needed for some tests.
   // Note we bump up the sample size here, purely for the sake of the test,
    // internally the function has to adjust the sample size so that we get
    // the right upper bound, our test undoes this, so we can verify the result.
    BOOST_CHECK_CLOSE_FRACTION(
      geometric_distribution<RealType>::find_upper_bound_on_p(
      1+k+1, Q),
      p, 0.02);

    if(Q < P)
    {
       //
       // We check two things here, that the upper and lower bounds
       // are the right way around, and that they do actually bracket
       // the naive estimate of p = successes / (sample size)
       //
      BOOST_CHECK(
        geometric_distribution<RealType>::find_lower_bound_on_p(
        1+k, Q)
        <=
        geometric_distribution<RealType>::find_upper_bound_on_p(
        1+k, Q)
        );
      BOOST_CHECK(
        geometric_distribution<RealType>::find_lower_bound_on_p(
        1+k, Q)
        <=
        1 / (1+k)
        );
      BOOST_CHECK(
        1 / (1+k)
        <=
        geometric_distribution<RealType>::find_upper_bound_on_p(
        1+k, Q)
        );
    }
    else
    {
       // As above but when P is small.
      BOOST_CHECK(
        geometric_distribution<RealType>::find_lower_bound_on_p(
        1+k, P)
        <=
        geometric_distribution<RealType>::find_upper_bound_on_p(
        1+k, P)
        );
      BOOST_CHECK(
        geometric_distribution<RealType>::find_lower_bound_on_p(
        1+k,  P)
        <=
        1 / (1+k)
        );
      BOOST_CHECK(
        1 / (1+k)
        <=
        geometric_distribution<RealType>::find_upper_bound_on_p(
        1+k, P)
        );
    }

    // Estimate sample size:
    BOOST_CHECK_CLOSE_FRACTION(
      geometric_distribution<RealType>::find_minimum_number_of_trials(
      k, p, P),
      1+k, 0.02); // Can differ 50 to 51 for small p
    BOOST_CHECK_CLOSE_FRACTION(
      geometric_distribution<RealType>::find_maximum_number_of_trials(
         k, p, Q),
      1+k, 0.02);

} // test_spot

template <class RealType> // Any floating-point type RealType.
void test_spots(RealType)
{
  // Basic sanity checks.
  // Most test data is to double precision (17 decimal digits) only,

  cout << "Floating point Type is " << typeid(RealType).name() << endl;

  // so set tolerance to 1000 eps expressed as a fraction,
  // or 1000 eps of type double expressed as a fraction,
  // whichever is the larger.

  RealType tolerance = (std::max)
    (boost::math::tools::epsilon<RealType>(),
    static_cast<RealType>(std::numeric_limits<double>::epsilon()));
  tolerance *= 10; // 10 eps

  cout << "Tolerance = " << tolerance << "." << endl;

  RealType tol1eps = boost::math::tools::epsilon<RealType>(); // Very tight, suit exact values.
  //RealType tol2eps = boost::math::tools::epsilon<RealType>() * 2; // Tight,  values.
  RealType tol5eps = boost::math::tools::epsilon<RealType>() * 5; // Wider 5 epsilon.
  cout << "Tolerance 5 eps = " << tol5eps << "." << endl;


  // Sources of spot test values are mainly R.

  using boost::math::geometric_distribution;
  using boost::math::geometric;
  using boost::math::cdf;
  using boost::math::pdf;
  using boost::math::quantile;
  using boost::math::complement;

  BOOST_MATH_STD_USING // for std math functions

  // Test geometric using cdf spot values R
  // These test quantiles and complements as well.

  test_spot(  //
  static_cast<RealType>(2),   // Number of failures, k
  static_cast<RealType>(0.5), // Probability of success as fraction, p
  static_cast<RealType>(0.875L), // Probability of result (CDF), P
  static_cast<RealType>(0.125L),  // complement CCDF Q = 1 - P
  static_cast<RealType>(-0.1335313926245226231463436209313499745894L),
  static_cast<RealType>(-2.079441541679835928251696364374529704227L),
  tolerance,
  tolerance);

  test_spot( //
  static_cast<RealType>(0),    // Number of failures, k
  static_cast<RealType>(0.25), // Probability of success as fraction, p
  static_cast<RealType>(0.25),   // Probability of result (CDF), P
  static_cast<RealType>(0.75),   // Q = 1 - P
  static_cast<RealType>(-1.386294361119890618834464242916353136151L),
  static_cast<RealType>(-0.2876820724517809274392190059938274315035L),
  tolerance,
  tolerance);

  test_spot(
    // R formatC(pgeom(10,0.25), digits=17) [1] "0.95776486396789551"
    // formatC(pgeom(10,0.25, FALSE), digits=17) [1] "0.042235136032104499"

  static_cast<RealType>(10),  // Number of failures, k
  static_cast<RealType>(0.25),  // Probability of success, p
  static_cast<RealType>(0.95776486396789551L),  // Probability of result (CDF), P
  static_cast<RealType>(0.042235136032104499L), // Q = 1 - P
  static_cast<RealType>(-0.04315297584768019483875419429616349387993L),
  static_cast<RealType>(-3.164502796969590201831409065932101746539L),
  tolerance,
  tolerance);

  test_spot(  //
  // > R formatC(pgeom(50,0.25, TRUE), digits=17) [1] "0.99999957525875771"
  // > R formatC(pgeom(50,0.25, FALSE), digits=17) [1] "4.2474124232020353e-07"
  static_cast<RealType>(50),     // Number of failures, k
  static_cast<RealType>(0.25),     // Probability of success, p
  static_cast<RealType>(0.99999957525875771),  // Probability of result (CDF), P
  static_cast<RealType>(4.2474124232020353e-07),   // Q = 1 - P
  static_cast<RealType>(-4.247413325227902241937783772756893037512e-7L),
  static_cast<RealType>(-14.67178569504082729940016930568519898711L),
  tolerance,
  tolerance);
  /*
  // This causes failures in find_upper_bound_on_p p is small branch.
  test_spot(  // formatC(pgeom(50,0.01, TRUE), digits=17)[1] "0.40104399353383874"
    // > formatC(pgeom(50,0.01, FALSE), digits=17) [1] "0.59895600646616121"
  static_cast<RealType>(50), // Number of failures, k
  static_cast<RealType>(0.01),   // Probability of success, p
  static_cast<RealType>(0.40104399353383874),   // Probability of result (CDF), P
  static_cast<RealType>(0.59895600646616121),   // Q = 1 - P
  tolerance);
  */

  test_spot( // > formatC(pgeom(50,0.99, TRUE), digits=17) [1] "                 1"
    // formatC(pgeom(50,0.99, FALSE), digits=17) [1] "1.0000000000000364e-102"
  static_cast<RealType>(50),     // Number of failures, k
  static_cast<RealType>(0.99),    // Probability of success, p
  static_cast<RealType>(1), // Probability of result (CDF), P
  static_cast<RealType>(1.0000000000000364e-102),   // Q = 1 - P
  static_cast<RealType>(-1.0000000000000364e-102L),
  static_cast<RealType>(-std::numeric_limits<RealType>::infinity()),
  tolerance,
  tolerance * 100);

  test_spot(  // > formatC(pgeom(1,0.99, TRUE), digits=17) [1] "0.99990000000000001"
    // > formatC(pgeom(1,0.99, FALSE), digits=17) [1] "0.00010000000000000009"
  static_cast<RealType>(1),     // Number of failures, k
  static_cast<RealType>(0.99),                    // Probability of success, p
  static_cast<RealType>(0.9999),     // Probability of result (CDF), P
  static_cast<RealType>(0.0001),   // Q = 1 - P
  static_cast<RealType>(-0.0001000050003333583353335000142869643968354L),
  static_cast<RealType>(-9.210340371976182736071965818737456830404L),
  tolerance,
  tolerance * 100);

if(std::numeric_limits<RealType>::is_specialized)
{ // An extreme value test that is more accurate than using negative binomial.
  // Since geometric only uses exp and log functions.
  test_spot(  // > formatC(pgeom(10000, 0.001, TRUE), digits=17) [1] "0.99995487182736897"
// > formatC(pgeom(10000,0.001, FALSE), digits=17) [1] "4.5128172631071587e-05"
  static_cast<RealType>(10000L), // Number of failures, k
  static_cast<RealType>(0.001L),                    // Probability of success, p
  static_cast<RealType>(0.99995487182736897L),     // Probability of result (CDF), P
  static_cast<RealType>(4.5128172631071587e-05L),   // Q = 1 - P,
  static_cast<RealType>(-0.00004512919093769043386238651458397312570531L),
  static_cast<RealType>(-10.00600383616891853492996552293751795172L),
  tolerance,
  tolerance * 100); //
  } // numeric_limit is specialized
 // End of single spot tests using RealType

  // Tests on PDF:

  BOOST_CHECK_CLOSE_FRACTION( //> formatC(dgeom(0,0.5), digits=17)[1] " 0.5"
  pdf(geometric_distribution<RealType>(static_cast<RealType>(0.5)),
  static_cast<RealType>(0.0) ),  // Number of failures, k is very small but not integral,
  static_cast<RealType>(0.5), // nearly success probability.
  tolerance);

  BOOST_CHECK_CLOSE_FRACTION( //> formatC(dgeom(0,0.5), digits=17)[1] "    0.5"
    //  R treats geom as a discrete distribution.
    // > formatC(dgeom(1.999999,0.5, FALSE), digits=17) [1] "   0"
    // Warning message:
    // In dgeom(1.999999, 0.5, FALSE) : non-integer x = 1.999999
  pdf(geometric_distribution<RealType>(static_cast<RealType>(0.5)),
  static_cast<RealType>(0.0001L) ),  // Number of failures, k is very small but not integral,
  static_cast<RealType>(0.4999653438420768L), // nearly success probability.
  tolerance);

  BOOST_CHECK_CLOSE_FRACTION( // > formatC(pgeom(0.0001,0.5, TRUE), digits=17)[1] " 0.5"
    // > formatC(pgeom(0.0001,0.5, FALSE), digits=17) [1] "               0.5"
    //  R treats geom as a discrete distribution.
  pdf(geometric_distribution<RealType>(static_cast<RealType>(0.5)),
  static_cast<RealType>(0.0001L) ),  // Number of failures, k is very small but not integral,
  static_cast<RealType>(0.4999653438420768L), // nearly success probability.
  tolerance);

  BOOST_CHECK_CLOSE_FRACTION( // formatC(dgeom(1,0.01), digits=17)[1] "0.0099000000000000008"
  pdf(geometric_distribution<RealType>(static_cast<RealType>(0.01L)),
  static_cast<RealType>(1) ),  // Number of failures, k
  static_cast<RealType>(0.0099000000000000008), //
  tolerance);

  BOOST_CHECK_CLOSE_FRACTION( //> formatC(dgeom(1,0.99), digits=17)[1] "0.0099000000000000043"
  pdf(geometric_distribution<RealType>(static_cast<RealType>(0.99L)),
  static_cast<RealType>(1) ),  // Number of failures, k
  static_cast<RealType>(0.00990000000000000043L), //
  tolerance);

  BOOST_CHECK_CLOSE_FRACTION( //> > formatC(dgeom(0,0.99), digits=17)[1] "0.98999999999999999"
  pdf(geometric_distribution<RealType>(static_cast<RealType>(0.99L)),
  static_cast<RealType>(0) ),  // Number of failures, k
  static_cast<RealType>(0.98999999999999999L), //
  tolerance);

  // p  near unity.
  BOOST_CHECK_CLOSE_FRACTION( // > formatC(dgeom(100,0.99), digits=17)[1] "9.9000000000003448e-201"
  pdf(geometric_distribution<RealType>(static_cast<RealType>(0.99L)),
  static_cast<RealType>(100) ),  // Number of failures, k
  static_cast<RealType>(9.9000000000003448e-201L), //
  100 * tolerance); // Note difference

  // p nearer unity.
  // On GPU this gets flushed to 0 which has an eps difference of 3.4e+38
  #ifndef BOOST_MATH_HAS_GPU_SUPPORT
  BOOST_CHECK_CLOSE_FRACTION( //
  pdf(geometric_distribution<RealType>(static_cast<RealType>(0.9999)),
  static_cast<RealType>(10) ),  // Number of failures, k
  // static_cast<double>(9.9989999999889024e-41), // Boost.Math
  // static_cast<float>(1.00156406e-040)
  static_cast<RealType>(9.999e-41), // exact from 100 digit calculator.
  2e3 * tolerance); // Note bigger tolerance needed.
  #endif

  // Moshier Cephes 100 digits calculator says 9.999e-41
  //0.9999*pow(1-0.9999,10)
  // 9.9990000000000000000000000000000000000000000000000000000000000000000000E-41
  // 9.998999999988988e-041
  // > formatC(dgeom(10, 0.9999), digits=17) [1] "9.9989999999889024e-41"
  // p *  pow(q, k)         9.9989999999889880e-041
  // exp(p * k * log1p(-p)) 9.9989999999889024e-041



  // 0.9999999999 * pow(1-0.9999999999,10)=  9.9999999990E-101
  // > formatC(dgeom(10,0.9999999999), digits=17)  [1] "1.0000008273040127e-100"
  BOOST_CHECK_CLOSE_FRACTION( //
  pdf(geometric_distribution<RealType>(static_cast<RealType>(0.9999999999L)),
  static_cast<RealType>(10) ),  //
  static_cast<RealType>(9.9999999990E-101L), // 1.0000008273040179e-100
  1e9 * tolerance); // Note big tolerance needed.
  // 1.0000008273040179e-100  Boost.Math
  // 1.0000008273040127e-100  R
  // 0.9999999990000004e-100  100 digit calculator 'exact'

  BOOST_CHECK_CLOSE_FRACTION( //
  pdf(geometric_distribution<RealType>(static_cast<RealType>(0.00000000001L)),
  static_cast<RealType>(10) ),  //
  static_cast<RealType>(9.999999999e-12L), // get 9.9999999989999994e-012
  1 * tolerance); // Note small tolerance needed.


    BOOST_CHECK_CLOSE_FRACTION( //
  pdf(geometric_distribution<RealType>(static_cast<RealType>(0.00000000001L)),
  static_cast<RealType>(1000) ),  //
  static_cast<RealType>(9.9999999e-12L), // get 9.9999998999999913e-012
  tolerance); // Note small tolerance needed.


  ///////////////////////////////////////////////////
  BOOST_CHECK_CLOSE_FRACTION( //
    // > formatC(dgeom(0.0001,0.5, FALSE), digits=17) [1] "               0.5"
    //  R treats geom as a discrete distribution.
    // But Boost.Math is continuous, so if you want R behaviour,
    // make number of failures, k into an integer with the floor function.
  pdf(geometric_distribution<RealType>(static_cast<RealType>(0.5)),
  static_cast<RealType>(floor(0.0001L)) ),  // Number of failures, k is very small but MADE integral,
  static_cast<RealType>(0.5), // nearly success probability.
  tolerance);

  // R switches over at about 1e7 from k = 0, returning 0.5,  to k = 1, returning 0.25.
  // Boost.Math does not do this, even for 0.9999999999999999
  // > formatC(pgeom(0.999999,0.5, FALSE), digits=17) [1] "               0.5"
  // > formatC(pgeom(0.9999999,0.5, FALSE), digits=17) [1] "              0.25"

  BOOST_CHECK_CLOSE_FRACTION( // > formatC(pgeom(0.0001,0.5, TRUE), digits=17)[1] "               0.5"
    // > formatC(pgeom(0.0001,0.5, FALSE), digits=17) [1] "               0.5"
    //  R treats geom as a discrete distribution.
    // But Boost.Math is continuous, so if you want R behaviour,
    // make number of failures, k into an integer with the floor function.
  pdf(geometric_distribution<RealType>(static_cast<RealType>(0.5)),
  static_cast<RealType>(floor(0.9999999999999999L)) ),  // Number of failures, k is very small but MADE integral,
  static_cast<RealType>(0.5), // nearly success probability.
  tolerance);

  BOOST_CHECK_CLOSE_FRACTION( // > formatC(pgeom(0.0001,0.5, TRUE), digits=17)[1] "               0.5"
    // > formatC(pgeom(0.0001,0.5, FALSE), digits=17) [1] "               0.5"
    //  R treats geom as a discrete distribution.
    // But Boost.Math is continuous, so if you want R behaviour,
    // make number of failures, k into an integer with the floor function.
  pdf(geometric_distribution<RealType>(static_cast<RealType>(0.5)),
  static_cast<RealType>(floor(1. - tolerance)) ),
  // Number of failures, k is very small but MADE integral,
  // Need to use tolerance here,
  // as epsilon is ill-defined for Real concept:
  // numeric_limits<RealType>::epsilon()  0
  static_cast<RealType>(0.5), // nearly success probability.
  tolerance * 10);

  BOOST_CHECK_CLOSE_FRACTION(
  pdf(geometric_distribution<RealType>(static_cast<RealType>(0.0001L)),
  static_cast<RealType>(2)),  // k = 2.
  static_cast<RealType>(9.99800010e-5L), // 'exact '
  tolerance);

  //> formatC(dgeom(2, 0.9999), digits=17) [1] "9.9989999999977806e-09"
  BOOST_CHECK_CLOSE_FRACTION(
  pdf(geometric_distribution<RealType>(static_cast<RealType>(0.9999L)),
  static_cast<RealType>(2)),  // k = 0
  static_cast<RealType>(9.999e-9L), // 'exact'
  1000*tolerance);

  BOOST_CHECK_CLOSE_FRACTION(
  pdf(geometric_distribution<RealType>(static_cast<RealType>(0.9999L)),
  static_cast<RealType>(3)),  // k = 3
  static_cast<RealType>(9.999e-13L), // get
  1000*tolerance);

  BOOST_CHECK_CLOSE_FRACTION(
  pdf(geometric_distribution<RealType>(static_cast<RealType>(0.9999L)),
  static_cast<RealType>(5)),  // k = 5
  static_cast<RealType>(9.999e-21L), //  9.9989999999944947e-021
  1000*tolerance);


  BOOST_CHECK_CLOSE_FRACTION(
  pdf(geometric_distribution<RealType>( static_cast<RealType>(0.0001L)),
  static_cast<RealType>(3)),  // k = 0.
  static_cast<RealType>(9.99700029999e-5L), //
  tolerance);
   // Tests on cdf:
  // MathCAD pgeom k, r, p) == failures, successes, probability.

  BOOST_CHECK_CLOSE_FRACTION(cdf(
    geometric_distribution<RealType>(static_cast<RealType>(0.5)), // prob 0.5
    static_cast<RealType>(0) ), // k = 0
    static_cast<RealType>(0.5), // probability =p
    tolerance);

  BOOST_CHECK_CLOSE_FRACTION(cdf(complement(
    geometric_distribution<RealType>(static_cast<RealType>(0.5)), //
    static_cast<RealType>(0) )), // k = 0
    static_cast<RealType>(0.5), // probability =
    tolerance);

  BOOST_CHECK_CLOSE_FRACTION(cdf(
    geometric_distribution<RealType>(static_cast<RealType>(0.25)), // prob 0.5
    static_cast<RealType>(1) ), // k = 0
    static_cast<RealType>(0.4375L), // probability =p
    tolerance);

  BOOST_CHECK_CLOSE_FRACTION(cdf(complement(
    geometric_distribution<RealType>(static_cast<RealType>(0.25)), //
    static_cast<RealType>(1) )), // k = 0
    static_cast<RealType>(1-0.4375L), // probability =
    tolerance);

  BOOST_CHECK_CLOSE_FRACTION(cdf(complement(
    geometric_distribution<RealType>(static_cast<RealType>(0.5)), //
    static_cast<RealType>(1) )), // k = 0
    static_cast<RealType>(0.25), // probability = exact 0.25
    tolerance);

  BOOST_CHECK_CLOSE_FRACTION( //
    cdf(geometric_distribution<RealType>(static_cast<RealType>(0.5)),
    static_cast<RealType>(4)),  // k =4.
    static_cast<RealType>(0.96875L), // exact
    tolerance);


  // Tests of other functions, mean and other moments ...

  geometric_distribution<RealType> dist(static_cast<RealType>(0.25));
  // mean:
  BOOST_CHECK_CLOSE_FRACTION(
    mean(dist), static_cast<RealType>((1 - 0.25) /0.25), tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(
    mode(dist), static_cast<RealType>(0), tol1eps);
  // variance:
  BOOST_CHECK_CLOSE_FRACTION(
    variance(dist), static_cast<RealType>((1 - 0.25) / (0.25 * 0.25)), tol5eps);

  // std deviation:
  // sqrt(0.75/0.125)

  BOOST_CHECK_CLOSE_FRACTION(
    standard_deviation(dist), //
    static_cast<RealType>(sqrt((1.0L - 0.25L) / (0.25L * 0.25L))), // using 100 digit calc
    tol5eps);

  BOOST_CHECK_CLOSE_FRACTION(
    skewness(dist), //
    static_cast<RealType>((2-0.25L) /sqrt(0.75L)),
    // using calculator
    tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(
    kurtosis_excess(dist), //
    static_cast<RealType>(6 + 0.0625L/0.75L), //
    tol5eps);
  // 6.083333333333333  6.166666666666667
  BOOST_CHECK_CLOSE_FRACTION(
    kurtosis(dist), // true
    static_cast<RealType>(9 + 0.0625L/0.75L), //
    tol5eps);
  // hazard:
  RealType x = static_cast<RealType>(0.125);
  BOOST_CHECK_CLOSE_FRACTION(
  hazard(dist, x)
  , pdf(dist, x) / cdf(complement(dist, x)), tol5eps);
  // cumulative hazard:
  BOOST_CHECK_CLOSE_FRACTION(
  chf(dist, x), -log(cdf(complement(dist, x))), tol5eps);
  // coefficient_of_variation:
  BOOST_CHECK_CLOSE_FRACTION(
  coefficient_of_variation(dist)
  , standard_deviation(dist) / mean(dist), tol5eps);

  // Special cases for PDF:
  BOOST_CHECK_EQUAL(
  pdf(
  geometric_distribution<RealType>(static_cast<RealType>(0)), //
  static_cast<RealType>(0)),
  static_cast<RealType>(0) );

  BOOST_CHECK_EQUAL(
  pdf(
  geometric_distribution<RealType>(static_cast<RealType>(0)),
  static_cast<RealType>(0.0001)),
  static_cast<RealType>(0) );

  BOOST_CHECK_EQUAL(
  pdf(
  geometric_distribution<RealType>(static_cast<RealType>(1)),
  static_cast<RealType>(0.001)),
  static_cast<RealType>(0) );

  BOOST_CHECK_EQUAL(
  pdf(
  geometric_distribution<RealType>(static_cast<RealType>(1)),
  static_cast<RealType>(8)),
  static_cast<RealType>(0) );

  BOOST_CHECK_SMALL(
  pdf(
   geometric_distribution<RealType>(static_cast<RealType>(0.25)),
  static_cast<RealType>(0))-
  static_cast<RealType>(0.25),
  2 * boost::math::tools::epsilon<RealType>() ); // Expect exact, but not quite.
  // numeric_limits<RealType>::epsilon()); // Not suitable for real concept!

  // Quantile boundary cases checks:
  BOOST_CHECK_EQUAL(
  quantile(  // zero P < cdf(0) so should be exactly zero.
  geometric_distribution<RealType>(static_cast<RealType>(0.25)),
  static_cast<RealType>(0)),
  static_cast<RealType>(0));

  BOOST_CHECK_EQUAL(
  quantile(  // min P < cdf(0) so should be exactly zero.
  geometric_distribution<RealType>(static_cast<RealType>(0.25)),
  static_cast<RealType>(boost::math::tools::min_value<RealType>())),
  static_cast<RealType>(0));

  BOOST_CHECK_CLOSE_FRACTION(
  quantile(  // Small P < cdf(0) so should be near zero.
  geometric_distribution<RealType>(static_cast<RealType>(0.25)),
  static_cast<RealType>(boost::math::tools::epsilon<RealType>())), //
  static_cast<RealType>(0),
    tol5eps);

  BOOST_CHECK_CLOSE_FRACTION(
  quantile(  // Small P < cdf(0) so should be exactly zero.
  geometric_distribution<RealType>(static_cast<RealType>(0.25)),
  static_cast<RealType>(0.0001)),
  static_cast<RealType>(0),
    tolerance);

  //BOOST_CHECK(  // Fails with overflow for real_concept
  //quantile(  // Small P near 1 so k failures should be big.
  //geometric_distribution<RealType>(static_cast<RealType>(8), static_cast<RealType>(0.25)),
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
    geometric_distribution<RealType>(static_cast<RealType>(0.25)),
    static_cast<RealType>(1)) ==
    //static_cast<RealType>(boost::math::tools::infinity<RealType>())
    static_cast<RealType>(std::numeric_limits<RealType>::infinity()) );

    BOOST_CHECK_EQUAL(
    quantile(  // At 1 == P  so should be infinite.
    geometric_distribution<RealType>( static_cast<RealType>(0.25)),
    static_cast<RealType>(1)), //
    std::numeric_limits<RealType>::infinity() );

    BOOST_CHECK_EQUAL(
    quantile(complement(  // Q zero 1 so P == 1 < cdf(0) so should be exactly infinity.
    geometric_distribution<RealType>(static_cast<RealType>(0.25)),
    static_cast<RealType>(0))),
    std::numeric_limits<RealType>::infinity() );
   } // test for infinity using std::numeric_limits<>::infinity()
  else
  { // real_concept case, so check it throws rather than returning infinity.
    BOOST_CHECK_EQUAL(
    quantile(  // At P == 1 so k failures should be infinite.
    geometric_distribution<RealType>(static_cast<RealType>(0.25)),
    static_cast<RealType>(1)),
    boost::math::tools::max_value<RealType>() );

    BOOST_CHECK_EQUAL(
    quantile(complement(  // Q zero 1 so P == 1 < cdf(0) so should be exactly infinity.
    geometric_distribution<RealType>(static_cast<RealType>(0.25)),
    static_cast<RealType>(0))),
    boost::math::tools::max_value<RealType>());
  } // has infinity

  BOOST_CHECK( // Should work for built-in and real_concept.
  quantile(complement(  // Q near to 1 so P nearly 1, so should be large > 300.
  geometric_distribution<RealType>(static_cast<RealType>(0.25)),
  static_cast<RealType>(boost::math::tools::min_value<RealType>())))
   >= static_cast<RealType>(300) );

  BOOST_CHECK_EQUAL(
  quantile(  //  P ==  0 < cdf(0) so should be zero.
  geometric_distribution<RealType>(static_cast<RealType>(0.25)),
  static_cast<RealType>(0)),
  static_cast<RealType>(0));

  // Quantile Complement boundary cases:

  BOOST_CHECK_EQUAL(
  quantile(complement(  // Q = 1 so P = 0 < cdf(0) so should be exactly zero.
  geometric_distribution<RealType>( static_cast<RealType>(0.25)),
  static_cast<RealType>(1))),
  static_cast<RealType>(0)
  );

  BOOST_CHECK_EQUAL(
  quantile(complement(  // Q very near 1 so P == epsilon < cdf(0) so should be exactly zero.
  geometric_distribution<RealType>(static_cast<RealType>(0.25)),
  static_cast<RealType>(1 - boost::math::tools::epsilon<RealType>()))),
  static_cast<RealType>(0)
  );

  // Check that duff arguments throw domain_error:

  BOOST_MATH_CHECK_THROW(
  pdf( // Negative success_fraction!
  geometric_distribution<RealType>(static_cast<RealType>(-0.25)),
  static_cast<RealType>(0)), std::domain_error);
  BOOST_MATH_CHECK_THROW(
  pdf( // Success_fraction > 1!
  geometric_distribution<RealType>(static_cast<RealType>(1.25)),
  static_cast<RealType>(0)),
  std::domain_error);
  BOOST_MATH_CHECK_THROW(
  pdf( // Negative k argument !
  geometric_distribution<RealType>(static_cast<RealType>(0.25)),
  static_cast<RealType>(-1)),
  std::domain_error);
  //BOOST_MATH_CHECK_THROW(
  //pdf( // check limit on k (failures)
  //geometric_distribution<RealType>(static_cast<RealType>(0.25)),
  //std::numeric_limits<RealType>infinity()),
  //std::domain_error);
  BOOST_MATH_CHECK_THROW(
  cdf(  // Negative k argument !
  geometric_distribution<RealType>(static_cast<RealType>(0.25)),
  static_cast<RealType>(-1)),
  std::domain_error);
  BOOST_MATH_CHECK_THROW(
  cdf( // Negative success_fraction!
  geometric_distribution<RealType>(static_cast<RealType>(-0.25)),
  static_cast<RealType>(0)), std::domain_error);
  BOOST_MATH_CHECK_THROW(
  cdf( // Success_fraction > 1!
  geometric_distribution<RealType>(static_cast<RealType>(1.25)),
  static_cast<RealType>(0)), std::domain_error);
  BOOST_MATH_CHECK_THROW(
  quantile(  // Negative success_fraction!
  geometric_distribution<RealType>(static_cast<RealType>(-0.25)),
  static_cast<RealType>(0)), std::domain_error);
  BOOST_MATH_CHECK_THROW(
  quantile( // Success_fraction > 1!
  geometric_distribution<RealType>(static_cast<RealType>(1.25)),
  static_cast<RealType>(0)), std::domain_error);
   check_out_of_range<geometric_distribution<RealType> >(0.5);
  // End of check throwing 'duff' out-of-domain values.

  { // Compare geometric and negative binomial functions.
    using boost::math::negative_binomial_distribution;
    using boost::math::geometric_distribution;

    RealType k = static_cast<RealType>(2.L);
    RealType alpha = static_cast<RealType>(0.05L);
    RealType p = static_cast<RealType>(0.5L);

    BOOST_CHECK_CLOSE_FRACTION( // Successes parameter in negative binomial is 1 for geometric.
      geometric_distribution<RealType>::find_lower_bound_on_p(k, alpha),
      negative_binomial_distribution<RealType>::find_lower_bound_on_p(k, static_cast<RealType>(1), alpha),
      tolerance);
    BOOST_CHECK_CLOSE_FRACTION( // Successes parameter in negative binomial is 1 for geometric.
      geometric_distribution<RealType>::find_upper_bound_on_p(k, alpha),
      negative_binomial_distribution<RealType>::find_upper_bound_on_p(k, static_cast<RealType>(1), alpha),
      tolerance);
    BOOST_CHECK_CLOSE_FRACTION( // Should be identical - successes parameter is not used.
       geometric_distribution<RealType>::find_maximum_number_of_trials(k, p, alpha),
      negative_binomial_distribution<RealType>::find_maximum_number_of_trials(k, p, alpha),
    tolerance);
  }
    //geometric::find_upper_bound_on_p(k, alpha);
   return;
} // template <class RealType> void test_spots(RealType) // Any floating-point type RealType.

BOOST_AUTO_TEST_CASE( test_main )
{
  // Check that can generate geometric distribution using the two convenience methods:
   using namespace boost::math;
   geometric g05d(0.5); // Using typedef - default type is double.
   geometric_distribution<> g05dd(0.5); // Using default RealType double.

  // Basic sanity-check spot values.

  // Test some simple double only examples.
  geometric_distribution<double> mydist(0.25);
  // success fraction == 0.25 == 25% or 1 in 4 successes.
  // Note: double values (matching the distribution definition) avoid the need for any casting.

  // Check accessor functions return exact values for double at least.
  BOOST_CHECK_EQUAL(mydist.success_fraction(), static_cast<double>(1./4.));

  //cout << numeric_limits<RealType>::epsilon() << endl;

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
  #if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x582)) && !defined(BOOST_MATH_NO_REAL_CONCEPT_TESTS)
#if defined(TEST_REAL_CONCEPT) && !defined(BOOST_MATH_NO_REAL_CONCEPT_TESTS)
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



*/

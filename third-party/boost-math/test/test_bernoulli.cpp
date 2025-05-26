// test_bernoulli.cpp

// Copyright John Maddock 2006.
// Copyright  Paul A. Bristow 2007, 2012.
// Copyright Matt Borland 2024

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Basic sanity test for Bernoulli Cumulative Distribution Function.

#ifdef _MSC_VER
#  pragma warning (disable : 4535) // calling _set_se_translator() requires /EHa.
#  pragma warning (disable : 4244) // conversion possible loss of data.
#  pragma warning (disable : 4996) // 'putenv': The POSIX name for this item is deprecated.
#  pragma warning (disable : 4127) // conditional expression is constant.
#endif

// Default domain error policy is
// #define BOOST_MATH_DOMAIN_ERROR_POLICY throw_on_error

#include <boost/math/concepts/real_concept.hpp> // for real_concept
using ::boost::math::concepts::real_concept;
#include "../include_private/boost/math/tools/test.hpp"

#include <boost/math/distributions/bernoulli.hpp> // for bernoulli_distribution
using boost::math::bernoulli_distribution;

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // for test_main
#include <boost/test/tools/floating_point_comparison.hpp> // for BOOST_CHECK_CLOSE_FRACTION, BOOST_CHECK_EQUAL...

#include <iostream>
using std::cout;
using std::endl;
using std::fixed;
using std::right;
using std::left;
using std::showpoint;
using std::showpos;
using std::setw;
using std::setprecision;

#include <limits>
using std::numeric_limits;

template <class RealType> // Any floating-point type RealType.
void test_spots(RealType)
{ // Parameter only provides the type, float, double... value ignored.

  // Basic sanity checks, test data may be to double precision only
  // so set tolerance to 100 eps expressed as a fraction,
  // or 100 eps of type double expressed as a fraction,
  // whichever is the larger.

  RealType tolerance = (std::max)
      (boost::math::tools::epsilon<RealType>(),
      static_cast<RealType>(std::numeric_limits<double>::epsilon()));
   tolerance *= 100;

  cout << "Tolerance for type " << typeid(RealType).name()  << " is "
    << setprecision(3) << tolerance  << " (or " << tolerance * 100 << "%)." << endl;

  // Sources of spot test values - calculator,
  // or Steve Moshier's command interpreter V1.3 100 decimal digit calculator,
  // Wolfram function evaluator.

  using boost::math::bernoulli_distribution; // of type RealType.
  using  ::boost::math::cdf;
  using  ::boost::math::pdf;

  BOOST_CHECK_EQUAL(bernoulli_distribution<RealType>(static_cast<RealType>(0.5)).success_fraction(), static_cast<RealType>(0.5));
  BOOST_CHECK_EQUAL(bernoulli_distribution<RealType>(static_cast<RealType>(0.1L)).success_fraction(), static_cast<RealType>(0.1L));
  BOOST_CHECK_EQUAL(bernoulli_distribution<RealType>(static_cast<RealType>(0.9L)).success_fraction(), static_cast<RealType>(0.9L));

#ifndef BOOST_MATH_NO_EXCEPTIONS
  BOOST_MATH_CHECK_THROW( // Constructor success_fraction outside 0 to 1.
       bernoulli_distribution<RealType>(static_cast<RealType>(2)), std::domain_error);
  BOOST_MATH_CHECK_THROW(
       bernoulli_distribution<RealType>(static_cast<RealType>(-2)), std::domain_error);

  BOOST_MATH_CHECK_THROW(
       pdf( // pdf k neither 0 nor 1.
          bernoulli_distribution<RealType>(static_cast<RealType>(0.25L)), static_cast<RealType>(-1)), std::domain_error);

  BOOST_MATH_CHECK_THROW(
       pdf( // pdf k neither 0 nor 1.
          bernoulli_distribution<RealType>(static_cast<RealType>(0.25L)), static_cast<RealType>(2)), std::domain_error);
#endif

  BOOST_CHECK_EQUAL(
    pdf( // OK k (or n)
    bernoulli_distribution<RealType>(static_cast<RealType>(0.5L)), static_cast<RealType>(0)),
      static_cast<RealType>(0.5)); // Expect 1 - p.

  BOOST_CHECK_CLOSE_FRACTION(
    pdf( // OK k (or n)
    bernoulli_distribution<RealType>(static_cast<RealType>(0.6L)), static_cast<RealType>(0)),
      static_cast<RealType>(0.4L), tolerance); // Expect  1 - p.

  BOOST_CHECK_CLOSE_FRACTION(
    pdf( // OK k (or n)
    bernoulli_distribution<RealType>(static_cast<RealType>(0.6L)), static_cast<RealType>(0)),
      static_cast<RealType>(0.4L), tolerance); // Expect  1- p.

  BOOST_CHECK_CLOSE_FRACTION(
    pdf( // OK k (or n)
    bernoulli_distribution<RealType>(static_cast<RealType>(0.4L)), static_cast<RealType>(0)),
      static_cast<RealType>(0.6L), tolerance); // Expect  1- p.

  BOOST_CHECK_EQUAL(
       mean(bernoulli_distribution<RealType>(static_cast<RealType>(0.5L))), static_cast<RealType>(0.5L));

  BOOST_CHECK_EQUAL(
       mean(bernoulli_distribution<RealType>(static_cast<RealType>(0.1L))),
       static_cast<RealType>(0.1L));

  BOOST_CHECK_CLOSE_FRACTION(
       variance(bernoulli_distribution<RealType>(static_cast<RealType>(0.1L))),
       static_cast<RealType>(0.09L),
       tolerance);

  BOOST_CHECK_CLOSE_FRACTION(
       skewness(bernoulli_distribution<RealType>(static_cast<RealType>(0.1L))),
       static_cast<RealType>(2.666666666666666666666666666666666666666666L),
       tolerance);

  BOOST_CHECK_CLOSE_FRACTION(
       kurtosis(bernoulli_distribution<RealType>(static_cast<RealType>(0.1L))),
       static_cast<RealType>(8.11111111111111111111111111111111111111111111L),
       tolerance);

  BOOST_CHECK_CLOSE_FRACTION(
       kurtosis_excess(bernoulli_distribution<RealType>(static_cast<RealType>(0.1L))),
       static_cast<RealType>(5.11111111111111111111111111111111111111111111L),
       tolerance);

#ifndef BOOST_MATH_NO_EXCEPTIONS
  BOOST_MATH_CHECK_THROW(
     quantile(
        bernoulli_distribution<RealType>(static_cast<RealType>(2)), // prob >1
        static_cast<RealType>(0)), std::domain_error
     );
  BOOST_MATH_CHECK_THROW(
     quantile(
        bernoulli_distribution<RealType>(static_cast<RealType>(-1)), // prob < 0
        static_cast<RealType>(0)), std::domain_error
     );
  BOOST_MATH_CHECK_THROW(
     quantile(
        bernoulli_distribution<RealType>(static_cast<RealType>(0.5L)), // k >1
        static_cast<RealType>(-1)), std::domain_error
     );
  BOOST_MATH_CHECK_THROW(
     quantile(
        bernoulli_distribution<RealType>(static_cast<RealType>(0.5L)), // k < 0
        static_cast<RealType>(2)), std::domain_error
     );
#endif

  BOOST_CHECK_CLOSE_FRACTION(
     cdf(
        bernoulli_distribution<RealType>(static_cast<RealType>(0.6L)),
        static_cast<RealType>(0)), 
        static_cast<RealType>(0.4L), // 1 - p
        tolerance
     );

  BOOST_CHECK_CLOSE_FRACTION(
     cdf(
        bernoulli_distribution<RealType>(static_cast<RealType>(0.6L)),
        static_cast<RealType>(1)), 
        static_cast<RealType>(1), // p
        tolerance
     );

  BOOST_CHECK_CLOSE_FRACTION(
     cdf(complement(
        bernoulli_distribution<RealType>(static_cast<RealType>(0.6L)),
        static_cast<RealType>(1))), 
        static_cast<RealType>(0),
        tolerance
     );

  BOOST_CHECK_CLOSE_FRACTION(
     cdf(complement(
        bernoulli_distribution<RealType>(static_cast<RealType>(0.6L)),
        static_cast<RealType>(0))), 
        static_cast<RealType>(0.6L),
        tolerance
     );

  BOOST_CHECK_EQUAL(
     quantile(
        bernoulli_distribution<RealType>(static_cast<RealType>(0.6L)),
        static_cast<RealType>(0.1L)),  // < p
        static_cast<RealType>(0) 
     );

  BOOST_CHECK_EQUAL(
     quantile(
        bernoulli_distribution<RealType>(static_cast<RealType>(0.6L)),
        static_cast<RealType>(0.9L)),  // > p
        static_cast<RealType>(1) 
     );

   BOOST_CHECK_EQUAL(
     quantile(complement(
        bernoulli_distribution<RealType>(static_cast<RealType>(0.6L)),
        static_cast<RealType>(0.1L))),  // < p
        static_cast<RealType>(1) 
     );

   BOOST_CHECK_EQUAL(
     quantile(complement(
        bernoulli_distribution<RealType>(static_cast<RealType>(0.6L)),
        static_cast<RealType>(0.9L))),  // > p
        static_cast<RealType>(0) 
     );

   // Checks for 'bad' parameters.
   // Construction.
   #ifndef BOOST_MATH_NO_EXCEPTIONS
   BOOST_MATH_CHECK_THROW(bernoulli_distribution<RealType>(-1), std::domain_error); // p outside 0 to 1.
   BOOST_MATH_CHECK_THROW(bernoulli_distribution<RealType>(+2), std::domain_error); // p outside 0 to 1.

   // Parameters.
   bernoulli_distribution<RealType> dist(RealType(1)); 
   BOOST_MATH_CHECK_THROW(pdf(dist, -1), std::domain_error);
   BOOST_MATH_CHECK_THROW(cdf(dist, -1), std::domain_error);
   BOOST_MATH_CHECK_THROW(cdf(complement(dist, -1)), std::domain_error);
   BOOST_MATH_CHECK_THROW(quantile(dist, 2), std::domain_error);
   BOOST_MATH_CHECK_THROW(quantile(complement(dist, -1)), std::domain_error);
   BOOST_MATH_CHECK_THROW(quantile(dist, -1), std::domain_error);
   BOOST_MATH_CHECK_THROW(quantile(complement(dist, -1)), std::domain_error);
     
   // No longer allow any parameter to be NaN or inf, so all these tests should throw.
   if (std::numeric_limits<RealType>::has_quiet_NaN)
   { 
    // Attempt to construct from non-finite should throw.
     RealType nan = std::numeric_limits<RealType>::quiet_NaN();
#ifndef BOOST_NO_EXCEPTIONS
     BOOST_MATH_CHECK_THROW(bernoulli_distribution<RealType> b(nan), std::domain_error);
#else
     BOOST_MATH_CHECK_THROW(bernoulli_distribution<RealType>(nan), std::domain_error);
#endif
    // Non-finite parameters should throw.
     bernoulli_distribution<RealType> b(RealType(1)); 
     BOOST_MATH_CHECK_THROW(pdf(b, +nan), std::domain_error); // x = NaN
     BOOST_MATH_CHECK_THROW(cdf(b, +nan), std::domain_error); // x = NaN
     BOOST_MATH_CHECK_THROW(cdf(complement(b, +nan)), std::domain_error); // x = + nan
     BOOST_MATH_CHECK_THROW(quantile(b, +nan), std::domain_error); // p = + nan
     BOOST_MATH_CHECK_THROW(quantile(complement(b, +nan)), std::domain_error); // p = + nan
  } // has_quiet_NaN

  if (std::numeric_limits<RealType>::has_infinity)
  {
     RealType inf = std::numeric_limits<RealType>::infinity(); 
#ifndef BOOST_NO_EXCEPTIONS
     BOOST_MATH_CHECK_THROW(bernoulli_distribution<RealType> w(inf), std::domain_error);
#else
     BOOST_MATH_CHECK_THROW(bernoulli_distribution<RealType>(inf), std::domain_error);
#endif
     bernoulli_distribution<RealType> w(RealType(1)); 
#ifndef BOOST_NO_EXCEPTIONS
     BOOST_MATH_CHECK_THROW(bernoulli_distribution<RealType> w(inf), std::domain_error);
#else
     BOOST_MATH_CHECK_THROW(bernoulli_distribution<RealType>(inf), std::domain_error);
#endif
     BOOST_MATH_CHECK_THROW(pdf(w, +inf), std::domain_error); // x = inf
     BOOST_MATH_CHECK_THROW(cdf(w, +inf), std::domain_error); // x = inf
     BOOST_MATH_CHECK_THROW(cdf(complement(w, +inf)), std::domain_error); // x = + inf
     BOOST_MATH_CHECK_THROW(quantile(w, +inf), std::domain_error); // p = + inf
     BOOST_MATH_CHECK_THROW(quantile(complement(w, +inf)), std::domain_error); // p = + inf
   } // has_infinity
   #endif
} // template <class RealType>void test_spots(RealType)

BOOST_AUTO_TEST_CASE( test_main )
{
   BOOST_MATH_CONTROL_FP;
   // Check that can generate bernoulli distribution using both convenience methods:
   bernoulli_distribution<double> bn1(0.5); // Using default RealType double.
   boost::math::bernoulli bn2(0.5); // Using typedef. 

  BOOST_CHECK_EQUAL(bn1.success_fraction(), 0.5);
  BOOST_CHECK_EQUAL(bn2.success_fraction(), 0.5);

  BOOST_CHECK_EQUAL(kurtosis(bn2) -3, kurtosis_excess(bn2));
  BOOST_CHECK_EQUAL(kurtosis_excess(bn2), -2);

  //using namespace boost::math; or 
  using boost::math::bernoulli;

  double tol5eps = std::numeric_limits<double>::epsilon() * 5; // 5 eps as a fraction.
  // Default bernoulli is type double, so these test values should also be type double.
  BOOST_CHECK_CLOSE_FRACTION(kurtosis_excess(bernoulli(0.1)), 5.11111111111111111111111111111111111111111111111111, tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(kurtosis_excess(bernoulli(0.9)), 5.11111111111111111111111111111111111111111111111111, tol5eps);
  BOOST_CHECK_CLOSE_FRACTION(kurtosis(bernoulli(0.6)), 1./0.4 + 1./0.6 -3., tol5eps);
  BOOST_CHECK_EQUAL(kurtosis(bernoulli(0)), +std::numeric_limits<double>::infinity());
  BOOST_CHECK_EQUAL(kurtosis(bernoulli(1)), +std::numeric_limits<double>::infinity());
 // 

  // Basic sanity-check spot values.

  // (Parameter value, arbitrarily zero, only communicates the floating point type).
  test_spots(0.0F); // Test float.
  test_spots(0.0); // Test double.
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
  test_spots(0.0L); // Test long double.
#endif
#if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x582)) && !defined(BOOST_MATH_NO_REAL_CONCEPT_TESTS)
  test_spots(boost::math::concepts::real_concept(0.)); // Test real concept.
#endif

} // BOOST_AUTO_TEST_CASE( test_main )

/*

Output is:

  Description: Autorun "J:\Cpp\MathToolkit\test\Math_test\Debug\test_bernouilli.exe"
  Running 1 test case...
  Tolerance for type float is 1.19e-005 (or 0.00119%).
  Tolerance for type double is 2.22e-014 (or 2.22e-012%).
  Tolerance for type long double is 2.22e-014 (or 2.22e-012%).
  Tolerance for type class boost::math::concepts::real_concept is 2.22e-014 (or 2.22e-012%).
  
  *** No errors detected


*/



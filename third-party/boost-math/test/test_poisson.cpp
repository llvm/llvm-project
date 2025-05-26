// test_poisson.cpp

// Copyright Paul A. Bristow 2007.
// Copyright John Maddock 2006.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Basic sanity test for Poisson Cumulative Distribution Function.

#define BOOST_MATH_DISCRETE_QUANTILE_POLICY real

#if !defined(TEST_FLOAT) && !defined(TEST_DOUBLE) && !defined(TEST_LDOUBLE) && !defined(TEST_REAL_CONCEPT)
#  define TEST_FLOAT
#  define TEST_DOUBLE
#  define TEST_LDOUBLE
#  define TEST_REAL_CONCEPT
#endif

#ifdef _MSC_VER
#  pragma warning(disable: 4127) // conditional expression is constant.
#endif

#include <boost/math/tools/config.hpp>

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // Boost.Test
#include <boost/test/tools/floating_point_comparison.hpp>

#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
#include <boost/math/concepts/real_concept.hpp> // for real_concept
#endif

#include <boost/math/distributions/poisson.hpp>
    using boost::math::poisson_distribution;

#include <boost/math/special_functions/gamma.hpp> // for (incomplete) gamma.
//   using boost::math::qamma_Q;
#include "table_type.hpp"
#include "test_out_of_range.hpp"
#include "../include_private/boost/math/tools/test.hpp"

#include <iostream>
   using std::cout;
   using std::endl;
   using std::setprecision;
   using std::showpoint;
   using std::ios;
#include <limits>
  using std::numeric_limits;

template <class RealType> // Any floating-point type RealType.
void test_spots(RealType)
{
  // Basic sanity checks, tolerance is about numeric_limits<RealType>::digits10 decimal places,
   // guaranteed for type RealType, eg 6 for float, 15 for double,
   // expressed as a percentage (so -2) for BOOST_CHECK_CLOSE,

   int decdigits = std::numeric_limits<RealType>::digits10;
  // May eb >15 for 80 and 128-bit FP types.
  if (decdigits <= 0)
  { // decdigits is not defined, for example real concept,
    // so assume precision of most test data is double (for example, MathCAD).
     decdigits = std::numeric_limits<double>::digits10; // == 15 for 64-bit
  }
  if (decdigits > 15 ) // numeric_limits<double>::digits10)
  { // 15 is the accuracy of the MathCAD test data.
    decdigits = 15; // numeric_limits<double>::digits10;
  }

   decdigits -= 1; // Perhaps allow some decimal digit(s) margin of numerical error.
   RealType tolerance = static_cast<RealType>(std::pow(10., static_cast<double>(2-decdigits))); // 1e-6 (-2 so as %)
   tolerance *= 2; // Allow some bit(s) small margin (2 means + or - 1 bit) of numerical error.
   // Typically 2e-13% = 2e-15 as fraction for double.

   // Sources of spot test values:

  // Many be some combinations for which the result is 'exact',
  // or at least is good to 40 decimal digits.
   // 40 decimal digits includes 128-bit significand User Defined Floating-Point types,
   
   // Best source of accurate values is:
   // Mathworld online calculator (40 decimal digits precision, suitable for up to 128-bit significands)
   // http://functions.wolfram.com/webMathematica/FunctionEvaluation.jsp?name=GammaRegularized
   // GammaRegularized is same as gamma incomplete, gamma or gamma_q(a, x) or Q(a, z).

  // http://documents.wolfram.com/calculationcenter/v2/Functions/ListsMatrices/Statistics/PoissonDistribution.html

  // MathCAD defines ppois(k, lambda== mean) as k integer, k >=0.
  // ppois(0, 5) =  6.73794699908547e-3
  // ppois(1, 5) = 0.040427681994513;
  // ppois(10, 10) = 5.830397501929850E-001
  // ppois(10, 1) = 9.999999899522340E-001
  // ppois(5,5) = 0.615960654833065

  // qpois returns inverse Poisson distribution, that is the smallest (floor) k so that ppois(k, lambda) >= p
  // p is real number, real mean lambda > 0
  // k is approximately the integer for which probability(X <= k) = p
  // when random variable X has the Poisson distribution with parameters lambda.
  // Uses discrete bisection.
  // qpois(6.73794699908547e-3, 5) = 1
  // qpois(0.040427681994513, 5) = 

  // Test Poisson with spot values from MathCAD 'known good'.

  using boost::math::poisson_distribution;
  using  ::boost::math::poisson;
  using  ::boost::math::cdf;
  using  ::boost::math::pdf;

   // Check that bad arguments throw.
   #ifndef BOOST_MATH_NO_EXCEPTIONS
   BOOST_MATH_CHECK_THROW(
   cdf(poisson_distribution<RealType>(static_cast<RealType>(0)), // mean zero is bad.
      static_cast<RealType>(0)),  // even for a good k.
      std::domain_error); // Expected error to be thrown.

    BOOST_MATH_CHECK_THROW(
   cdf(poisson_distribution<RealType>(static_cast<RealType>(-1)), // mean negative is bad.
      static_cast<RealType>(0)),
      std::domain_error);

   BOOST_MATH_CHECK_THROW(
   cdf(poisson_distribution<RealType>(static_cast<RealType>(1)), // mean unit OK,
      static_cast<RealType>(-1)),  // but negative events is bad.
      std::domain_error);

  BOOST_MATH_CHECK_THROW(
     cdf(poisson_distribution<RealType>(static_cast<RealType>(0)), // mean zero is bad.
      static_cast<RealType>(99999)),  // for any k events. 
      std::domain_error);
  
  BOOST_MATH_CHECK_THROW(
     cdf(poisson_distribution<RealType>(static_cast<RealType>(0)), // mean zero is bad.
      static_cast<RealType>(99999)),  // for any k events. 
      std::domain_error);

  BOOST_MATH_CHECK_THROW(
     quantile(poisson_distribution<RealType>(static_cast<RealType>(0)), // mean zero.
      static_cast<RealType>(0.5)),  // probability OK. 
      std::domain_error);

  BOOST_MATH_CHECK_THROW(
     quantile(poisson_distribution<RealType>(static_cast<RealType>(-1)), 
      static_cast<RealType>(-1)),  // bad probability. 
      std::domain_error);

  BOOST_MATH_CHECK_THROW(
     quantile(poisson_distribution<RealType>(static_cast<RealType>(1)), 
      static_cast<RealType>(-1)),  // bad probability. 
      std::domain_error);

  BOOST_MATH_CHECK_THROW(
     quantile(poisson_distribution<RealType>(static_cast<RealType>(1)), 
      static_cast<RealType>(1)),  // bad probability. 
      std::overflow_error);

  BOOST_MATH_CHECK_THROW(
     quantile(complement(poisson_distribution<RealType>(static_cast<RealType>(1)), 
      static_cast<RealType>(0))),  // bad probability. 
      std::overflow_error);
   #endif

  BOOST_CHECK_EQUAL(
     quantile(poisson_distribution<RealType>(static_cast<RealType>(1)), 
      static_cast<RealType>(0)),  // bad probability. 
      0);

  BOOST_CHECK_EQUAL(
     quantile(complement(poisson_distribution<RealType>(static_cast<RealType>(1)), 
      static_cast<RealType>(1))),  // bad probability. 
      0);

  // Check some test values.

  BOOST_CHECK_CLOSE( // mode
     mode(poisson_distribution<RealType>(static_cast<RealType>(4))), // mode = mean = 4.
      static_cast<RealType>(4), // mode.
         tolerance);

  //BOOST_CHECK_CLOSE( // mode
  //   median(poisson_distribution<RealType>(static_cast<RealType>(4))), // mode = mean = 4.
  //    static_cast<RealType>(4), // mode.
      //   tolerance);
  poisson_distribution<RealType> dist4(static_cast<RealType>(40));

  BOOST_CHECK_CLOSE( // median
     median(dist4), // mode = mean = 4. median = 40.328333333333333 
      quantile(dist4, static_cast<RealType>(0.5)), // 39.332839138842637
         tolerance);

  // PDF
  BOOST_CHECK_CLOSE(
     pdf(poisson_distribution<RealType>(static_cast<RealType>(4)), // mean 4.
      static_cast<RealType>(0)),   
      static_cast<RealType>(1.831563888873410E-002), // probability.
         tolerance);

  BOOST_CHECK_CLOSE(
     pdf(poisson_distribution<RealType>(static_cast<RealType>(4)), // mean 4.
      static_cast<RealType>(2)),   
      static_cast<RealType>(1.465251111098740E-001), // probability.
         tolerance);

  BOOST_CHECK_CLOSE(
     pdf(poisson_distribution<RealType>(static_cast<RealType>(20)), // mean big.
      static_cast<RealType>(1)),   //  k small
      static_cast<RealType>(4.122307244877130E-008), // probability.
         tolerance);

  BOOST_CHECK_CLOSE(
     pdf(poisson_distribution<RealType>(static_cast<RealType>(4)), // mean 4.
      static_cast<RealType>(20)),   //  K>> mean 
      static_cast<RealType>(8.277463646553730E-009), // probability.
         tolerance);

  // LOGPDF
  BOOST_CHECK_CLOSE(
     logpdf(poisson_distribution<RealType>(static_cast<RealType>(4)), // mean 4.
      static_cast<RealType>(0)),   
      log(static_cast<RealType>(1.831563888873410E-002)), // probability.
         tolerance);

  BOOST_CHECK_CLOSE(
     logpdf(poisson_distribution<RealType>(static_cast<RealType>(4)), // mean 4.
      static_cast<RealType>(2)),   
      log(static_cast<RealType>(1.465251111098740E-001)), // probability.
         tolerance);

  BOOST_CHECK_CLOSE(
     logpdf(poisson_distribution<RealType>(static_cast<RealType>(20)), // mean big.
      static_cast<RealType>(1)),   //  k small
      log(static_cast<RealType>(4.122307244877130E-008)), // probability.
         tolerance);

  BOOST_CHECK_CLOSE(
     logpdf(poisson_distribution<RealType>(static_cast<RealType>(4)), // mean 4.
      static_cast<RealType>(20)),   //  K>> mean 
      log(static_cast<RealType>(8.277463646553730E-009)), // probability.
         tolerance);
  
  // CDF
  BOOST_CHECK_CLOSE(
     cdf(poisson_distribution<RealType>(static_cast<RealType>(1)), // mean unity.
      static_cast<RealType>(0)),  // zero k events. 
      static_cast<RealType>(3.678794411714420E-1), // probability.
         tolerance);

  BOOST_CHECK_CLOSE(
     cdf(poisson_distribution<RealType>(static_cast<RealType>(1)), // mean unity.
      static_cast<RealType>(1)),  // one k event. 
      static_cast<RealType>(7.357588823428830E-1), // probability.
         tolerance);

  BOOST_CHECK_CLOSE(
     cdf(poisson_distribution<RealType>(static_cast<RealType>(1)), // mean unity.
      static_cast<RealType>(2)),  // two k events. 
      static_cast<RealType>(9.196986029286060E-1), // probability.
         tolerance);

  BOOST_CHECK_CLOSE(
     cdf(poisson_distribution<RealType>(static_cast<RealType>(1)), // mean unity.
      static_cast<RealType>(10)),  // two k events. 
      static_cast<RealType>(9.999999899522340E-1), // probability.
         tolerance);

  BOOST_CHECK_CLOSE(
     cdf(poisson_distribution<RealType>(static_cast<RealType>(1)), // mean unity.
      static_cast<RealType>(15)),  // two k events. 
      static_cast<RealType>(9.999999999999810E-1), // probability.
         tolerance);

  BOOST_CHECK_CLOSE(
     cdf(poisson_distribution<RealType>(static_cast<RealType>(1)), // mean unity.
      static_cast<RealType>(16)),  // two k events. 
      static_cast<RealType>(9.999999999999990E-1), // probability.
         tolerance);

  BOOST_CHECK_CLOSE(
     cdf(poisson_distribution<RealType>(static_cast<RealType>(1)), // mean unity.
      static_cast<RealType>(17)),  // two k events. 
      static_cast<RealType>(1.), // probability unity for double.
         tolerance);

  BOOST_CHECK_CLOSE(
     cdf(poisson_distribution<RealType>(static_cast<RealType>(1)), // mean unity.
      static_cast<RealType>(33)),  // k events at limit for float unchecked_factorial table. 
      static_cast<RealType>(1.), // probability.
         tolerance);

  BOOST_CHECK_CLOSE(
     cdf(poisson_distribution<RealType>(static_cast<RealType>(100)), // mean 100.
      static_cast<RealType>(33)),  // k events at limit for float unchecked_factorial table. 
      static_cast<RealType>(6.328271240363390E-15), // probability is tiny.
         tolerance * static_cast<RealType>(2e11)); // 6.3495253382825722e-015 MathCAD
      // Note that there two tiny probability are much more different.

   BOOST_CHECK_CLOSE(
     cdf(poisson_distribution<RealType>(static_cast<RealType>(100)), // mean 100.
      static_cast<RealType>(34)),  // k events at limit for float unchecked_factorial table. 
      static_cast<RealType>(1.898481372109020E-14), // probability is tiny.
         tolerance*static_cast<RealType>(2e11)); //         1.8984813721090199e-014 MathCAD


 BOOST_CHECK_CLOSE(
     cdf(poisson_distribution<RealType>(static_cast<RealType>(33)), // mean = k
      static_cast<RealType>(33)),  // k events above limit for float unchecked_factorial table. 
      static_cast<RealType>(5.461191812386560E-1), // probability.
         tolerance);

 BOOST_CHECK_CLOSE(
     cdf(poisson_distribution<RealType>(static_cast<RealType>(33)), // mean = k-1
      static_cast<RealType>(34)),  // k events above limit for float unchecked_factorial table. 
      static_cast<RealType>(6.133535681502950E-1), // probability.
         tolerance);

 BOOST_CHECK_CLOSE(
     cdf(poisson_distribution<RealType>(static_cast<RealType>(1)), // mean unity.
      static_cast<RealType>(34)),  // k events above limit for float unchecked_factorial table. 
      static_cast<RealType>(1.), // probability.
         tolerance);

  BOOST_CHECK_CLOSE(
     cdf(poisson_distribution<RealType>(static_cast<RealType>(5.)), // mean
      static_cast<RealType>(5)),  // k events. 
      static_cast<RealType>(0.615960654833065), // probability.
         tolerance);
  BOOST_CHECK_CLOSE(
     cdf(poisson_distribution<RealType>(static_cast<RealType>(5.)), // mean
      static_cast<RealType>(1)),  // k events. 
      static_cast<RealType>(0.040427681994512805), // probability.
         tolerance);

  BOOST_CHECK_CLOSE(
     cdf(poisson_distribution<RealType>(static_cast<RealType>(5.)), // mean
      static_cast<RealType>(0)),  // k events (uses special case formula, not gamma). 
      static_cast<RealType>(0.006737946999085467), // probability.
         tolerance);

  BOOST_CHECK_CLOSE(
     cdf(poisson_distribution<RealType>(static_cast<RealType>(1.)), // mean
      static_cast<RealType>(0)),  // k events (uses special case formula, not gamma). 
      static_cast<RealType>(0.36787944117144233), // probability.
         tolerance);

  BOOST_CHECK_CLOSE(
     cdf(poisson_distribution<RealType>(static_cast<RealType>(10.)), // mean
      static_cast<RealType>(10)),  // k events. 
      static_cast<RealType>(0.5830397501929856), // probability.
         tolerance);

  BOOST_CHECK_CLOSE(
     cdf(poisson_distribution<RealType>(static_cast<RealType>(4.)), // mean
      static_cast<RealType>(5)),  // k events. 
      static_cast<RealType>(0.785130387030406), // probability.
         tolerance);

  // complement CDF
  BOOST_CHECK_CLOSE( // Complement CDF
     cdf(complement(poisson_distribution<RealType>(static_cast<RealType>(4.)), // mean
      static_cast<RealType>(5))),  // k events. 
      static_cast<RealType>(1 - 0.785130387030406), // probability.
         tolerance);

  BOOST_CHECK_CLOSE( // Complement CDF
     cdf(complement(poisson_distribution<RealType>(static_cast<RealType>(4.)), // mean
      static_cast<RealType>(0))),  // Zero k events (uses special case formula, not gamma).
      static_cast<RealType>(0.98168436111126578), // probability.
         tolerance);
  BOOST_CHECK_CLOSE( // Complement CDF
     cdf(complement(poisson_distribution<RealType>(static_cast<RealType>(1.)), // mean
      static_cast<RealType>(0))),  // Zero k events (uses special case formula, not gamma).
      static_cast<RealType>(0.63212055882855767), // probability.
         tolerance);

  // Example where k is bigger than max_factorial (>34 for float)
  // (therefore using log gamma so perhaps less accurate).
  BOOST_CHECK_CLOSE(
     cdf(poisson_distribution<RealType>(static_cast<RealType>(40.)), // mean
      static_cast<RealType>(40)),  // k events. 
      static_cast<RealType>(0.5419181783625430), // probability.
         tolerance);

   // Quantile & complement.
  BOOST_CHECK_CLOSE(
    boost::math::quantile(
         poisson_distribution<RealType>(5),  // mean.
         static_cast<RealType>(0.615960654833065)),  //  probability.
         static_cast<RealType>(5.), // Expect k = 5
         tolerance/5); // 

  // EQUAL is too optimistic - fails [5.0000000000000124 != 5]
  // BOOST_CHECK_EQUAL(boost::math::quantile( // 
  //       poisson_distribution<RealType>(5.),  // mean.
  //       static_cast<RealType>(0.615960654833065)),  //  probability.
  //       static_cast<RealType>(5.)); // Expect k = 5 events.
 
  BOOST_CHECK_CLOSE(boost::math::quantile(
         poisson_distribution<RealType>(4),  // mean.
         static_cast<RealType>(0.785130387030406)),  //  probability.
         static_cast<RealType>(5.), // Expect k = 5 events.
         tolerance/5); 

  // Check on quantile of other examples of inverse of cdf.
  BOOST_CHECK_CLOSE( 
     cdf(poisson_distribution<RealType>(static_cast<RealType>(10.)), // mean
      static_cast<RealType>(10)),  // k events. 
      static_cast<RealType>(0.5830397501929856), // probability.
         tolerance);

  BOOST_CHECK_CLOSE(boost::math::quantile( // inverse of cdf above.
         poisson_distribution<RealType>(10.),  // mean.
         static_cast<RealType>(0.5830397501929856)),  //  probability.
         static_cast<RealType>(10.), // Expect k = 10 events.
         tolerance/5); 


  BOOST_CHECK_CLOSE(
     cdf(poisson_distribution<RealType>(static_cast<RealType>(4.)), // mean
      static_cast<RealType>(5)),  // k events. 
      static_cast<RealType>(0.785130387030406), // probability.
         tolerance);

  BOOST_CHECK_CLOSE(boost::math::quantile( // inverse of cdf above.
         poisson_distribution<RealType>(4.),  // mean.
         static_cast<RealType>(0.785130387030406)),  //  probability.
         static_cast<RealType>(5.), // Expect k = 10 events.
         tolerance/5); 



  //BOOST_CHECK_CLOSE(boost::math::quantile(
  //       poisson_distribution<RealType>(5),  // mean.
  //       static_cast<RealType>(0.785130387030406)),  //  probability.
  //        // 6.1882832344329559 result but MathCAD givest smallest integer ppois(k, mean) >= prob
  //       static_cast<RealType>(6.), // Expect k = 6 events. 
  //       tolerance/5); 

  //BOOST_CHECK_CLOSE(boost::math::quantile(
  //       poisson_distribution<RealType>(5),  // mean.
  //       static_cast<RealType>(0.77)),  //  probability.
  //        // 6.1882832344329559 result but MathCAD givest smallest integer ppois(k, mean) >= prob
  //       static_cast<RealType>(7.), // Expect k = 6 events. 
  //       tolerance/5); 

  //BOOST_CHECK_CLOSE(boost::math::quantile(
  //       poisson_distribution<RealType>(5),  // mean.
  //       static_cast<RealType>(0.75)),  //  probability.
  //        // 6.1882832344329559 result but MathCAD givest smallest integer ppois(k, mean) >= prob
  //       static_cast<RealType>(6.), // Expect k = 6 events. 
  //       tolerance/5); 

  BOOST_CHECK_CLOSE(
    boost::math::quantile(
         complement(
           poisson_distribution<RealType>(4),
           static_cast<RealType>(1 - 0.785130387030406))),  // complement.
           static_cast<RealType>(5), // Expect k = 5 events.
         tolerance/5);

  BOOST_CHECK_EQUAL(boost::math::quantile( // Check case when probability < cdf(0) (== pdf(0))
         poisson_distribution<RealType>(1),  // mean is small, so cdf and pdf(0) are about 0.35.
         static_cast<RealType>(0.0001)),  //  probability < cdf(0).
         static_cast<RealType>(0)); // Expect k = 0 events exactly.
          
  BOOST_CHECK_EQUAL(
    boost::math::quantile(
         complement(
           poisson_distribution<RealType>(1),
           static_cast<RealType>(0.9999))),  // complement, so 1-probability < cdf(0)
           static_cast<RealType>(0)); // Expect k = 0 events exactly.

  //
  // Test quantile policies against test data:
  //
#define T RealType
#include "poisson_quantile.ipp"

  for(unsigned i = 0; i < poisson_quantile_data.size(); ++i)
  {
     using namespace boost::math::policies;
     typedef policy<discrete_quantile<boost::math::policies::real> > P1;
     typedef policy<discrete_quantile<integer_round_down> > P2;
     typedef policy<discrete_quantile<integer_round_up> > P3;
     typedef policy<discrete_quantile<integer_round_outwards> > P4;
     typedef policy<discrete_quantile<integer_round_inwards> > P5;
     typedef policy<discrete_quantile<integer_round_nearest> > P6;
     RealType tol = boost::math::tools::epsilon<RealType>() * 20;
     if(!boost::is_floating_point<RealType>::value)
        tol *= 7;
     //
     // Check full real value first:
     //
     poisson_distribution<RealType, P1> p1(poisson_quantile_data[i][0]);
     RealType x = quantile(p1, poisson_quantile_data[i][1]);
     BOOST_CHECK_CLOSE_FRACTION(x, poisson_quantile_data[i][2], tol);
     x = quantile(complement(p1, poisson_quantile_data[i][1]));
     BOOST_CHECK_CLOSE_FRACTION(x, poisson_quantile_data[i][3], tol * 3);
     //
     // Now with round down to integer:
     //
     poisson_distribution<RealType, P2> p2(poisson_quantile_data[i][0]);
     x = quantile(p2, poisson_quantile_data[i][1]);
     BOOST_CHECK_EQUAL(x, floor(poisson_quantile_data[i][2]));
     x = quantile(complement(p2, poisson_quantile_data[i][1]));
     BOOST_CHECK_EQUAL(x, floor(poisson_quantile_data[i][3]));
     //
     // Now with round up to integer:
     //
     poisson_distribution<RealType, P3> p3(poisson_quantile_data[i][0]);
     x = quantile(p3, poisson_quantile_data[i][1]);
     BOOST_CHECK_EQUAL(x, ceil(poisson_quantile_data[i][2]));
     x = quantile(complement(p3, poisson_quantile_data[i][1]));
     BOOST_CHECK_EQUAL(x, ceil(poisson_quantile_data[i][3]));
     //
     // Now with round to integer "outside":
     //
     poisson_distribution<RealType, P4> p4(poisson_quantile_data[i][0]);
     x = quantile(p4, poisson_quantile_data[i][1]);
     BOOST_CHECK_EQUAL(x, poisson_quantile_data[i][1] < 0.5f ? floor(poisson_quantile_data[i][2]) : ceil(poisson_quantile_data[i][2]));
     x = quantile(complement(p4, poisson_quantile_data[i][1]));
     BOOST_CHECK_EQUAL(x, poisson_quantile_data[i][1] < 0.5f ? ceil(poisson_quantile_data[i][3]) : floor(poisson_quantile_data[i][3]));
     //
     // Now with round to integer "inside":
     //
     poisson_distribution<RealType, P5> p5(poisson_quantile_data[i][0]);
     x = quantile(p5, poisson_quantile_data[i][1]);
     BOOST_CHECK_EQUAL(x, poisson_quantile_data[i][1] < 0.5f ? ceil(poisson_quantile_data[i][2]) : floor(poisson_quantile_data[i][2]));
     x = quantile(complement(p5, poisson_quantile_data[i][1]));
     BOOST_CHECK_EQUAL(x, poisson_quantile_data[i][1] < 0.5f ? floor(poisson_quantile_data[i][3]) : ceil(poisson_quantile_data[i][3]));
     //
     // Now with round to nearest integer:
     //
     poisson_distribution<RealType, P6> p6(poisson_quantile_data[i][0]);
     x = quantile(p6, poisson_quantile_data[i][1]);
     BOOST_CHECK_EQUAL(x, floor(poisson_quantile_data[i][2] + 0.5f));
     x = quantile(complement(p6, poisson_quantile_data[i][1]));
     BOOST_CHECK_EQUAL(x, floor(poisson_quantile_data[i][3] + 0.5f));
  }
   check_out_of_range<poisson_distribution<RealType> >(1);
} // template <class RealType>void test_spots(RealType)

//

BOOST_AUTO_TEST_CASE( test_main )
{
  // Check that can construct normal distribution using the two convenience methods:
  using namespace boost::math;
  poisson myp1(2); // Using typedef
   poisson_distribution<> myp2(2); // Using default RealType double.

   // Basic sanity-check spot values.

  // Some plain double examples & tests:
  cout.precision(17); // double max_digits10
  cout.setf(ios::showpoint);
  
  poisson mypoisson(4.); // // mean = 4, default FP type is double.
  cout << "mean(mypoisson, 4.) == " << mean(mypoisson) << endl;
  cout << "mean(mypoisson, 0.) == " << mean(mypoisson) << endl;
  cout << "cdf(mypoisson, 2.) == " << cdf(mypoisson, 2.) << endl;
  cout << "pdf(mypoisson, 2.) == " << pdf(mypoisson, 2.) << endl;
  
  // poisson mydudpoisson(0.);
  // throws (if BOOST_MATH_DOMAIN_ERROR_POLICY == throw_on_error).

#ifndef BOOST_MATH_NO_EXCEPTIONS
#ifndef BOOST_NO_EXCEPTIONS
  BOOST_MATH_CHECK_THROW(poisson mydudpoisson(-1), std::domain_error);// Mean must be > 0.
  BOOST_MATH_CHECK_THROW(poisson mydudpoisson(-1), std::logic_error);// Mean must be > 0.
#else
  BOOST_MATH_CHECK_THROW(poisson(-1), std::domain_error);// Mean must be > 0.
  BOOST_MATH_CHECK_THROW(poisson(-1), std::logic_error);// Mean must be > 0.
#endif
  // Passes the check because logic_error is a parent????
  // BOOST_MATH_CHECK_THROW(poisson mydudpoisson(-1), std::overflow_error); // fails the check
  // because overflow_error is unrelated - except from std::exception
  BOOST_MATH_CHECK_THROW(cdf(mypoisson, -1), std::domain_error); // k must be >= 0
#endif
  BOOST_CHECK_EQUAL(mean(mypoisson), 4.);
  BOOST_CHECK_CLOSE(
  pdf(mypoisson, 2.),  // k events = 2. 
    1.465251111098740E-001, // probability.
      5e-13);

  BOOST_CHECK_CLOSE(
  cdf(mypoisson, 2.),  // k events = 2. 
    0.238103305553545, // probability.
      5e-13);


#if 0
  // Compare cdf from finite sum of pdf and gamma_q.
  using boost::math::cdf;
  using boost::math::pdf;

  double mean = 4.;
  cout.precision(17); // double max_digits10
  cout.setf(ios::showpoint);
  cout << showpoint << endl;  // Ensure trailing zeros are shown.
  // This also helps show the expected precision max_digits10
  //cout.unsetf(ios::showpoint); // No trailing zeros are shown.

  cout << "k          pdf                     sum                  cdf                   diff" << endl;
  double sum = 0.;
  for (int i = 0; i <= 50; i++)
  {
   cout << i << ' ' ;
   double p =  pdf(poisson_distribution<double>(mean), static_cast<double>(i));
   sum += p;

   cout << p << ' ' << sum << ' ' 
   << cdf(poisson_distribution<double>(mean), static_cast<double>(i)) << ' ';
     {
       cout << boost::math::gamma_q<double>(i+1, mean); // cdf
       double diff = boost::math::gamma_q<double>(i+1, mean) - sum; // cdf -sum
       cout << setprecision (2) << ' ' << diff; // 0 0 to 4, 1 eps 5 to 9, 10 to 20 2 eps, 21 upwards 3 eps
      
     }
    BOOST_CHECK_CLOSE(
    cdf(mypoisson, static_cast<double>(i)),
      sum, // of pdfs.
      4e-14); // Fails at 2e-14
   // This call puts the precision etc back to default 6 !!!
   cout << setprecision(17) << showpoint;


     cout << endl;
  }

   cout << cdf(poisson_distribution<double>(5), static_cast<double>(0)) << ' ' << endl; // 0.006737946999085467
   cout << cdf(poisson_distribution<double>(5), static_cast<double>(1)) << ' ' << endl; // 0.040427681994512805
   cout << cdf(poisson_distribution<double>(2), static_cast<double>(3)) << ' ' << endl; // 0.85712346049854715 

   { // Compare approximate formula in Wikipedia with quantile(half)
     for (int i = 1; i < 100; i++)
     {
       poisson_distribution<double> distn(static_cast<double>(i));
       cout << i << ' ' << median(distn) << ' ' << quantile(distn, 0.5) << ' ' 
         << median(distn) - quantile(distn, 0.5) << endl; // formula appears to be out-by-one??
     }  // so quantile(half) used via derived accressors.
   }
#endif

   // (Parameter value, arbitrarily zero, only communicates the floating-point type).
#ifdef TEST_POISSON
  test_spots(0.0F); // Test float.
#endif
#ifdef TEST_DOUBLE
  test_spots(0.0); // Test double.
#endif
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
  if (std::numeric_limits<long double>::digits10 > std::numeric_limits<double>::digits10)
  { // long double is better than double (so not MSVC where they are same).
#ifdef TEST_LDOUBLE
     test_spots(0.0L); // Test long double.
#endif
  }

#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
#ifdef TEST_REAL_CONCEPT
  test_spots(boost::math::concepts::real_concept(0.)); // Test real concept.
#endif
#endif
#endif
   
} // BOOST_AUTO_TEST_CASE( test_main )

/*

Output:

Autorun "i:\boost-06-05-03-1300\libs\math\test\Math_test\debug\test_poisson.exe"
Running 1 test case...
mean(mypoisson, 4.) == 4.0000000000000000
mean(mypoisson, 0.) == 4.0000000000000000
cdf(mypoisson, 2.) == 0.23810330555354431
pdf(mypoisson, 2.) == 0.14652511110987343
*** No errors detected

*/

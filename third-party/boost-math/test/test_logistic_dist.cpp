// Copyright 2008 Gautam Sewani
// Copyright 2013 Paul A. Bristow
// Copyright 2024 Matt Borland
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef _MSC_VER
#  pragma warning (disable : 4127) // conditional expression is constant.
#  pragma warning (disable : 4512) // assignment operator could not be generated.
#endif

#include <boost/config.hpp>
#include <boost/math/tools/config.hpp>

#if !defined(BOOST_NO_EXCEPTIONS) && !defined(BOOST_MATH_NO_EXCEPTIONS)
#  define BOOST_MATH_UNDERFLOW_ERROR_POLICY throw_on_error
#  define BOOST_MATH_OVERFLOW_ERROR_POLICY throw_on_error
#endif

#include "../include_private/boost/math/tools/test.hpp"
#include <boost/math/concepts/real_concept.hpp> // for real_concept
#include <boost/math/distributions/logistic.hpp>
    using boost::math::logistic_distribution;

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // Boost.Test
#include <boost/test/tools/floating_point_comparison.hpp>
#include "test_out_of_range.hpp"

#include <type_traits>
#include <iostream>
   using std::cout;
   using std::endl;
   using std::setprecision;

template <class RealType>
void test_spot(RealType location, RealType scale, RealType x, RealType p, RealType q, RealType logp, RealType logq, RealType tolerance, RealType logtolerance, RealType logtoleranceq)
{
   if (std::is_same<RealType, long double>::value || std::is_same<RealType, boost::math::concepts::real_concept>::value)
   {
      logtoleranceq *= 100;
   }
   
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
      logistic_distribution<RealType>(location,scale),      
      x),
      p,
      tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::logcdf(
      logistic_distribution<RealType>(location,scale),      
      x),
      logp,
      logtolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
      complement(logistic_distribution<RealType>(location,scale),      
      x)),
      q,
      tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::logcdf(
      complement(logistic_distribution<RealType>(location,scale),      
      x)),
      log(q),
      logtoleranceq); // %
   if(p < 0.999)
   {
      BOOST_CHECK_CLOSE(
         ::boost::math::quantile(
         logistic_distribution<RealType>(location,scale),      
         p),
         x,
         tolerance); // %
   }
   if(q < 0.999)
   {
      BOOST_CHECK_CLOSE(
         ::boost::math::quantile(
         complement(logistic_distribution<RealType>(location,scale),      
         q)),
         x,
         3 * tolerance); // %
   }
}


template <class RealType>
void test_spots(RealType T)
{
   // Basic sanity checks.
   // 50 eps as a percentage, up to a maximum of double precision
   // Test data taken from Mathematica 6
   RealType tolerance = (std::max)(
      static_cast<RealType>(1e-33L),
      boost::math::tools::epsilon<RealType>());

   #if !(defined(__amd64__) || defined(__x86_64__) || defined(_M_AMD64))
   if (std::is_same<RealType, long double>::value || std::is_same<RealType, boost::math::concepts::real_concept>::value)
   {
      tolerance = std::numeric_limits<double>::epsilon();
   } 
   #endif

   cout<<"Absolute tolerance:"<<tolerance<<endl;

   tolerance *= 50 * 100; 
   // #  pragma warning(disable: 4100) // unreferenced formal parameter.
   // prevent his spurious warning.
   if (T != 0)
   {
      cout << "Expect parameter T == 0!" << endl;
   }
   cout << "Tolerance for type " << typeid(T).name()  << " is " << tolerance << " %" << endl;

   test_spot(
      static_cast<RealType>(1), // location
      static_cast<RealType>(0.5L), // scale
      static_cast<RealType>(0.1L), // x
      static_cast<RealType>(0.141851064900487789594278108470953L), // p
      static_cast<RealType>(0.858148935099512210405721891529047L), //q
      static_cast<RealType>(-1.95297761052607413459937686381969L), // log(p)
      static_cast<RealType>(-0.15297761052607413459937686381969L), // log(q)
      tolerance,
      tolerance,
      tolerance);
   
   test_spot(
      static_cast<RealType>(5), // location
      static_cast<RealType>(2), // scale
      static_cast<RealType>(3.123123123L),//x 
      static_cast<RealType>(0.281215878622547904873088053477813L), // p
      static_cast<RealType>(0.718784121377452095126911946522187L), //q
      static_cast<RealType>(-1.26863265327659162571930803425160L), // log(p)
      static_cast<RealType>(-0.33019421477659162571930803425160L), // log(q)
      tolerance,
      tolerance,
      tolerance);
   
   test_spot(
      static_cast<RealType>(1.2345L), // location
      static_cast<RealType>(0.12345L), // scale
      static_cast<RealType>(3.123123123L),//x
      static_cast<RealType>(0.999999773084685079723328282229357L), // p
      static_cast<RealType>(2.26915314920276671717770643005212e-7L), //q
      static_cast<RealType>(-2.2691534066556063906815875163479345e-7L), // Log(p) Edge case
      static_cast<RealType>(-15.298688951095170556204624487357L), // log(q)
      tolerance,
      tolerance * 100000,
      tolerance * 100000);


   //High probability
   test_spot(
      static_cast<RealType>(1), // location
      static_cast<RealType>(0.5L), // scale
      static_cast<RealType>(10), // x
      static_cast<RealType>(0.99999998477002048723965105559179L), // p  
      static_cast<RealType>(1.5229979512760348944408208801237e-8L), //q
      static_cast<RealType>(-1.5229979628736488101501003766470705e-8L), // log(p)
      static_cast<RealType>(-18.000000015229979628736488101501L), // Log(q)
      tolerance,
      tolerance * 100000,
      tolerance * 100000);

   //negative x
   test_spot(
      static_cast<RealType>(5), // location
      static_cast<RealType>(2), // scale
      static_cast<RealType>(-0.1L), // scale
      static_cast<RealType>(0.0724264853615177178439235061476928L), // p
      static_cast<RealType>(0.927573514638482282156076493852307L), //q
      static_cast<RealType>(-2.6251832265757900570194868863670116L), // log(p)
      static_cast<RealType>(-0.075183226575790057019486886367011L), // log(q)
      tolerance,
      tolerance,
      tolerance);

   test_spot(
      static_cast<RealType>(5), // location
      static_cast<RealType>(2), // scale
      static_cast<RealType>(-20), // x
      static_cast<RealType>(3.72663928418656138608800947863869e-6L), // p
      static_cast<RealType>(0.999996273360715813438613911990521L), //q
      static_cast<RealType>(-12.500003726646228123990312715291263L), // log(p)
      static_cast<RealType>(-3.7266462281239903127152912633031e-6L), // Log(q)
      tolerance,
      tolerance,
      tolerance * 100000);

   // Test value to check cancellation error in straight/complemented quantile.
   // The subtraction in the formula location-scale*log term introduces catastrophic
   // cancellation error if location and scale*log term are close.
   // For these values, the tests fail at tolerance, but work at 100*tolerance.
   test_spot(
      static_cast<RealType>(-1.2345L), // location
      static_cast<RealType>(1.4555L), // scale
      static_cast<RealType>(-0.00125796420642514024493852425918807L),// x
      static_cast<RealType>(0.7L), // p
      static_cast<RealType>(0.3L), //q
      static_cast<RealType>(-0.35667494393873237891263871124118448L), // Log(p)
      static_cast<RealType>(-1.2039728043259359926227462177618L), // log(q)
      80*tolerance,
      tolerance,
      tolerance);   

   test_spot(
      static_cast<RealType>(1.2345L), // location
      static_cast<RealType>(0.12345L), // scale
      static_cast<RealType>(0.0012345L), // x
      static_cast<RealType>(0.0000458541039469413343331170952855318L), // p
      static_cast<RealType>(0.999954145896053058665666882904714L), //q
      static_cast<RealType>(-9.9900458551552785044234447480622748L), // Log(p)
      static_cast<RealType>(-0.000045855155278504423444748062274193L), // log(q)
      80*tolerance,
      tolerance,
      tolerance * 100000);
   
   test_spot(
      static_cast<RealType>(5L), // location
      static_cast<RealType>(2L), // scale
      static_cast<RealType>(0.0012345L), // x
      static_cast<RealType>(0.0759014628704232983512906076564256L), // p
      static_cast<RealType>(0.924098537129576701648709392343574L), //q
      static_cast<RealType>(-2.5783193211111713937468666265459702L), // Log(p)
      static_cast<RealType>(-0.078936571111171393746866626545970L), // log(q)
      80*tolerance,
      tolerance,
      tolerance);

   //negative location
   test_spot(
      static_cast<RealType>(-123.123123L), // location
      static_cast<RealType>(2.123L), // scale
      static_cast<RealType>(3), // x
      static_cast<RealType>(0.999999999999999999999999984171276L), // p
      static_cast<RealType>(1.58287236765203121622150720373972e-26L), //q
      static_cast<RealType>(-1.58287236765203121622150720373972e-26L), // Log(p)
      static_cast<RealType>(-59.407971267074894017899199262178L), // log(q)
      tolerance,
      tolerance * 100000,
      tolerance * 10000000);

   //PDF Testing
   BOOST_CHECK_CLOSE(
      ::boost::math::pdf(
      logistic_distribution<RealType>(5,2),      
         static_cast<RealType>(0.125L) ),//x
         static_cast<RealType>(0.0369500730133475464584898192104821L),              // probability
      tolerance); // %

   BOOST_CHECK_CLOSE(
      ::boost::math::pdf(
         logistic_distribution<RealType>(static_cast<RealType>(1.2345L), static_cast<RealType>(0.12345L)),      
         static_cast<RealType>(0.0012345L) ),//x
         static_cast<RealType>(0.000371421639109700748742498671686243L),              // probability
      tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::pdf(
      logistic_distribution<RealType>(2,1),      
         static_cast<RealType>(2L) ),//x
         static_cast<RealType>(0.25L),              // probability
      tolerance); // %

   //Extreme value testing

   if(std::numeric_limits<RealType>::has_infinity)
   {
      BOOST_CHECK_EQUAL(pdf(logistic_distribution<RealType>(), +std::numeric_limits<RealType>::infinity()), 0); // x = + infinity, pdf = 0
      BOOST_CHECK_EQUAL(pdf(logistic_distribution<RealType>(), -std::numeric_limits<RealType>::infinity()), 0); // x = - infinity, pdf = 0
      BOOST_CHECK_EQUAL(cdf(logistic_distribution<RealType>(), +std::numeric_limits<RealType>::infinity()), 1); // x = + infinity, cdf = 1
      BOOST_CHECK_EQUAL(cdf(logistic_distribution<RealType>(), -std::numeric_limits<RealType>::infinity()), 0); // x = - infinity, cdf = 0
      BOOST_CHECK_EQUAL(cdf(complement(logistic_distribution<RealType>(), +std::numeric_limits<RealType>::infinity())), 0); // x = + infinity, c cdf = 0
      BOOST_CHECK_EQUAL(cdf(complement(logistic_distribution<RealType>(), -std::numeric_limits<RealType>::infinity())), 1); // x = - infinity, c cdf = 1
   }
   BOOST_MATH_CHECK_THROW(quantile(logistic_distribution<RealType>(), static_cast<RealType>(1)), std::overflow_error); // x = + infinity, cdf = 1
   BOOST_MATH_CHECK_THROW(quantile(logistic_distribution<RealType>(), static_cast<RealType>(0)), std::overflow_error); // x = - infinity, cdf = 0
   BOOST_MATH_CHECK_THROW(quantile(complement(logistic_distribution<RealType>(), static_cast<RealType>(1))), std::overflow_error); // x = - infinity, cdf = 0
   BOOST_MATH_CHECK_THROW(quantile(complement(logistic_distribution<RealType>(), static_cast<RealType>(0))), std::overflow_error); // x = + infinity, cdf = 1
   BOOST_CHECK_EQUAL(cdf(logistic_distribution<RealType>(), +boost::math::tools::max_value<RealType>()), 1); // x = + infinity, cdf = 1
   BOOST_CHECK_EQUAL(cdf(logistic_distribution<RealType>(), -boost::math::tools::max_value<RealType>()), 0); // x = - infinity, cdf = 0
   BOOST_CHECK_EQUAL(cdf(complement(logistic_distribution<RealType>(), +boost::math::tools::max_value<RealType>())), 0); // x = + infinity, c cdf = 0
   BOOST_CHECK_EQUAL(cdf(complement(logistic_distribution<RealType>(), -boost::math::tools::max_value<RealType>())), 1); // x = - infinity, c cdf = 1
   BOOST_CHECK_EQUAL(pdf(logistic_distribution<RealType>(), +boost::math::tools::max_value<RealType>()), 0); // x = + infinity, pdf = 0
   BOOST_CHECK_EQUAL(pdf(logistic_distribution<RealType>(), -boost::math::tools::max_value<RealType>()), 0); // x = - infinity, pdf = 0

   //
   // Things that are errors:
   // 1. Domain errors for scale and location.
   // 2. x being NAN.
   // 3. Probabilities being outside (0,1).
   check_out_of_range<logistic_distribution<RealType> >(0, 1);
   if(std::numeric_limits<RealType>::has_infinity)
   {
      RealType inf = std::numeric_limits<RealType>::infinity();
      BOOST_CHECK_EQUAL(pdf(logistic_distribution<RealType>(0, 1), inf), 0);
      BOOST_CHECK_EQUAL(pdf(logistic_distribution<RealType>(0, 1), -inf), 0);
      BOOST_CHECK_EQUAL(cdf(logistic_distribution<RealType>(0, 1), inf), 1);
      BOOST_CHECK_EQUAL(cdf(logistic_distribution<RealType>(0, 1), -inf), 0);
      BOOST_CHECK_EQUAL(cdf(complement(logistic_distribution<RealType>(0, 1), inf)), 0);
      BOOST_CHECK_EQUAL(cdf(complement(logistic_distribution<RealType>(0, 1), -inf)), 1);
   }

   // location/scale can't be infinity.
   if(std::numeric_limits<RealType>::has_infinity)
   {
#ifndef BOOST_NO_EXCEPTIONS
      BOOST_MATH_CHECK_THROW(
         logistic_distribution<RealType> dist(std::numeric_limits<RealType>::infinity(), 0.5),
         std::domain_error);
      BOOST_MATH_CHECK_THROW(
         logistic_distribution<RealType> dist(0.5, std::numeric_limits<RealType>::infinity()),
         std::domain_error);
#else
      BOOST_MATH_CHECK_THROW(
         logistic_distribution<RealType>(std::numeric_limits<RealType>::infinity(), 0.5),
         std::domain_error);
      BOOST_MATH_CHECK_THROW(
         logistic_distribution<RealType>(0.5, std::numeric_limits<RealType>::infinity()),
         std::domain_error);
#endif
   }
   // scale can't be negative or 0.
#ifndef BOOST_NO_EXCEPTIONS
   BOOST_MATH_CHECK_THROW(
      logistic_distribution<RealType> dist(0.5, -0.5),
      std::domain_error);
   BOOST_MATH_CHECK_THROW(
      logistic_distribution<RealType> dist(0.5, 0),
      std::domain_error);
#else
   BOOST_MATH_CHECK_THROW(
      logistic_distribution<RealType>(0.5, -0.5),
      std::domain_error);
   BOOST_MATH_CHECK_THROW(
      logistic_distribution<RealType>(0.5, 0),
      std::domain_error);
#endif

   logistic_distribution<RealType> dist(0.5, 0.5);
   // x can't be NaN, p can't be NaN.

   if (std::numeric_limits<RealType>::has_quiet_NaN)
   {
      // No longer allow x to be NaN, then these tests should throw.
      BOOST_MATH_CHECK_THROW(pdf(dist, +std::numeric_limits<RealType>::quiet_NaN()), std::domain_error); // x = NaN
      BOOST_MATH_CHECK_THROW(cdf(dist, +std::numeric_limits<RealType>::quiet_NaN()), std::domain_error); // x = NaN
      BOOST_MATH_CHECK_THROW(cdf(complement(dist, +std::numeric_limits<RealType>::quiet_NaN())), std::domain_error); // x = + infinity
      BOOST_MATH_CHECK_THROW(quantile(dist, +std::numeric_limits<RealType>::quiet_NaN()), std::domain_error); // p = + infinity
      BOOST_MATH_CHECK_THROW(quantile(complement(dist, +std::numeric_limits<RealType>::quiet_NaN())), std::domain_error); // p = + infinity
   }
   if (std::numeric_limits<RealType>::has_infinity)
   {
  // Added test for Trac https://svn.boost.org/trac/boost/ticket/9126#comment:1
     logistic_distribution<RealType> dist(0., 0.5);
     BOOST_CHECK_EQUAL(pdf(dist, +std::numeric_limits<RealType>::infinity()), static_cast<RealType>(0) ); // x = infinity

   }


   // p can't be outside (0,1).
   BOOST_MATH_CHECK_THROW(quantile(dist, static_cast<RealType>(1.1)), std::domain_error); 
   BOOST_MATH_CHECK_THROW(quantile(dist, static_cast<RealType>(-0.1)), std::domain_error);
   BOOST_MATH_CHECK_THROW(quantile(dist, static_cast<RealType>(1)), std::overflow_error); 
   BOOST_MATH_CHECK_THROW(quantile(dist, static_cast<RealType>(0)), std::overflow_error);

   BOOST_MATH_CHECK_THROW(quantile(complement(dist, static_cast<RealType>(1.1))), std::domain_error); 
   BOOST_MATH_CHECK_THROW(quantile(complement(dist, static_cast<RealType>(-0.1))), std::domain_error);
   BOOST_MATH_CHECK_THROW(quantile(complement(dist, static_cast<RealType>(1))), std::overflow_error); 
   BOOST_MATH_CHECK_THROW(quantile(complement(dist, static_cast<RealType>(0))), std::overflow_error); 

   // Tests for mean,mode,median,variance,skewness,kurtosis.
   //mean
   BOOST_CHECK_CLOSE(
      ::boost::math::mean(
      logistic_distribution<RealType>(2,1)      
      ),//x
      static_cast<RealType>(2),              // probability
      tolerance); // %
   //median
   BOOST_CHECK_CLOSE(
      ::boost::math::median(
      logistic_distribution<RealType>(2,1)      
      ),//x
      static_cast<RealType>(2),              // probability
      tolerance);
   //mode
   BOOST_CHECK_CLOSE(
      ::boost::math::mode(
      logistic_distribution<RealType>(2,1)      
      ),//x
      static_cast<RealType>(2),              // probability
      tolerance);
   //variance
   BOOST_CHECK_CLOSE(
      ::boost::math::variance(
      logistic_distribution<RealType>(2,1)      
      ),//x
      static_cast<RealType>(3.28986813369645287294483033329205L),  // probability
      tolerance);
   //skewness
   BOOST_CHECK_CLOSE(
      ::boost::math::skewness(
      logistic_distribution<RealType>(2,1)      
      ),//x
      static_cast<RealType>(0),              // probability
      tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::kurtosis_excess(
      logistic_distribution<RealType>(2,1)      
      ),//x
      static_cast<RealType>(1.2L),              // probability
      tolerance);

   BOOST_CHECK_CLOSE(
      ::boost::math::entropy(
      logistic_distribution<RealType>(2,1)
      ),
      static_cast<RealType>(2),
      tolerance);

} // template <class RealType>void test_spots(RealType)


BOOST_AUTO_TEST_CASE( test_main )
{
  // Check that can generate logistic distribution using the two convenience methods:
   boost::math::logistic mycexp1(1.); // Using typedef
   logistic_distribution<> myexp2(1.); // Using default RealType double.

   // Basic sanity-check spot values.
   // (Parameter value, arbitrarily zero, only communicates the floating point type).
  test_spots(0.0F); // Test float. OK at decdigits = 0 tolerance = 0.0001 %
  test_spots(0.0); // Test double. OK at decdigits 7, tolerance = 1e07 %
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
  test_spots(0.0L); // Test long double.
#if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x582)) && !defined(BOOST_MATH_NO_REAL_CONCEPT_TESTS)
  test_spots(boost::math::concepts::real_concept(0.)); // Test real concept.
#endif
#else
   std::cout << "<note>The long double tests have been disabled on this platform "
      "either because the long double overloads of the usual math functions are "
      "not available at all, or because they are too inaccurate for these tests "
      "to pass.</note>" << std::endl;
#endif
   
} // BOOST_AUTO_TEST_CASE( test_main )


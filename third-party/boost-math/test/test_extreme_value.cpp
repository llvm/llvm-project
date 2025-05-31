// Copyright John Maddock 2006.
// Copyright Matt Borland 2024.
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// test_extreme_value.cpp

#include "../include_private/boost/math/tools/test.hpp"
#include <boost/math/concepts/real_concept.hpp> // for real_concept
#include <boost/math/distributions/extreme_value.hpp>
    using boost::math::extreme_value_distribution;

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // Boost.Test
#include <boost/test/tools/floating_point_comparison.hpp>
#include "test_out_of_range.hpp"

#include <iostream>
   using std::cout;
   using std::endl;
   using std::setprecision;
#include <type_traits>

template <class RealType>
void test_spot(RealType a, RealType b, RealType x, RealType p, RealType q, RealType logp, RealType logq, RealType tolerance, RealType logtolerance)
{
   BOOST_IF_CONSTEXPR (std::is_same<RealType, long double>::value || std::is_same<RealType, boost::math::concepts::real_concept>::value)
   {
      logtolerance *= 100;
   }
   
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         extreme_value_distribution<RealType>(a, b),      
         x),
         p,
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         complement(extreme_value_distribution<RealType>(a, b),      
         x)),
         q,
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::logcdf(
         extreme_value_distribution<RealType>(a, b),      
         x),
         logp,
         logtolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::logcdf(
         complement(extreme_value_distribution<RealType>(a, b),      
         x)),
         logq,
         logtolerance); // %
   if((p < 0.999) && (p > 0))
   {
      BOOST_CHECK_CLOSE(
         ::boost::math::quantile(
            extreme_value_distribution<RealType>(a, b),      
            p),
            x,
            tolerance); // %
   }
   if((q < 0.999) && (q > 0))
   {
      BOOST_CHECK_CLOSE(
         ::boost::math::quantile(
            complement(extreme_value_distribution<RealType>(a, b),      
            q)),
            x,
            tolerance); // %
   }
}

template <class RealType>
void test_spots(RealType)
{
   // Basic sanity checks.
   // 50eps as a percentage, up to a maximum of double precision
   // (that's the limit of our test data).
   RealType tolerance = (std::max)(
      static_cast<RealType>(boost::math::tools::epsilon<double>()),
      boost::math::tools::epsilon<RealType>());
   tolerance *= 50 * 100;  

   cout << "Tolerance for type " << typeid(RealType).name()  << " is " << tolerance << " %" << endl;

   // Results calculated by punching numbers into a calculator,
   // and using the formula at http://mathworld.wolfram.com/ExtremeValueDistribution.html
   test_spot(
      static_cast<RealType>(0.5), // a
      static_cast<RealType>(1.5), // b
      static_cast<RealType>(0.125), // x
      static_cast<RealType>(0.27692033409990891617007608217222L), // p
      static_cast<RealType>(0.72307966590009108382992391782778L), //q
      static_cast<RealType>(-1.2840254166877414840734205680624364583362808652814L), // Log(p)
      static_cast<RealType>(-0.324235874926689525622193916272L), // Log(q)
      tolerance,
      tolerance);
   test_spot(
      static_cast<RealType>(0.5), // a
      static_cast<RealType>(2), // b
      static_cast<RealType>(-5), // x
      static_cast<RealType>(1.6087601139887776413169427645933e-7L), // p
      static_cast<RealType>(0.99999983912398860112223586830572L), //q
      static_cast<RealType>(-15.6426318841881716102126980461566588450380350341076L), // Log(p)
      static_cast<RealType>(-1.60876024339424673820018469895e-7), // Log(q)
      tolerance,
      tolerance);
   test_spot(
      static_cast<RealType>(0.5), // a
      static_cast<RealType>(0.25), // b
      static_cast<RealType>(0.75), // x
      static_cast<RealType>(0.69220062755534635386542199718279L), // p
      static_cast<RealType>(0.30779937244465364613457800281721L), //q
      static_cast<RealType>(-0.36787944117144232159552377016146086744581113103177L), // Log(p)
      static_cast<RealType>(-1.17830709642071784241681100298L), // Log(q)
      tolerance,
      tolerance);
   // Edge case throws overflow exception in complement logcdf for float type
   BOOST_IF_CONSTEXPR (!std::is_same<RealType, float>::value)
   {  
      test_spot(
      static_cast<RealType>(0.5), // a
      static_cast<RealType>(0.25), // b
      static_cast<RealType>(5), // x
      static_cast<RealType>(0.99999998477002037126351248727041L), // p
      static_cast<RealType>(1.5229979628736487512729586276294e-8L), //q
      static_cast<RealType>(-1.52299797447126284361366292335174318621748e-8L), // Log(p)
      static_cast<RealType>(-18.0000000076149898626916357587L), // Log(q)
      tolerance,
      tolerance * 10000);
   }
   BOOST_CHECK_CLOSE(
      ::boost::math::pdf(
         extreme_value_distribution<RealType>(0.5, 2),      
         static_cast<RealType>(0.125)),              // x
         static_cast<RealType>(0.18052654830890205978204427757846L),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::pdf(
         extreme_value_distribution<RealType>(1, 3),      
         static_cast<RealType>(5)),              // x
         static_cast<RealType>(0.0675057324099851209129017326286L),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::pdf(
         extreme_value_distribution<RealType>(1, 3),      
         static_cast<RealType>(0)),              // x
         static_cast<RealType>(0.11522236828583456431277265757312L),                // probability.
         tolerance); // %

   BOOST_CHECK_CLOSE(
      ::boost::math::logpdf(
         extreme_value_distribution<RealType>(0.5, 2),      
         static_cast<RealType>(0.125)),              // x
         log(static_cast<RealType>(0.18052654830890205978204427757846L)),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::logpdf(
         extreme_value_distribution<RealType>(1, 3),      
         static_cast<RealType>(5)),              // x
         log(static_cast<RealType>(0.0675057324099851209129017326286L)),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::logpdf(
         extreme_value_distribution<RealType>(1, 3),      
         static_cast<RealType>(0)),              // x
         log(static_cast<RealType>(0.11522236828583456431277265757312L)),                // probability.
         tolerance); // %

   BOOST_CHECK_CLOSE(
      ::boost::math::mean(
         extreme_value_distribution<RealType>(2, 3)),
         static_cast<RealType>(3.731646994704598581819536270246L),           
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::standard_deviation(
         extreme_value_distribution<RealType>(1, 0.5)), 
         static_cast<RealType>(0.6412749150809320477720181798355L),
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::mode(
         extreme_value_distribution<RealType>(2, 3)),
         static_cast<RealType>(2),           
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::median(
         extreme_value_distribution<RealType>(0, 1)),
         static_cast<RealType>(+0.36651292058166432701243915823266946945426344783710526305367771367056),           
         tolerance); // %

   BOOST_CHECK_CLOSE(
      ::boost::math::skewness(
         extreme_value_distribution<RealType>(2, 3)),
         static_cast<RealType>(1.1395470994046486574927930193898461120875997958366L),           
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::kurtosis(
         extreme_value_distribution<RealType>(2, 3)),
         static_cast<RealType>(5.4),           
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::kurtosis_excess(
         extreme_value_distribution<RealType>(2, 3)),
         static_cast<RealType>(2.4),           
         tolerance); // %

   //
   // Things that are errors:
   //
   extreme_value_distribution<RealType> dist(0.5, 2);
   BOOST_MATH_CHECK_THROW(
       quantile(dist, RealType(1.0)),
       std::overflow_error);
   BOOST_MATH_CHECK_THROW(
       quantile(complement(dist, RealType(0.0))),
       std::overflow_error);
   BOOST_MATH_CHECK_THROW(
       quantile(dist, RealType(0.0)),
       std::overflow_error);
   BOOST_MATH_CHECK_THROW(
       quantile(complement(dist, RealType(1.0))),
       std::overflow_error);
   BOOST_MATH_CHECK_THROW(
       cdf(extreme_value_distribution<RealType>(0, -1), RealType(1)),
       std::domain_error);
   BOOST_MATH_CHECK_THROW(
       logcdf(extreme_value_distribution<RealType>(0, -1), RealType(1)),
       std::domain_error);
   BOOST_MATH_CHECK_THROW(
       quantile(dist, RealType(-1)),
       std::domain_error);
   BOOST_MATH_CHECK_THROW(
       quantile(dist, RealType(2)),
       std::domain_error);
   check_out_of_range<extreme_value_distribution<RealType> >(1, 2);
   if(std::numeric_limits<RealType>::has_infinity)
   {
      RealType inf = std::numeric_limits<RealType>::infinity();
      BOOST_CHECK_EQUAL(pdf(extreme_value_distribution<RealType>(), -inf), 0);
      BOOST_CHECK_EQUAL(pdf(extreme_value_distribution<RealType>(), inf), 0);
      BOOST_CHECK_EQUAL(cdf(extreme_value_distribution<RealType>(), -inf), 0);
      BOOST_CHECK_EQUAL(cdf(extreme_value_distribution<RealType>(), inf), 1);
      BOOST_CHECK_EQUAL(cdf(complement(extreme_value_distribution<RealType>(), -inf)), 1);
      BOOST_CHECK_EQUAL(cdf(complement(extreme_value_distribution<RealType>(), inf)), 0);
      BOOST_CHECK_EQUAL(logcdf(extreme_value_distribution<RealType>(), -inf), 0);
      BOOST_CHECK_EQUAL(logcdf(extreme_value_distribution<RealType>(), inf), 1);
   }
   //
   // Bug reports:
   //
   // https://svn.boost.org/trac/boost/ticket/10938:
   BOOST_CHECK_CLOSE(
   ::boost::math::pdf(
      extreme_value_distribution<RealType>(0, 1),
      static_cast<RealType>(-1000)),              // x
      static_cast<RealType>(0),                // probability.
      tolerance); // %

} // template <class RealType>void test_spots(RealType)

BOOST_AUTO_TEST_CASE( test_main )
{

  // Check that can generate extreme_value distribution using the two convenience methods:
   boost::math::extreme_value mycev1(1.); // Using typedef
   extreme_value_distribution<> myev2(1.); // Using default RealType double.

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

/*

Output is:

-Running 1 test case...
Tolerance for type float is 0.000596046 %
Tolerance for type double is 1.11022e-012 %
Tolerance for type long double is 1.11022e-012 %
Tolerance for type class boost::math::concepts::real_concept is 1.11022e-012 %
*** No errors detected
*/



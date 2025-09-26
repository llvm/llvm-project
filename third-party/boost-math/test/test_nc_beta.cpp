// test_nc_beta.cpp

// Copyright John Maddock 2008.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

//
// This must appear *before* any #includes, and precludes pch usage:
//
#define BOOST_MATH_ASSERT_UNDEFINED_POLICY false
#ifndef BOOST_MATH_OVERFLOW_ERROR_POLICY
#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error
#endif

#ifdef _MSC_VER
#pragma warning (disable:4127 4512)
#elif __GNUC__ >= 5
#  pragma GCC diagnostic ignored "-Woverflow"
#elif defined(__clang__)
#  pragma clang diagnostic ignored "-Wliteral-range"
#endif

#if !defined(TEST_FLOAT) && !defined(TEST_DOUBLE) && !defined(TEST_LDOUBLE) && !defined(TEST_REAL_CONCEPT)
#  define TEST_FLOAT
#  define TEST_DOUBLE
#  define TEST_LDOUBLE
#  define TEST_REAL_CONCEPT
#endif

#include <boost/math/tools/config.hpp>

#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
#include <boost/math/concepts/real_concept.hpp> // for real_concept
#endif

#include <boost/math/distributions/non_central_beta.hpp> // for chi_squared_distribution
#include <boost/math/distributions/poisson.hpp> // for poisson_distribution
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // for test_main
#include <boost/test/results_collector.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp> // for BOOST_CHECK_CLOSE

#include "functor.hpp"
#include "handle_test_result.hpp"
#include "test_ncbeta_hooks.hpp"
#include "table_type.hpp"
#include "test_nc_beta.hpp"
#include "../include_private/boost/math/tools/test.hpp"

#include <iostream>
using std::cout;
using std::endl;
#include <limits>
using std::numeric_limits;

void expected_results()
{
   //
   // Define the max and mean errors expected for
   // various compilers and platforms.
   //
   const char* largest_type;
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   if(boost::math::policies::digits<double, boost::math::policies::policy<> >() == boost::math::policies::digits<long double, boost::math::policies::policy<> >())
   {
      largest_type = "(long\\s+)?double|real_concept";
   }
   else
   {
      largest_type = "long double|real_concept";
   }
#else
   largest_type = "(long\\s+)?double|real_concept";
#endif

#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   if(boost::math::tools::digits<long double>() == 64)
   {
      //
      // Allow a small amount of error leakage from long double to double:
      //
      add_expected_result(
         "[^|]*",                          // compiler
         "[^|]*",                          // stdlib
         "[^|]*",                          // platform
         "double",                         // test type(s)
         "[^|]*large[^|]*",                // test data group
         "[^|]*", 5, 5);                   // test function
   }

   if(boost::math::tools::digits<long double>() == 64)
   {
      add_expected_result(
         "[^|]*",                          // compiler
         "[^|]*",                          // stdlib
         "[^|]*",                          // platform
         largest_type,                     // test type(s)
         "[^|]*medium[^|]*",               // test data group
         "[^|]*", 1200, 500);               // test function
      add_expected_result(
         "[^|]*",                          // compiler
         "[^|]*",                          // stdlib
         "[^|]*",                          // platform
         largest_type,                     // test type(s)
         "[^|]*large[^|]*",                // test data group
         "[^|]*", 40000, 6000);            // test function
   }
#endif
   //
   // Catch all cases come last:
   //
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      largest_type,                     // test type(s)
      "[^|]*medium[^|]*",               // test data group
      "[^|]*", 1500, 500);               // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      "real_concept",                   // test type(s)
      "[^|]*large[^|]*",                // test data group
      "[^|]*", 30000, 5000);             // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      largest_type,                     // test type(s)
      "[^|]*large[^|]*",                // test data group
      "[^|]*", 20000, 2000);             // test function
   //
   // Finish off by printing out the compiler/stdlib/platform names,
   // we do this to make it easier to mark up expected error rates.
   //
   std::cout << "Tests run with " << BOOST_COMPILER << ", " 
      << BOOST_STDLIB << ", " << BOOST_PLATFORM << std::endl;
}

template <class RealType>
RealType naive_pdf(RealType a, RealType b, RealType lam, RealType x)
{
   using namespace boost::math;

   RealType term = pdf(poisson_distribution<RealType>(lam/2), 0)
      * ibeta_derivative(a, b, x);
   RealType sum = term;

   int i = 1;
   while(term / sum > tools::epsilon<RealType>())
   {
      term = pdf(poisson_distribution<RealType>(lam/2), i)
      * ibeta_derivative(a + i, b, x);
      ++i;
      sum += term;
   }
   return sum;
}

template <class RealType>
void test_spot(
     RealType a,     // alpha
     RealType b,     // beta
     RealType ncp,   // non-centrality param
     RealType cs,    // Chi Square statistic
     RealType P,     // CDF
     RealType Q,     // Complement of CDF
     RealType D,     // PDF
     RealType tol)   // Test tolerance
{
   boost::math::non_central_beta_distribution<RealType> dist(a, b, ncp);
   BOOST_CHECK_CLOSE(
      cdf(dist, cs), P, tol);
   //
   // Sanity checking using the naive PDF calculation above fails at
   // float precision:
   //
   if(!boost::is_same<float, RealType>::value)
   {
      BOOST_CHECK_CLOSE(
         pdf(dist, cs), naive_pdf(dist.alpha(), dist.beta(), ncp, cs), tol);
   }
   BOOST_CHECK_CLOSE(
      pdf(dist, cs), D, tol);

   if((P < 0.99) && (Q < 0.99))
   {
      //
      // We can only check this if P is not too close to 1,
      // so that we can guarantee Q is reasonably free of error:
      //
      BOOST_CHECK_CLOSE(
         cdf(complement(dist, cs)), Q, tol);
      BOOST_CHECK_CLOSE(
            quantile(dist, P), cs, tol * 10);
      BOOST_CHECK_CLOSE(
            quantile(complement(dist, Q)), cs, tol * 10);
   }
}

template <class RealType> // Any floating-point type RealType.
void test_spots(RealType)
{
   RealType tolerance = (std::max)(
      boost::math::tools::epsilon<RealType>() * 100,
      (RealType)1e-6) * 100;
   RealType abs_tolerance = boost::math::tools::epsilon<RealType>() * 100;

   cout << "Tolerance = " << tolerance << "%." << endl;

   //
   // Spot tests use values computed by the R statistical
   // package and the pbeta and dbeta functions:
   //
   test_spot(
     RealType(2),                   // alpha
     RealType(5),                   // beta
     RealType(1),                   // non-centrality param
     RealType(0.25),                // Chi Square statistic
     RealType(0.3658349),           // CDF
     RealType(1-0.3658349),         // Complement of CDF
     RealType(2.184465),            // PDF
     RealType(tolerance));
   test_spot(
     RealType(20),                  // alpha
     RealType(15),                  // beta
     RealType(35),                  // non-centrality param
     RealType(0.75),                // Chi Square statistic
     RealType(0.6994175),           // CDF
     RealType(1-0.6994175),         // Complement of CDF
     RealType(5.576146),            // PDF
     RealType(tolerance));
   test_spot(
     RealType(100),                 // alpha
     RealType(3),                   // beta
     RealType(63),                  // non-centrality param
     RealType(0.95),                // Chi Square statistic
     RealType(0.03529306),          // CDF
     RealType(1-0.03529306),        // Complement of CDF
     RealType(3.637894),            // PDF
     RealType(tolerance));
   test_spot(
     RealType(0.25),                // alpha
     RealType(0.75),                // beta
     RealType(150),                 // non-centrality param
     RealType(0.975),               // Chi Square statistic
     RealType(0.09752216),          // CDF
     RealType(1-0.09752216),        // Complement of CDF
     RealType(8.020935),            // PDF
     RealType(tolerance));

   BOOST_MATH_STD_USING
   boost::math::non_central_beta_distribution<RealType> dist(100, 3, 63);
   BOOST_CHECK_CLOSE(mean(dist), RealType(4.82280451915522329944315287538684030781836554279474240490936e13L) * exp(-RealType(31.5)) * 100 / 103, tolerance);
   // Variance only guarantees small absolute error:
   BOOST_CHECK_SMALL(variance(dist) 
      - static_cast<RealType>(RealType(4.85592267707818899235900237275021938334418424134218087127572e13L)
      * exp(RealType(-31.5)) * 100 * 101 / (103 * 104) - 
      RealType(4.82280451915522329944315287538684030781836554279474240490936e13L) * RealType(4.82280451915522329944315287538684030781836554279474240490936e13L) 
      * exp(RealType(-63)) * 10000 / (103 * 103)), abs_tolerance);
   BOOST_MATH_CHECK_THROW(skewness(dist), boost::math::evaluation_error);
   BOOST_MATH_CHECK_THROW(kurtosis(dist), boost::math::evaluation_error);
   BOOST_MATH_CHECK_THROW(kurtosis_excess(dist), boost::math::evaluation_error);
   //
   // Some special error handling tests, if the non-centrality param is too large
   // then we have no evaluation method and should get a domain_error:
   //
   using std::ldexp;
   using distro1 = boost::math::non_central_beta_distribution<RealType>;
   using distro2 = boost::math::non_central_beta_distribution<RealType, boost::math::policies::policy<boost::math::policies::domain_error<boost::math::policies::ignore_error>>>;
   using de = std::domain_error;
   BOOST_MATH_CHECK_THROW(distro1(2, 3, ldexp(RealType(1), 100)), de);
   if (std::numeric_limits<RealType>::has_quiet_NaN)
   {
      distro2 d2(2, 3, ldexp(RealType(1), 100));
      BOOST_CHECK(boost::math::isnan(pdf(d2, 0.5)));
      BOOST_CHECK(boost::math::isnan(cdf(d2, 0.5)));
      BOOST_CHECK(boost::math::isnan(cdf(complement(d2, 0.5))));
   }
} // template <class RealType>void test_spots(RealType)


BOOST_AUTO_TEST_CASE( test_main )
{
   BOOST_MATH_CONTROL_FP;
   // Basic sanity-check spot values.
    expected_results();
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
#ifdef TEST_REAL_CONCEPT
   test_spots(boost::math::concepts::real_concept(0.)); // Test real concept.
#endif
#endif
#endif

#ifdef TEST_FLOAT
   test_accuracy(0.0F, "float"); // Test float.
#endif
#ifdef TEST_DOUBLE
   test_accuracy(0.0, "double"); // Test double.
#endif
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
#ifdef TEST_LDOUBLE
   test_accuracy(0.0L, "long double"); // Test long double.
#endif
#if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x582)) && !defined(BOOST_MATH_NO_REAL_CONCEPT_TESTS)
#ifdef TEST_REAL_CONCEPT
   test_accuracy(boost::math::concepts::real_concept(0.), "real_concept"); // Test real concept.
#endif
#endif
#endif
   
} // BOOST_AUTO_TEST_CASE( test_main )


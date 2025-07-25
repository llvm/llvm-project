// Copyright John Maddock 2006.
// Copyright Paul A. Bristow 2007.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// test_dist_overloads.cpp

#include <boost/math/concepts/real_concept.hpp> // for real_concept
#include <boost/math/distributions/normal.hpp>
    using boost::math::normal_distribution;

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // Boost.Test
#include <boost/test/tools/floating_point_comparison.hpp>

#include <iostream>
   using std::cout;
   using std::endl;
   using std::setprecision;

template <class RealType>
void test_spots(RealType)
{
   // Basic sanity checks,
   // 2 eps as a percentage:
   RealType tolerance = boost::math::tools::epsilon<RealType>() * 2 * 100;

   cout << "Tolerance for type " << typeid(RealType).name()  << " is " << tolerance << " %" << endl;

   for(int i = -4; i <= 4; ++i)
   {
      BOOST_CHECK_CLOSE(
         ::boost::math::cdf(normal_distribution<RealType>(), i),
         ::boost::math::cdf(normal_distribution<RealType>(), static_cast<RealType>(i)),
         tolerance);
      BOOST_CHECK_CLOSE(
         ::boost::math::pdf(normal_distribution<RealType>(), i),
         ::boost::math::pdf(normal_distribution<RealType>(), static_cast<RealType>(i)),
         tolerance);
      BOOST_CHECK_CLOSE(
         ::boost::math::cdf(complement(normal_distribution<RealType>(), i)),
         ::boost::math::cdf(complement(normal_distribution<RealType>(), static_cast<RealType>(i))),
         tolerance);
      BOOST_CHECK_CLOSE(
         ::boost::math::hazard(normal_distribution<RealType>(), i),
         ::boost::math::hazard(normal_distribution<RealType>(), static_cast<RealType>(i)),
         tolerance);
      BOOST_CHECK_CLOSE(
         ::boost::math::chf(normal_distribution<RealType>(), i),
         ::boost::math::chf(normal_distribution<RealType>(), static_cast<RealType>(i)),
         tolerance);
   }
   for(float f = 0.01f; f < 1; f += 0.01f)
   {
      BOOST_CHECK_CLOSE(
         ::boost::math::quantile(normal_distribution<RealType>(), f),
         ::boost::math::quantile(normal_distribution<RealType>(), static_cast<RealType>(f)),
         tolerance);
      BOOST_CHECK_CLOSE(
         ::boost::math::quantile(complement(normal_distribution<RealType>(), f)),
         ::boost::math::quantile(complement(normal_distribution<RealType>(), static_cast<RealType>(f))),
         tolerance);
   }
} // template <class RealType>void test_spots(RealType)

BOOST_AUTO_TEST_CASE( test_main )
{
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

Output:

Running 1 test case...
Tolerance for type float is 2.38419e-005 %
Tolerance for type double is 4.44089e-014 %
Tolerance for type long double is 4.44089e-014 %
Tolerance for type class boost::math::concepts::real_concept is 4.44089e-014 %
*** No errors detected

*/


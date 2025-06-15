// Copyright John Maddock 2010
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/tools/config.hpp>
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS

#include <boost/math/concepts/real_concept.hpp> // for real_concept
#include <boost/math/constants/constants.hpp>

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // Boost.Test
#include <boost/test/results_collector.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <iostream>
#include <iomanip>


BOOST_AUTO_TEST_CASE( test_main )
{
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS

   typedef boost::math::concepts::real_concept rc_t;

   rc_t r1(2.5), r2(0.125), r3(45.5);
   long double               l1(2.5), l2(0.125), l3(45.5);
   long double tol = std::numeric_limits<long double>::epsilon() * 2;

   {
      rc_t t(r1);
      long double   t2(l1);
      t += r2;
      t2 += l2;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t += l2;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t += 0.125;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t += 0.125f;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);

      t = r1;
      t += 23L;
      t2 = 23 + l1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t += 23uL;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t += 23;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t += 23u;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t += static_cast<short>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t += static_cast<unsigned short>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t += static_cast<char>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t += static_cast<signed char>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t += static_cast<unsigned char>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
   }

   {
      rc_t t(r1);
      long double   t2(l1);
      t -= r2;
      t2 -= l2;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t -= l2;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t -= 0.125;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t -= 0.125f;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);

      t = r1;
      t -= 23L;
      t2 = l1 - 23;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t -= 23uL;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t -= 23;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t -= 23u;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t -= static_cast<short>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t -= static_cast<unsigned short>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t -= static_cast<char>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t -= static_cast<signed char>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t -= static_cast<unsigned char>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
   }

   {
      rc_t t(r1);
      long double   t2(l1);
      t *= r2;
      t2 *= l2;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t *= l2;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t *= 0.125;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t *= 0.125f;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);

      t = r1;
      t *= 23L;
      t2 = 23 * l1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t *= 23uL;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t *= 23;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t *= 23u;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t *= static_cast<short>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t *= static_cast<unsigned short>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t *= static_cast<char>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t *= static_cast<signed char>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t *= static_cast<unsigned char>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
   }
   {
      rc_t t(r1);
      long double   t2(l1);
      t /= r2;
      t2 /= l2;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t /= l2;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t /= 0.125;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t /= 0.125f;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);

      t = r1;
      t /= 23L;
      t2 = l1 / 23;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t /= 23uL;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t /= 23;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t /= 23u;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t /= static_cast<short>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t /= static_cast<unsigned short>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t /= static_cast<char>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t /= static_cast<signed char>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1;
      t /= static_cast<unsigned char>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
   }

   {
      rc_t t;
      long double t2;
      t = r1 + r2;
      t2 = l1 + l2;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 + l2;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 + 0.125;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 + 0.125f;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = l2 + r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = 0.125 + r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = 0.125f + r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);

      t2 = l1 + 23L;
      t = r1 + 23L;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 + 23uL;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 + 23;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 + 23u;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 + static_cast<short>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 + static_cast<unsigned short>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 + static_cast<char>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 + static_cast<signed char>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 + static_cast<unsigned char>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);

      t = 23L + r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = 23uL + r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = 23 + r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = 23u + r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = static_cast<short>(23) + r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = static_cast<unsigned short>(23) + r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = static_cast<char>(23) + r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = static_cast<signed char>(23) + r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = static_cast<unsigned char>(23) + r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
   }
   {
      rc_t t;
      long double t2;
      t = r1 - r2;
      t2 = l1 - l2;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 - l2;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 - 0.125;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 - 0.125f;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t2 = l2 - l1;
      t = l2 - r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = 0.125 - r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = 0.125f - r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);

      t2 = l1 - 23L;
      t = r1 - 23L;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 - 23uL;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 - 23;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 - 23u;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 - static_cast<short>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 - static_cast<unsigned short>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 - static_cast<char>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 - static_cast<signed char>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 - static_cast<unsigned char>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);

      t2 = 23L - l1;
      t = 23L - r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = 23uL - r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = 23 - r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = 23u - r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = static_cast<short>(23) - r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = static_cast<unsigned short>(23) - r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = static_cast<char>(23) - r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = static_cast<signed char>(23) - r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = static_cast<unsigned char>(23) - r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
   }
   {
      rc_t t;
      long double t2;
      t = r1 * r2;
      t2 = l1 * l2;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 * l2;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 * 0.125;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 * 0.125f;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = l2 * r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = 0.125 * r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = 0.125f * r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);

      t2 = l1 * 23L;
      t = r1 * 23L;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 * 23uL;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 * 23;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 * 23u;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 * static_cast<short>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 * static_cast<unsigned short>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 * static_cast<char>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 * static_cast<signed char>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 * static_cast<unsigned char>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);

      t = 23L * r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = 23uL * r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = 23 * r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = 23u * r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = static_cast<short>(23) * r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = static_cast<unsigned short>(23) * r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = static_cast<char>(23) * r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = static_cast<signed char>(23) * r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = static_cast<unsigned char>(23) * r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
   }
   {
      rc_t t;
      long double t2;
      t = r1 / r2;
      t2 = l1 / l2;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 / l2;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 / 0.125;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 / 0.125f;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t2 = l2 / l1;
      t = l2 / r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = 0.125 / r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = 0.125f / r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);

      t2 = l1 / 23L;
      t = r1 / 23L;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 / 23uL;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 / 23;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 / 23u;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 / static_cast<short>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 / static_cast<unsigned short>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 / static_cast<char>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 / static_cast<signed char>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = r1 / static_cast<unsigned char>(23);
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);

      t2 = 23L / l1;
      t = 23L / r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = 23uL / r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = 23 / r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = 23u / r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = static_cast<short>(23) / r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = static_cast<unsigned short>(23) / r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = static_cast<char>(23) / r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = static_cast<signed char>(23) / r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
      t = static_cast<unsigned char>(23) / r1;
      BOOST_CHECK_CLOSE_FRACTION(t.value(), t2, tol);
   }

   {
      BOOST_CHECK_EQUAL(r1 == r2, l1 == l2);
      BOOST_CHECK_EQUAL(r1 == l2, l1 == l2);
      BOOST_CHECK_EQUAL(r1 == 0.125, l1 == l2);
      BOOST_CHECK_EQUAL(r1 == 0.125f, l1 == l2);
      BOOST_CHECK_EQUAL(l1 == r2, l1 == l2);
      BOOST_CHECK_EQUAL(2.5 == r2, l1 == l2);
      BOOST_CHECK_EQUAL(2.5f == r2, l1 == l2);

      BOOST_CHECK_EQUAL(r1 <= r2, l1 <= l2);
      BOOST_CHECK_EQUAL(r1 <= l2, l1 <= l2);
      BOOST_CHECK_EQUAL(r1 <= 0.125, l1 <= l2);
      BOOST_CHECK_EQUAL(r1 <= 0.125f, l1 <= l2);
      BOOST_CHECK_EQUAL(l1 <= r2, l1 <= l2);
      BOOST_CHECK_EQUAL(2.5 <= r2, l1 <= l2);
      BOOST_CHECK_EQUAL(2.5f <= r2, l1 <= l2);

      BOOST_CHECK_EQUAL(r1 >= r2, l1 >= l2);
      BOOST_CHECK_EQUAL(r1 >= l2, l1 >= l2);
      BOOST_CHECK_EQUAL(r1 >= 0.125, l1 >= l2);
      BOOST_CHECK_EQUAL(r1 >= 0.125f, l1 >= l2);
      BOOST_CHECK_EQUAL(l1 >= r2, l1 >= l2);
      BOOST_CHECK_EQUAL(2.5 >= r2, l1 >= l2);
      BOOST_CHECK_EQUAL(2.5f >= r2, l1 >= l2);

      BOOST_CHECK_EQUAL(r1 < r2, l1 < l2);
      BOOST_CHECK_EQUAL(r1 < l2, l1 < l2);
      BOOST_CHECK_EQUAL(r1 < 0.125, l1 < l2);
      BOOST_CHECK_EQUAL(r1 < 0.125f, l1 < l2);
      BOOST_CHECK_EQUAL(l1 < r2, l1 < l2);
      BOOST_CHECK_EQUAL(2.5 < r2, l1 < l2);
      BOOST_CHECK_EQUAL(2.5f < r2, l1 < l2);

      BOOST_CHECK_EQUAL(r1 > r2, l1 > l2);
      BOOST_CHECK_EQUAL(r1 > l2, l1 > l2);
      BOOST_CHECK_EQUAL(r1 > 0.125, l1 > l2);
      BOOST_CHECK_EQUAL(r1 > 0.125f, l1 > l2);
      BOOST_CHECK_EQUAL(l1 > r2, l1 > l2);
      BOOST_CHECK_EQUAL(2.5 > r2, l1 > l2);
      BOOST_CHECK_EQUAL(2.5f > r2, l1 > l2);
   }

   {
      BOOST_MATH_STD_USING
      BOOST_CHECK_CLOSE_FRACTION(acos(r2), acos(l2), tol);
      BOOST_CHECK_CLOSE_FRACTION(cos(r2), cos(l2), tol);
      BOOST_CHECK_CLOSE_FRACTION(asin(r2), asin(l2), tol);
      BOOST_CHECK_CLOSE_FRACTION(atan(r2), atan(l2), tol);
      BOOST_CHECK_CLOSE_FRACTION(atan2(r2, r3), atan2(l2, l3), tol);
      BOOST_CHECK_CLOSE_FRACTION(ceil(r2), ceil(l2), tol);
      BOOST_CHECK_CLOSE_FRACTION(fmod(r2, r3), fmod(l2, l3), tol);
      BOOST_CHECK_CLOSE_FRACTION(cosh(r2), cosh(l2), tol);
      BOOST_CHECK_CLOSE_FRACTION(exp(r2), exp(l2), tol);
      BOOST_CHECK_CLOSE_FRACTION(fabs(r2), fabs(l2), tol);
      BOOST_CHECK_CLOSE_FRACTION(abs(r2), abs(l2), tol);
      rc_t rc_result;
      long double ld_result;
#ifdef __MINGW32__
      BOOST_CHECK_CLOSE_FRACTION(modf(r2, &rc_result), boost::math::modf(l2, &ld_result), tol);
#else
      BOOST_CHECK_CLOSE_FRACTION(modf(r2, &rc_result), modf(l2, &ld_result), tol);
#endif
      BOOST_CHECK_CLOSE_FRACTION(rc_result, ld_result, tol);
      int i1, i2;
      BOOST_CHECK_CLOSE_FRACTION(frexp(r3, &i1), frexp(l3, &i2), tol);
      BOOST_CHECK_EQUAL(i1, i2);
      BOOST_CHECK_CLOSE_FRACTION(ldexp(r3, i1), ldexp(l3, i1), tol);
      BOOST_CHECK_CLOSE_FRACTION(log(r2), log(l2), tol);
      BOOST_CHECK_CLOSE_FRACTION(log10(r2), log10(l2), tol);
      BOOST_CHECK_CLOSE_FRACTION(tan(r2), tan(l2), tol);
      BOOST_CHECK_CLOSE_FRACTION(pow(r2, r3), pow(l2, l3), tol);
      BOOST_CHECK_CLOSE_FRACTION(pow(r2, i1), pow(l2, i1), tol);
      BOOST_CHECK_CLOSE_FRACTION(sin(r2), sin(l2), tol);
      BOOST_CHECK_CLOSE_FRACTION(sinh(r2), sinh(l2), tol);
      BOOST_CHECK_CLOSE_FRACTION(sqrt(r2), sqrt(l2), tol);
      BOOST_CHECK_CLOSE_FRACTION(tanh(r2), tanh(l2), tol);

      BOOST_CHECK_EQUAL(iround(r2), boost::math::iround(l2));
      BOOST_CHECK_EQUAL(lround(r2), boost::math::lround(l2));
#ifdef BOOST_HAS_LONG_LONG
      BOOST_CHECK_EQUAL(llround(r2), boost::math::llround(l2));
#endif
      BOOST_CHECK_EQUAL(itrunc(r2), boost::math::itrunc(l2));
      BOOST_CHECK_EQUAL(ltrunc(r2), boost::math::ltrunc(l2));
#ifdef BOOST_HAS_LONG_LONG
      BOOST_CHECK_EQUAL(lltrunc(r2), boost::math::lltrunc(l2));
#endif
   }

   {
      using namespace boost::math::tools;
      tol = std::numeric_limits<long double>::epsilon();
      BOOST_CHECK_CLOSE_FRACTION(max_value<rc_t>(), max_value<long double>(), tol);
      BOOST_CHECK_CLOSE_FRACTION(min_value<rc_t>(), min_value<long double>(), tol);
      BOOST_CHECK_CLOSE_FRACTION(log_max_value<rc_t>(), log_max_value<long double>(), tol);
      BOOST_CHECK_CLOSE_FRACTION(log_min_value<rc_t>(), log_min_value<long double>(), tol);
      BOOST_CHECK_CLOSE_FRACTION(epsilon<rc_t>(), epsilon<long double>(), tol);
      BOOST_CHECK_EQUAL(digits<rc_t>(), digits<long double>());
   }

   {
      using namespace boost::math::constants;
      BOOST_CHECK_CLOSE_FRACTION(pi<rc_t>(), pi<long double>(), tol);
      BOOST_CHECK_CLOSE_FRACTION(root_pi<rc_t>(), root_pi<long double>(), tol);
      BOOST_CHECK_CLOSE_FRACTION(root_half_pi<rc_t>(), root_half_pi<long double>(), tol);
      BOOST_CHECK_CLOSE_FRACTION(root_two_pi<rc_t>(), root_two_pi<long double>(), tol);
      BOOST_CHECK_CLOSE_FRACTION(root_ln_four<rc_t>(), root_ln_four<long double>(), tol);
      BOOST_CHECK_CLOSE_FRACTION(half<rc_t>(), half<long double>(), tol);
      BOOST_CHECK_CLOSE_FRACTION(euler<rc_t>(), euler<long double>(), tol);
      BOOST_CHECK_CLOSE_FRACTION(root_two<rc_t>(), root_two<long double>(), tol);
      BOOST_CHECK_CLOSE_FRACTION(ln_two<rc_t>(), ln_two<long double>(), tol);
      BOOST_CHECK_CLOSE_FRACTION(ln_ln_two<rc_t>(), ln_ln_two<long double>(), tol);
      BOOST_CHECK_CLOSE_FRACTION(third<rc_t>(), third<long double>(), tol);
      BOOST_CHECK_CLOSE_FRACTION(twothirds<rc_t>(), twothirds<long double>(), tol);
      BOOST_CHECK_CLOSE_FRACTION(pi_minus_three<rc_t>(), pi_minus_three<long double>(), tol);
      BOOST_CHECK_CLOSE_FRACTION(four_minus_pi<rc_t>(), four_minus_pi<long double>(), tol);
 //     BOOST_CHECK_CLOSE_FRACTION(pow23_four_minus_pi<rc_t>(), pow23_four_minus_pi<long double>(), tol);
      BOOST_CHECK_CLOSE_FRACTION(exp_minus_half<rc_t>(), exp_minus_half<long double>(), tol);
   }

#else
   std::cout << "<note>The long double tests have been disabled on this platform "
      "either because the long double overloads of the usual math functions are "
      "not available at all, or because they are too inaccurate for these tests "
      "to pass.</note>" << std::endl;
#endif

   
} // BOOST_AUTO_TEST_CASE( test_main )

#else
int main(void) { return 0; }
#endif // BOOST_MATH_NO_REAL_CONCEPT_TESTS

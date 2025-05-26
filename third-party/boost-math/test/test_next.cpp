//  (C) Copyright John Maddock 2008.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pch.hpp>

#include <boost/math/concepts/real_concept.hpp>
#include <boost/math/tools/test.hpp>
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/special_functions/next.hpp>
#include <boost/math/special_functions/ulp.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <iostream>
#include <iomanip>

#ifdef _MSC_VER
#pragma warning(disable:4127)
#endif

#if !defined(_CRAYC) && !defined(__CUDACC__) && (!defined(__GNUC__) || (__GNUC__ > 3) || ((__GNUC__ == 3) && (__GNUC_MINOR__ > 3)))
#if (defined(_M_IX86_FP) && (_M_IX86_FP >= 2)) || defined(__SSE2__) || defined(TEST_SSE2)
#include <cfloat>
#include "xmmintrin.h"
#define TEST_SSE2
#endif
#endif


template <class T>
void test_value(const T& val, const char* name)
{
   using namespace boost::math;
   T upper = tools::max_value<T>();
   T lower = -upper;

   std::cout << "Testing type " << name << " with initial value " << val << std::endl;

   BOOST_CHECK_EQUAL(float_distance(float_next(val), val), -1);
   BOOST_CHECK(float_next(val) > val);
   BOOST_CHECK_EQUAL(float_distance(float_prior(val), val), 1);
   BOOST_CHECK(float_prior(val) < val);
   BOOST_CHECK_EQUAL(float_distance((boost::math::nextafter)(val, upper), val), -1);
   BOOST_CHECK((boost::math::nextafter)(val, upper) > val);
   BOOST_CHECK_EQUAL(float_distance((boost::math::nextafter)(val, lower), val), 1);
   BOOST_CHECK((boost::math::nextafter)(val, lower) < val);
   BOOST_CHECK_EQUAL(float_distance(float_next(float_next(val)), val), -2);
   BOOST_CHECK_EQUAL(float_distance(float_prior(float_prior(val)), val), 2);
   BOOST_CHECK_EQUAL(float_distance(float_prior(float_prior(val)), float_next(float_next(val))), 4);
   BOOST_CHECK_EQUAL(float_distance(float_prior(float_next(val)), val), 0);
   BOOST_CHECK_EQUAL(float_distance(float_next(float_prior(val)), val), 0);
   BOOST_CHECK_EQUAL(float_prior(float_next(val)), val);
   BOOST_CHECK_EQUAL(float_next(float_prior(val)), val);

   BOOST_CHECK_EQUAL(float_distance(float_advance(val, 4), val), -4);
   BOOST_CHECK_EQUAL(float_distance(float_advance(val, -4), val), 4);
   if(std::numeric_limits<T>::is_specialized && boost::math::detail::has_denorm_now<T>())
   {
      BOOST_CHECK_EQUAL(float_distance(float_advance(float_next(float_next(val)), 4), float_next(float_next(val))), -4);
      BOOST_CHECK_EQUAL(float_distance(float_advance(float_next(float_next(val)), -4), float_next(float_next(val))), 4);
   }
   if(val > 0)
   {
      T n = val + ulp(val);
      T fn = float_next(val);
      if(n > fn)
      {
         BOOST_CHECK_LE(ulp(val), boost::math::tools::min_value<T>());
      }
      else
      {
         BOOST_CHECK_EQUAL(fn, n);
      }
   }
   else if(val == 0)
   {
      BOOST_CHECK_GE(boost::math::tools::min_value<T>(), ulp(val));
   }
   else
   {
      T n = val - ulp(val);
      T fp = float_prior(val);
      if(n < fp)
      {
         BOOST_CHECK_LE(ulp(val), boost::math::tools::min_value<T>());
      }
      else
      {
         BOOST_CHECK_EQUAL(fp, n);
      }
   }
}

template <class T>
void test_values(const T& val, const char* name)
{
   static const T a = static_cast<T>(1.3456724e22);
   static const T b = static_cast<T>(1.3456724e-22);
   static const T z = 0;
   static const T one = 1;
   static const T two = 2;

   std::cout << "Testing type " << name << std::endl;

   T den = (std::numeric_limits<T>::min)() / 4;
   if(den != 0)
   {
      std::cout << "Denormals are active\n";
   }
   else
   {
      std::cout << "Denormals are flushed to zero.\n";
   }

   test_value(a, name);
   test_value(-a, name);
   test_value(b, name);
   test_value(-b, name);
   test_value(boost::math::tools::epsilon<T>(), name);
   test_value(-boost::math::tools::epsilon<T>(), name);
   test_value(boost::math::tools::min_value<T>(), name);
   test_value(-boost::math::tools::min_value<T>(), name);
   if (std::numeric_limits<T>::is_specialized && boost::math::detail::has_denorm_now<T>() && ((std::numeric_limits<T>::min)() / 2 != 0))
   {
      test_value(z, name);
      test_value(-z, name);
   }
   test_value(one, name);
   test_value(-one, name);
   test_value(two, name);
   test_value(-two, name);
#if defined(TEST_SSE2)
   if((_mm_getcsr() & (_MM_FLUSH_ZERO_ON | 0x40)) == 0)
   {
#endif
      if(std::numeric_limits<T>::is_specialized && boost::math::detail::has_denorm_now<T>() && ((std::numeric_limits<T>::min)() / 2 != 0))
      {
         test_value(std::numeric_limits<T>::denorm_min(), name);
         test_value(-std::numeric_limits<T>::denorm_min(), name);
         test_value(2 * std::numeric_limits<T>::denorm_min(), name);
         test_value(-2 * std::numeric_limits<T>::denorm_min(), name);
      }
#if defined(TEST_SSE2)
   }
#endif
   static const int primes[] = {
      11,     13,     17,     19,     23,     29, 
      31,     37,     41,     43,     47,     53,     59,     61,     67,     71, 
      73,     79,     83,     89,     97,    101,    103,    107,    109,    113, 
      127,    131,    137,    139,    149,    151,    157,    163,    167,    173, 
      179,    181,    191,    193,    197,    199,    211,    223,    227,    229, 
      233,    239,    241,    251,    257,    263,    269,    271,    277,    281, 
      283,    293,    307,    311,    313,    317,    331,    337,    347,    349, 
      353,    359,    367,    373,    379,    383,    389,    397,    401,    409, 
      419,    421,    431,    433,    439,    443,    449,    457,    461,    463, 
   };

   for(unsigned i = 0; i < sizeof(primes)/sizeof(primes[0]); ++i)
   {
      T v1 = val;
      T v2 = val;
      for(int j = 0; j < primes[i]; ++j)
      {
         v1 = boost::math::float_next(v1);
         v2 = boost::math::float_prior(v2);
      }
      BOOST_CHECK_EQUAL(boost::math::float_distance(v1, val), -primes[i]);
      BOOST_CHECK_EQUAL(boost::math::float_distance(v2, val), primes[i]);
      BOOST_CHECK_EQUAL(boost::math::float_advance(val, primes[i]), v1);
      BOOST_CHECK_EQUAL(boost::math::float_advance(val, -primes[i]), v2);
   }
   BOOST_IF_CONSTEXPR(std::numeric_limits<T>::is_specialized && (std::numeric_limits<T>::has_infinity))
   {
      BOOST_CHECK_EQUAL(boost::math::float_prior(std::numeric_limits<T>::infinity()), (std::numeric_limits<T>::max)());
      BOOST_CHECK_EQUAL(boost::math::float_next(-std::numeric_limits<T>::infinity()), -(std::numeric_limits<T>::max)());
      BOOST_CHECK_EQUAL(boost::math::float_prior(-std::numeric_limits<T>::infinity()), -std::numeric_limits<T>::infinity());
      BOOST_CHECK_EQUAL(boost::math::float_next(std::numeric_limits<T>::infinity()), std::numeric_limits<T>::infinity());
      if(boost::math::policies:: BOOST_MATH_OVERFLOW_ERROR_POLICY == boost::math::policies::throw_on_error)
      {
         BOOST_MATH_CHECK_THROW(boost::math::float_prior(-(std::numeric_limits<T>::max)()), std::overflow_error);
         BOOST_MATH_CHECK_THROW(boost::math::float_next((std::numeric_limits<T>::max)()), std::overflow_error);
      }
      else
      {
         BOOST_CHECK_EQUAL(boost::math::float_prior(-(std::numeric_limits<T>::max)()), -std::numeric_limits<T>::infinity());
         BOOST_CHECK_EQUAL(boost::math::float_next((std::numeric_limits<T>::max)()), std::numeric_limits<T>::infinity());
      }
   }
   BOOST_IF_CONSTEXPR(std::numeric_limits<T>::is_specialized && (std::numeric_limits<T>::has_quiet_NaN))
   {
      BOOST_MATH_CHECK_THROW(boost::math::float_prior((std::numeric_limits<T>::quiet_NaN)()), std::domain_error);
      BOOST_MATH_CHECK_THROW(boost::math::float_next((std::numeric_limits<T>::quiet_NaN)()), std::domain_error);
   }
   //
   // We need to test float_distance over multiple orders of magnitude,
   // the only way to get an accurate true result is to count the representations
   // between the two end points, but we can only really do this for type float:
   //
   if (std::numeric_limits<T>::is_specialized && (std::numeric_limits<T>::digits < 30) && (std::numeric_limits<T>::radix == 2))
   {
      T left, right, dist, fresult;
      std::uintmax_t result;

      left = static_cast<T>(0.1);
      right = left * static_cast<T>(4.2);
      dist = boost::math::float_distance(left, right);
      // We have to use a wider integer type for the accurate count, since there
      // aren't enough bits in T to get a true result if the values differ
      // by more than a factor of 2:
      result = 0;
      for (; left != right; ++result, left = boost::math::float_next(left));
      fresult = static_cast<T>(result);
      BOOST_CHECK_EQUAL(fresult, dist);

      left = static_cast<T>(-0.1);
      right = left * static_cast<T>(4.2);
      dist = boost::math::float_distance(right, left);
      result = 0;
      for (; left != right; ++result, left = boost::math::float_prior(left));
      fresult = static_cast<T>(result);
      BOOST_CHECK_EQUAL(fresult, dist);

      left = static_cast<T>(-1.1) * (std::numeric_limits<T>::min)();
      right = static_cast<T>(-4.1) * left;
      dist = boost::math::float_distance(left, right);
      result = 0;
      for (; left != right; ++result, left = boost::math::float_next(left));
      fresult = static_cast<T>(result);
      BOOST_CHECK_EQUAL(fresult, dist);
   }
}

BOOST_AUTO_TEST_CASE( test_main )
{
   test_values(1.0f, "float");
   test_values(1.0, "double");
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   test_values(1.0L, "long double");

   // MSVC-14.3 fails with real concept on Github Actions, but the failure cannot be reproduced locally
   // See: https://github.com/boostorg/math/pull/720
   #if !defined(_MSC_VER) || _MSC_VER < 1930
   test_values(boost::math::concepts::real_concept(0), "real_concept");
   #endif
#endif

   //
   // Test some multiprecision types:
   //
   test_values(boost::multiprecision::cpp_bin_float_quad(0), "cpp_bin_float_quad");
   // This is way to slow to test routinely:
   //test_values(boost::multiprecision::cpp_bin_float_single(0), "cpp_bin_float_single");
   test_values(boost::multiprecision::cpp_bin_float_50(0), "cpp_bin_float_50");

#if defined(TEST_SSE2)

   int mmx_flags = _mm_getcsr(); // We'll restore these later.

#ifdef _WIN32
   // These tests fail pretty badly on Linux x64, especially with Intel-12.1
   _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
   std::cout << "Testing again with Flush-To-Zero set" << std::endl;
   std::cout << "SSE2 control word is: " << std::hex << _mm_getcsr() << std::endl;
   test_values(1.0f, "float");
   test_values(1.0, "double");
   _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
#endif
   BOOST_MATH_ASSERT((_mm_getcsr() & 0x40) == 0);
   _mm_setcsr(_mm_getcsr() | 0x40);
   std::cout << "Testing again with Denormals-Are-Zero set" << std::endl;
   std::cout << "SSE2 control word is: " << std::hex << _mm_getcsr() << std::endl;
   test_values(1.0f, "float");
   test_values(1.0, "double");

   // Restore the MMX flags:
   _mm_setcsr(mmx_flags);
#endif
   
}



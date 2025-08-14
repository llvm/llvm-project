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
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/debug_adaptor.hpp>
#include <iostream>
#include <iomanip>

#ifdef _MSC_VER
#pragma warning(disable:4127)
#endif

template <class T>
bool is_normalized_value(const T& val)
{
   //
   // Returns false if value has guard digits that are non-zero
   //
   std::intmax_t shift = std::numeric_limits<T>::digits - ilogb(val) - 1;
   T shifted = scalbn(val, shift);
   return floor(shifted) == shifted;
}

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
   if (is_normalized_value(val))
   {
      BOOST_CHECK_EQUAL(float_prior(float_next(val)), val);
      BOOST_CHECK_EQUAL(float_next(float_prior(val)), val);
   }
   BOOST_CHECK_EQUAL(float_distance(float_advance(val, 4), val), -4);
   BOOST_CHECK_EQUAL(float_distance(float_advance(val, -4), val), 4);
   if(std::numeric_limits<T>::is_specialized && boost::math::detail::has_denorm_now<T>())
   {
      BOOST_CHECK_EQUAL(float_distance(float_advance(float_next(float_next(val)), 4), float_next(float_next(val))), -4);
      BOOST_CHECK_EQUAL(float_distance(float_advance(float_next(float_next(val)), -4), float_next(float_next(val))), 4);
   }
   if (is_normalized_value(val))
   {
      if (val > 0)
      {
         T n = val + ulp(val);
         T fn = float_next(val);
         if (n > fn)
         {
            BOOST_CHECK_LE(ulp(val), boost::math::tools::min_value<T>());
         }
         else
         {
            BOOST_CHECK_EQUAL(fn, n);
         }
      }
      else if (val == 0)
      {
         BOOST_CHECK_GE(boost::math::tools::min_value<T>(), ulp(val));
      }
      else
      {
         T n = val - ulp(val);
         T fp = float_prior(val);
         if (n < fp)
         {
            BOOST_CHECK_LE(ulp(val), boost::math::tools::min_value<T>());
         }
         else
         {
            BOOST_CHECK_EQUAL(fp, n);
         }
      }
   }
}

template <class T>
void test_values(const T& val, const char* name)
{
   static const T a = T("1.3456724e22");
   static const T b = T("1.3456724e-22");
   static const T z = 0;
   static const T one = 1;
   static const T radix = std::numeric_limits<T>::radix;

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
   test_value(T(-a), name);
   test_value(b, name);
   test_value(T(-b), name);
   test_value(T(b / 3), name);
   test_value(T(-b / 3), name);
   test_value(boost::math::tools::epsilon<T>(), name);
   test_value(T(-boost::math::tools::epsilon<T>()), name);
   test_value(boost::math::tools::min_value<T>(), name);
   test_value(T(-boost::math::tools::min_value<T>()), name);
   if (std::numeric_limits<T>::is_specialized && boost::math::detail::has_denorm_now<T>() && ((std::numeric_limits<T>::min)() / 2 != 0))
   {
      test_value(z, name);
      test_value(T(-z), name);
   }
   test_value(one, name);
   test_value(T(-one), name);
   test_value(radix, name);
   test_value(T(-radix), name);

   if(std::numeric_limits<T>::is_specialized && boost::math::detail::has_denorm_now<T>() && ((std::numeric_limits<T>::min)() / 2 != 0))
   {
      test_value(std::numeric_limits<T>::denorm_min(), name);
      test_value(T(-std::numeric_limits<T>::denorm_min()), name);
      test_value(T(2 * std::numeric_limits<T>::denorm_min()), name);
      test_value(T(-2 * std::numeric_limits<T>::denorm_min()), name);
   }

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
      BOOST_MATH_CHECK_THROW(boost::math::float_prior(std::numeric_limits<T>::quiet_NaN()), std::domain_error);
      BOOST_MATH_CHECK_THROW(boost::math::float_next(std::numeric_limits<T>::quiet_NaN()), std::domain_error);
   }
}

BOOST_AUTO_TEST_CASE( test_main )
{
   // Very slow, but debuggable:
   //test_values(boost::multiprecision::number<boost::multiprecision::debug_adaptor<boost::multiprecision::cpp_dec_float_50::backend_type> >(0), "cpp_dec_float_50");
   
   // Faster, but no good for diagnosing the cause of any issues:
   #ifndef BOOST_MATH_STANDALONE
   test_values(boost::multiprecision::cpp_dec_float_50(0), "cpp_dec_float_50");
   #endif
}



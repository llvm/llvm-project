//  (C) Copyright John Maddock 2015.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/concepts/real_concept.hpp>
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/special_functions/next.hpp>
#include <boost/math/special_functions/ulp.hpp>
#include <boost/math/special_functions/relative_difference.hpp>
#include <iostream>
#include <iomanip>

template <class T>
void test_value(const T& val, const char* name)
{
   using namespace boost::math;
   using std::fabs;
   T next = float_next(val);
   T prev = float_prior(val);

   if((boost::math::isinf)(next))
   {
      BOOST_CHECK_EQUAL(relative_difference(val, next), tools::max_value<T>());
      return;
   }
   if((boost::math::isinf)(prev))
   {
      BOOST_CHECK_EQUAL(relative_difference(val, prev), tools::max_value<T>());
      return;
   }

   BOOST_CHECK_EQUAL(relative_difference(val, next), relative_difference(next, val));
   BOOST_CHECK_EQUAL(epsilon_difference(val, next), epsilon_difference(next, val));
   BOOST_CHECK_LE(relative_difference(val, next), boost::math::tools::epsilon<T>());
   BOOST_CHECK_LE(epsilon_difference(val, next), T(1));
   if((fabs(val) > tools::min_value<T>()) || (fabs(next) > tools::min_value<T>()))
   {
      BOOST_CHECK_GT(relative_difference(val, next), T(0));
      BOOST_CHECK_GT(epsilon_difference(val, next), T(0));
   }
   else
   {
      BOOST_CHECK_EQUAL(relative_difference(val, next), T(0));
      BOOST_CHECK_EQUAL(epsilon_difference(val, next), T(0));
   }

   BOOST_CHECK_EQUAL(relative_difference(val, prev), relative_difference(prev, val));
   BOOST_CHECK_EQUAL(epsilon_difference(val, prev), epsilon_difference(prev, val));
   if((fabs(val) > tools::min_value<T>()) || (fabs(prev) > tools::min_value<T>()))
   {
      BOOST_CHECK_GT(relative_difference(val, prev), T(0));
      BOOST_CHECK_GT(epsilon_difference(val, prev), T(0));
   }
   else
   {
      BOOST_CHECK_EQUAL(relative_difference(val, prev), T(0));
      BOOST_CHECK_EQUAL(epsilon_difference(val, prev), T(0));
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

   for(unsigned i = 0; i < sizeof(primes) / sizeof(primes[0]); ++i)
   {
      for(unsigned j = 0; j < sizeof(primes) / sizeof(primes[0]); ++j)
      {
         test_value(T(primes[i]) / T(primes[j]), name);
         test_value(-T(primes[i]) / T(primes[j]), name);
      }
   }
   using namespace boost::math;
   BOOST_CHECK_EQUAL(relative_difference(tools::min_value<T>(), -tools::min_value<T>()), tools::max_value<T>());
   BOOST_CHECK_EQUAL(epsilon_difference(tools::min_value<T>(), -tools::min_value<T>()), tools::max_value<T>());
   if(std::numeric_limits<T>::has_infinity)
   {
      BOOST_CHECK_EQUAL(relative_difference(std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity()), T(0));
      BOOST_CHECK_EQUAL(epsilon_difference(std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity()), T(0));
      BOOST_CHECK_EQUAL(relative_difference(std::numeric_limits<T>::infinity(), tools::max_value<T>()), tools::max_value<T>());
      BOOST_CHECK_EQUAL(epsilon_difference(std::numeric_limits<T>::infinity(), tools::max_value<T>()), tools::max_value<T>());
      BOOST_CHECK_EQUAL(relative_difference(std::numeric_limits<T>::infinity(), tools::max_value<T>() / 2), tools::max_value<T>());
      BOOST_CHECK_EQUAL(epsilon_difference(std::numeric_limits<T>::infinity(), tools::max_value<T>() / 2), tools::max_value<T>());
      BOOST_CHECK_EQUAL(relative_difference(tools::max_value<T>(), std::numeric_limits<T>::infinity()), tools::max_value<T>());
      BOOST_CHECK_EQUAL(epsilon_difference(tools::max_value<T>(), std::numeric_limits<T>::infinity()), tools::max_value<T>());
      BOOST_CHECK_EQUAL(relative_difference(tools::max_value<T>() / 2, std::numeric_limits<T>::infinity()), tools::max_value<T>());
      BOOST_CHECK_EQUAL(epsilon_difference(tools::max_value<T>() / 2, std::numeric_limits<T>::infinity()), tools::max_value<T>());
   }
   if(std::numeric_limits<T>::has_quiet_NaN)
   {
      BOOST_CHECK_EQUAL(relative_difference(std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN()), tools::max_value<T>());
      BOOST_CHECK_EQUAL(epsilon_difference(std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN()), tools::max_value<T>());
      BOOST_CHECK_EQUAL(relative_difference(std::numeric_limits<T>::quiet_NaN(), T(2)), tools::max_value<T>());
      BOOST_CHECK_EQUAL(epsilon_difference(std::numeric_limits<T>::quiet_NaN(), T(2)), tools::max_value<T>());
      BOOST_CHECK_EQUAL(relative_difference(T(2), std::numeric_limits<T>::quiet_NaN()), tools::max_value<T>());
      BOOST_CHECK_EQUAL(epsilon_difference(T(2), std::numeric_limits<T>::quiet_NaN()), tools::max_value<T>());
   }

}

BOOST_AUTO_TEST_CASE( test_main )
{
   test_values(1.0f, "float");
   test_values(1.0, "double");
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   test_values(1.0L, "long double");
   test_values(boost::math::concepts::real_concept(0), "real_concept");
#endif
}



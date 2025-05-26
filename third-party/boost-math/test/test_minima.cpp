//  Copyright John Maddock 2006.
//  Copyright Paul A. Bristow 2007.

//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pch.hpp>

#include <boost/math/tools/minima.hpp>
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <iostream>
#include <iomanip>

template <class T>
struct poly_test
{
   // minima is at (3,4):
   T operator()(const T& v)
   {
      T a = v - 3;
      return 3 * a * a + 4;
   }
};

template <class T>
void test_minima(T, const char* /* name */)
{
   std::pair<T, T> m = boost::math::tools::brent_find_minima(poly_test<T>(), T(-10), T(10), 50);
   BOOST_CHECK_CLOSE(m.first, T(3), T(0.001));
   BOOST_CHECK_CLOSE(m.second, T(4), T(0.001));

   T (*fp)(T);
#if defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   fp = boost::math::lgamma<T>;
#else
   fp = boost::math::lgamma;
#endif

   m = boost::math::tools::brent_find_minima(fp, T(0.5), T(10), 50);
   BOOST_CHECK_CLOSE(m.first, T(1.461632), T(0.1));
#if defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   fp = boost::math::tgamma<T>;
#else
   fp = boost::math::tgamma;
#endif
   m = boost::math::tools::brent_find_minima(fp, T(0.5), T(10), 50);
   BOOST_CHECK_CLOSE(m.first, T(1.461632), T(0.1));
}

BOOST_AUTO_TEST_CASE( test_main )
{
   test_minima(0.1f, "float");
   test_minima(0.1, "double");
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   test_minima(0.1L, "long double");
#endif
   
}



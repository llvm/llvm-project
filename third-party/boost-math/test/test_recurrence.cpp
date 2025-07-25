//  (C) Copyright John Maddock 2018.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MODULE test_recurrences

#include <boost/math/tools/config.hpp>

#ifndef BOOST_NO_CXX11_HDR_TUPLE
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/math/tools/recurrence.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
//#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/concepts/real_concept.hpp>

#ifdef _MSC_VER
#pragma warning(disable:4127)
#endif

template <class T>
struct bessel_jy_recurrence
{
   bessel_jy_recurrence(T v, T z) : v(v), z(z) {}
   boost::math::tuple<T, T, T> operator()(int k)const
   {
      return boost::math::tuple<T, T, T>(T(1), -2 * (v + k) / z, T(1));
   }

   T v, z;
};

template <class T>
struct bessel_ik_recurrence
{
   bessel_ik_recurrence(T v, T z) : v(v), z(z) {}
   boost::math::tuple<T, T, T> operator()(int k)const
   {
      return boost::math::tuple<T, T, T>(T(1), -2 * (v + k) / z, T(-1));
   }

   T v, z;
};


template <class T>
void test_spots(T, const char* name)
{
   std::cout << "Running tests for type " << name << std::endl;
   T tol = boost::math::tools::epsilon<T>() * 5;
   if ((std::numeric_limits<T>::digits > 53) || (std::numeric_limits<T>::digits == 0))
      tol *= 5;
   //
   // Test forward recurrence on Y_v(x):
   //
   {
      T v = 22.25;
      T x = 4.125;
      bessel_jy_recurrence<T> coef(v, x);
      T prev;
      T first = boost::math::cyl_neumann(v - 1, x);
      T second = boost::math::cyl_neumann(v, x);
      T sixth = boost::math::tools::apply_recurrence_relation_forward(coef, 6, first, second, (long long*)0, &prev);
      T expected1 = boost::math::cyl_neumann(v + 6, x);
      T expected2 = boost::math::cyl_neumann(v + 5, x);
      BOOST_CHECK_CLOSE_FRACTION(sixth, expected1, tol);
      BOOST_CHECK_CLOSE_FRACTION(prev, expected2, tol);

      boost::math::tools::forward_recurrence_iterator< bessel_jy_recurrence<T> > it(coef, first, second);
      for (unsigned i = 0; i < 15; ++i)
      {
         expected1 = boost::math::cyl_neumann(v + i, x);
         T found = *it;
         BOOST_CHECK_CLOSE_FRACTION(found, expected1, tol);
         ++it;
      }

      if (std::numeric_limits<T>::max_exponent > 300)
      {
         //
         // This calculates the ratio Y_v(x)/Y_v+1(x) from the recurrence relations
         // which are only transiently stable since Y_v is not minimal as v->-INF
         // but only as v->0.  We have to be sure that v is sufficiently large that
         // convergence is complete before we reach the origin.
         //
         v = 102.75;
         std::uintmax_t max_iter = 200;
         T ratio = boost::math::tools::function_ratio_from_forwards_recurrence(bessel_jy_recurrence<T>(v, x), boost::math::tools::epsilon<T>(), max_iter);
         first = boost::math::cyl_neumann(v, x);
         second = boost::math::cyl_neumann(v + 1, x);
         BOOST_CHECK_CLOSE_FRACTION(ratio, first / second, tol);

         boost::math::tools::forward_recurrence_iterator< bessel_jy_recurrence<T> > it2(bessel_jy_recurrence<T>(v, x), boost::math::cyl_neumann(v, x));
         for (unsigned i = 0; i < 15; ++i)
         {
            expected1 = boost::math::cyl_neumann(v + i, x);
            T found = *it2;
            BOOST_CHECK_CLOSE_FRACTION(found, expected1, tol);
            ++it2;
         }
      }

   }
   //
   // Test backward recurrence on J_v(x):
   //
   {
      if ((std::numeric_limits<T>::digits > 53) || !std::numeric_limits<T>::is_specialized)
         tol *= 5;

      T v = 22.25;
      T x = 4.125;
      bessel_jy_recurrence<T> coef(v, x);
      T prev;
      T first = boost::math::cyl_bessel_j(v + 1, x);
      T second = boost::math::cyl_bessel_j(v, x);
      T sixth = boost::math::tools::apply_recurrence_relation_backward(coef, 6, first, second, (long long*)0, &prev);
      T expected1 = boost::math::cyl_bessel_j(v - 6, x);
      T expected2 = boost::math::cyl_bessel_j(v - 5, x);
      BOOST_CHECK_CLOSE_FRACTION(sixth, expected1, tol);
      BOOST_CHECK_CLOSE_FRACTION(prev, expected2, tol);

      boost::math::tools::backward_recurrence_iterator< bessel_jy_recurrence<T> > it(coef, first, second);
      for (unsigned i = 0; i < 15; ++i)
      {
         expected1 = boost::math::cyl_bessel_j(v - i, x);
         T found = *it;
         BOOST_CHECK_CLOSE_FRACTION(found, expected1, tol);
         ++it;
      }

      std::uintmax_t max_iter = 200;
      T ratio = boost::math::tools::function_ratio_from_backwards_recurrence(bessel_jy_recurrence<T>(v, x), boost::math::tools::epsilon<T>(), max_iter);
      first = boost::math::cyl_bessel_j(v, x);
      second = boost::math::cyl_bessel_j(v - 1, x);
      BOOST_CHECK_CLOSE_FRACTION(ratio, first / second, tol);

      boost::math::tools::backward_recurrence_iterator< bessel_jy_recurrence<T> > it2(bessel_jy_recurrence<T>(v, x), boost::math::cyl_bessel_j(v, x));
      //boost::math::tools::backward_recurrence_iterator< bessel_jy_recurrence<T> > it3(bessel_jy_recurrence<T>(v, x), boost::math::cyl_neumann(v+1, x), boost::math::cyl_neumann(v, x));
      for (unsigned i = 0; i < 15; ++i)
      {
         expected1 = boost::math::cyl_bessel_j(v - i, x);
         T found = *it2;
         BOOST_CHECK_CLOSE_FRACTION(found, expected1, tol);
         ++it2;
      }

   }
}


BOOST_AUTO_TEST_CASE( test_main )
{
   BOOST_MATH_CONTROL_FP;
#if !defined(TEST) || TEST == 1
   test_spots(0.0F, "float");
   test_spots(0.0, "double");
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   test_spots(0.0L, "long double");
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
   test_spots(boost::math::concepts::real_concept(0.1), "real_concept");
#endif
#endif
#endif
#if !defined(TEST) || TEST == 2 || TEST == 3
   #ifndef BOOST_MATH_NO_MP_TESTS
   test_spots(boost::multiprecision::cpp_bin_float_quad(), "cpp_bin_float_quad");
   #endif
#endif
}

#else

int main() { return 0; }

#endif

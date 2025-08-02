//  (C) Copyright John Maddock 2015.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "pch.hpp"

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/test/results_collector.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/math/special_functions/cbrt.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <iostream>
#include <iomanip>
#include <tuple>
#include "table_type.hpp"

// No derivatives - using TOMS748 internally.
struct cbrt_functor_noderiv
{ //  cube root of x using only function - no derivatives.
   cbrt_functor_noderiv(double to_find_root_of) : a(to_find_root_of)
   { // Constructor just stores value a to find root of.
   }
   double operator()(double x)
   {
      double fx = x*x*x - a; // Difference (estimate x^3 - a).
      return fx;
   }
private:
   double a; // to be 'cube_rooted'.
}; // template <class T> struct cbrt_functor_noderiv

// Using 1st derivative only Newton-Raphson
struct cbrt_functor_deriv
{ // Functor also returning 1st derivative.
   cbrt_functor_deriv(double const& to_find_root_of) : a(to_find_root_of)
   { // Constructor stores value a to find root of,
      // for example: calling cbrt_functor_deriv<double>(x) to use to get cube root of x.
   }
   std::pair<double, double> operator()(double const& x)
   { // Return both f(x) and f'(x).
      double fx = x*x*x - a; // Difference (estimate x^3 - value).
      double dx = 3 * x*x; // 1st derivative = 3x^2.
      return std::make_pair(fx, dx); // 'return' both fx and dx.
   }
private:
   double a; // to be 'cube_rooted'.
};
// Using 1st and 2nd derivatives with Halley algorithm.
struct cbrt_functor_2deriv
{ // Functor returning both 1st and 2nd derivatives.
   cbrt_functor_2deriv(double const& to_find_root_of) : a(to_find_root_of)
   { // Constructor stores value a to find root of, for example:
      // calling cbrt_functor_2deriv<double>(x) to get cube root of x,
   }
   std::tuple<double, double, double> operator()(double const& x)
   { // Return both f(x) and f'(x) and f''(x).
      double fx = x*x*x - a; // Difference (estimate x^3 - value).
      double dx = 3 * x*x; // 1st derivative = 3x^2.
      double d2x = 6 * x; // 2nd derivative = 6x.
      return std::make_tuple(fx, dx, d2x); // 'return' fx, dx and d2x.
   }
private:
   double a; // to be 'cube_rooted'.
};

template <class T, class Policy>
struct ibeta_roots_1   // for first order algorithms
{
   ibeta_roots_1(T _a, T _b, T t, bool inv = false)
      : a(_a), b(_b), target(t), invert(inv) {}

   T operator()(const T& x)
   {
      return boost::math::detail::ibeta_imp(a, b, x, Policy(), invert, true) - target;
   }
private:
   T a, b, target;
   bool invert;
};

template <class T, class Policy>
struct ibeta_roots_2   // for second order algorithms
{
   ibeta_roots_2(T _a, T _b, T t, bool inv = false)
      : a(_a), b(_b), target(t), invert(inv) {}

   boost::math::tuple<T, T> operator()(const T& x)
   {
      typedef boost::math::lanczos::lanczos<T, Policy> S;
      typedef typename S::type L;
      T f = boost::math::detail::ibeta_imp(a, b, x, Policy(), invert, true) - target;
      T f1 = invert ?
         -boost::math::detail::ibeta_power_terms(b, a, 1 - x, x, L(), true, Policy())
         : boost::math::detail::ibeta_power_terms(a, b, x, 1 - x, L(), true, Policy());
      T y = 1 - x;
      if (y == 0)
         y = boost::math::tools::min_value<T>() * 8;
      f1 /= y * x;

      // make sure we don't have a zero derivative:
      if (f1 == 0)
         f1 = (invert ? -1 : 1) * boost::math::tools::min_value<T>() * 64;

      return boost::math::make_tuple(f, f1);
   }
private:
   T a, b, target;
   bool invert;
};

template <class T, class Policy>
struct ibeta_roots_3   // for third order algorithms
{
   ibeta_roots_3(T _a, T _b, T t, bool inv = false)
      : a(_a), b(_b), target(t), invert(inv) {}

   boost::math::tuple<T, T, T> operator()(const T& x)
   {
      typedef typename boost::math::lanczos::lanczos<T, Policy>::type L;
      T f = boost::math::detail::ibeta_imp(a, b, x, Policy(), invert, true) - target;
      T f1 = invert ?
         -boost::math::detail::ibeta_power_terms(b, a, 1 - x, x, L(), true, Policy())
         : boost::math::detail::ibeta_power_terms(a, b, x, 1 - x, L(), true, Policy());
      T y = 1 - x;
      if (y == 0)
         y = boost::math::tools::min_value<T>() * 8;
      f1 /= y * x;
      T f2 = f1 * (-y * a + (b - 2) * x + 1) / (y * x);
      if (invert)
         f2 = -f2;

      // make sure we don't have a zero derivative:
      if (f1 == 0)
         f1 = (invert ? -1 : 1) * boost::math::tools::min_value<T>() * 64;

      return boost::math::make_tuple(f, f1, f2);
   }
private:
   T a, b, target;
   bool invert;
};


BOOST_AUTO_TEST_CASE( test_main )
{
   int newton_limits = static_cast<int>(std::numeric_limits<double>::digits * 0.6);

   double arg = 1e-50;
   std::uintmax_t iters;
   double guess;
   double dr;

   while(arg < 1e50)
   {
      double result = boost::math::cbrt(arg);
      //
      // Start with a really bad guess 5 times below the result:
      //
      guess = result / 5;
      iters = 1000;
      // TOMS algo first:
      std::pair<double, double> r = boost::math::tools::bracket_and_solve_root(cbrt_functor_noderiv(arg), guess, 2.0, true, boost::math::tools::eps_tolerance<double>(), iters);
      BOOST_CHECK_CLOSE_FRACTION((r.first + r.second) / 2, result, std::numeric_limits<double>::epsilon() * 4);
      BOOST_CHECK_LE(iters, 14);
      // Newton next:
      iters = 1000;
      dr = boost::math::tools::newton_raphson_iterate(cbrt_functor_deriv(arg), guess, guess / 2, result * 10, newton_limits, iters);
      BOOST_CHECK_CLOSE_FRACTION(dr, result, std::numeric_limits<double>::epsilon() * 2);
      BOOST_CHECK_LE(iters, 12);
      // Halley next:
      iters = 1000;
      dr = boost::math::tools::halley_iterate(cbrt_functor_2deriv(arg), guess, result / 10, result * 10, newton_limits, iters);
      BOOST_CHECK_CLOSE_FRACTION(dr, result, std::numeric_limits<double>::epsilon() * 2);
      BOOST_CHECK_LE(iters, 7);
      // Schroder next:
      iters = 1000;
      dr = boost::math::tools::schroder_iterate(cbrt_functor_2deriv(arg), guess, result / 10, result * 10, newton_limits, iters);
      BOOST_CHECK_CLOSE_FRACTION(dr, result, std::numeric_limits<double>::epsilon() * 2);
      BOOST_CHECK_LE(iters, 11);
      //
      // Over again with a bad guess 5 times larger than the result:
      //
      iters = 1000;
      guess = result * 5;
      r = boost::math::tools::bracket_and_solve_root(cbrt_functor_noderiv(arg), guess, 2.0, true, boost::math::tools::eps_tolerance<double>(), iters);
      BOOST_CHECK_CLOSE_FRACTION((r.first + r.second) / 2, result, std::numeric_limits<double>::epsilon() * 4);
      BOOST_CHECK_LE(iters, 14);
      // Newton next:
      iters = 1000;
      dr = boost::math::tools::newton_raphson_iterate(cbrt_functor_deriv(arg), guess, result / 10, result * 10, newton_limits, iters);
      BOOST_CHECK_CLOSE_FRACTION(dr, result, std::numeric_limits<double>::epsilon() * 2);
      BOOST_CHECK_LE(iters, 12);
      // Halley next:
      iters = 1000;
      dr = boost::math::tools::halley_iterate(cbrt_functor_2deriv(arg), guess, result / 10, result * 10, newton_limits, iters);
      BOOST_CHECK_CLOSE_FRACTION(dr, result, std::numeric_limits<double>::epsilon() * 2);
      BOOST_CHECK_LE(iters, 7);
      // Schroder next:
      iters = 1000;
      dr = boost::math::tools::schroder_iterate(cbrt_functor_2deriv(arg), guess, result / 10, result * 10, newton_limits, iters);
      BOOST_CHECK_CLOSE_FRACTION(dr, result, std::numeric_limits<double>::epsilon() * 2);
      BOOST_CHECK_LE(iters, 11);
      //
      // A much better guess, 1% below result:
      //
      iters = 1000;
      guess = result * 0.9;
      r = boost::math::tools::bracket_and_solve_root(cbrt_functor_noderiv(arg), guess, 2.0, true, boost::math::tools::eps_tolerance<double>(), iters);
      BOOST_CHECK_CLOSE_FRACTION((r.first + r.second) / 2, result, std::numeric_limits<double>::epsilon() * 4);
      BOOST_CHECK_LE(iters, 12);
      // Newton next:
      iters = 1000;
      dr = boost::math::tools::newton_raphson_iterate(cbrt_functor_deriv(arg), guess, result / 10, result * 10, newton_limits, iters);
      BOOST_CHECK_CLOSE_FRACTION(dr, result, std::numeric_limits<double>::epsilon() * 2);
      BOOST_CHECK_LE(iters, 5);
      // Halley next:
      iters = 1000;
      dr = boost::math::tools::halley_iterate(cbrt_functor_2deriv(arg), guess, result / 10, result * 10, newton_limits, iters);
      BOOST_CHECK_CLOSE_FRACTION(dr, result, std::numeric_limits<double>::epsilon() * 2);
      BOOST_CHECK_LE(iters, 3);
      // Schroder next:
      iters = 1000;
      dr = boost::math::tools::schroder_iterate(cbrt_functor_2deriv(arg), guess, result / 10, result * 10, newton_limits, iters);
      BOOST_CHECK_CLOSE_FRACTION(dr, result, std::numeric_limits<double>::epsilon() * 2);
      BOOST_CHECK_LE(iters, 4);
      //
      // A much better guess, 1% above result:
      //
      iters = 1000;
      guess = result * 1.1;
      r = boost::math::tools::bracket_and_solve_root(cbrt_functor_noderiv(arg), guess, 2.0, true, boost::math::tools::eps_tolerance<double>(), iters);
      BOOST_CHECK_CLOSE_FRACTION((r.first + r.second) / 2, result, std::numeric_limits<double>::epsilon() * 4);
      BOOST_CHECK_LE(iters, 12);
      // Newton next:
      iters = 1000;
      dr = boost::math::tools::newton_raphson_iterate(cbrt_functor_deriv(arg), guess, result / 10, result * 10, newton_limits, iters);
      BOOST_CHECK_CLOSE_FRACTION(dr, result, std::numeric_limits<double>::epsilon() * 2);
      BOOST_CHECK_LE(iters, 5);
      // Halley next:
      iters = 1000;
      dr = boost::math::tools::halley_iterate(cbrt_functor_2deriv(arg), guess, result / 10, result * 10, newton_limits, iters);
      BOOST_CHECK_CLOSE_FRACTION(dr, result, std::numeric_limits<double>::epsilon() * 2);
      BOOST_CHECK_LE(iters, 3);
      // Schroder next:
      iters = 1000;
      dr = boost::math::tools::schroder_iterate(cbrt_functor_2deriv(arg), guess, result / 10, result * 10, newton_limits, iters);
      BOOST_CHECK_CLOSE_FRACTION(dr, result, std::numeric_limits<double>::epsilon() * 2);
      BOOST_CHECK_LE(iters, 4);

      arg *= 3.5;
   }

   //
   // Test ibeta as this triggers all the pathological cases!
   //
#ifndef SC_
#define SC_(x) x
#endif
#define T double

#  include "ibeta_small_data.ipp"

   for (unsigned i = 0; i < ibeta_small_data.size(); ++i)
   {
      //
      // These inverse tests are thrown off if the output of the
      // incomplete beta is too close to 1: basically there is insuffient
      // information left in the value we're using as input to the inverse
      // to be able to get back to the original value.
      //
      if (ibeta_small_data[i][5] == 0)
      {
         iters = 1000;
         dr = boost::math::tools::newton_raphson_iterate(ibeta_roots_2<double, boost::math::policies::policy<> >(ibeta_small_data[i][0], ibeta_small_data[i][1], ibeta_small_data[i][5]), 0.5, 0.0, 1.0, 53, iters);
         BOOST_CHECK_EQUAL(dr, 0.0);
         BOOST_CHECK_LE(iters, 27);
         iters = 1000;
         dr = boost::math::tools::halley_iterate(ibeta_roots_3<double, boost::math::policies::policy<> >(ibeta_small_data[i][0], ibeta_small_data[i][1], ibeta_small_data[i][5]), 0.5, 0.0, 1.0, 53, iters);
         BOOST_CHECK_EQUAL(dr, 0.0);
         BOOST_CHECK_LE(iters, 10);
      }
      else if ((1 - ibeta_small_data[i][5] > 0.001)
         && (fabs(ibeta_small_data[i][5]) > 2 * boost::math::tools::min_value<double>()))
      {
         iters = 1000;
         double result = ibeta_small_data[i][2];
         dr = boost::math::tools::newton_raphson_iterate(ibeta_roots_2<double, boost::math::policies::policy<> >(ibeta_small_data[i][0], ibeta_small_data[i][1], ibeta_small_data[i][5]), 0.5, 0.0, 1.0, 53, iters);
         BOOST_CHECK_CLOSE_FRACTION(dr, result, std::numeric_limits<double>::epsilon() * 200);
#if defined(BOOST_MSVC) && (BOOST_MSVC == 1600)
         BOOST_CHECK_LE(iters, 40);
#else
         BOOST_CHECK_LE(iters, 27);
#endif
         iters = 1000;
         result = ibeta_small_data[i][2];
         dr = boost::math::tools::halley_iterate(ibeta_roots_3<double, boost::math::policies::policy<> >(ibeta_small_data[i][0], ibeta_small_data[i][1], ibeta_small_data[i][5]), 0.5, 0.0, 1.0, 53, iters);
         BOOST_CHECK_CLOSE_FRACTION(dr, result, std::numeric_limits<double>::epsilon() * 200);
#if defined(__PPC__) || defined(__aarch64__) || (LDBL_MANT_DIG > 100)
         BOOST_CHECK_LE(iters, 55);
#else
         BOOST_CHECK_LE(iters, 40);
#endif
      }
      else if (1 == ibeta_small_data[i][5])
      {
         iters = 1000;
         dr = boost::math::tools::newton_raphson_iterate(ibeta_roots_2<double, boost::math::policies::policy<> >(ibeta_small_data[i][0], ibeta_small_data[i][1], ibeta_small_data[i][5]), 0.5, 0.0, 1.0, 53, iters);
         BOOST_CHECK_EQUAL(dr, 1.0);
         BOOST_CHECK_LE(iters, 27);
         iters = 1000;
         dr = boost::math::tools::halley_iterate(ibeta_roots_3<double, boost::math::policies::policy<> >(ibeta_small_data[i][0], ibeta_small_data[i][1], ibeta_small_data[i][5]), 0.5, 0.0, 1.0, 53, iters);
         BOOST_CHECK_EQUAL(dr, 1.0);
         BOOST_CHECK_LE(iters, 10);
      }
   }

}


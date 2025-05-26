// Copyright John Maddock 2014

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // Boost.Test
#include <boost/test/results_collector.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/fusion/include/tuple.hpp>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/math/tools/toms748_solve.hpp>
#include <tuple>

#include <iostream>
#include <iomanip>
   using std::cout;
   using std::endl;
   using std::setprecision;

#include <boost/math/tools/roots.hpp>

//
// We'll use cbrt as an example:
//
struct cbtr_functor_1
{
   cbtr_functor_1(double x) : m_target(x) {}
   double operator()(double x)
   {
      return x * x * x - m_target;
   }
private:
   double m_target;
};

struct cbtr_functor_2a
{
   cbtr_functor_2a(double x) : m_target(x) {}
   std::pair<double, double> operator()(double x)
   {
      return std::make_pair(x * x * x - m_target, 3 * x * x);
   }
private:
   double m_target;
};

struct cbtr_functor_2b
{
   cbtr_functor_2b(double x) : m_target(x) {}
   std::tuple<double, double> operator()(double x)
   {
      return std::tuple<double, double>(x * x * x - m_target, 3 * x * x);
   }
private:
   double m_target;
};
struct cbtr_functor_2c
{
   cbtr_functor_2c(double x) : m_target(x) {}
   boost::tuple<double, double> operator()(double x)
   {
      return boost::tuple<double, double>(x * x * x - m_target, 3 * x * x);
   }
private:
   double m_target;
};
struct cbtr_functor_2d
{
   cbtr_functor_2d(double x) : m_target(x) {}
   boost::fusion::tuple<double, double> operator()(double x)
   {
      return boost::fusion::tuple<double, double>(x * x * x - m_target, 3 * x * x);
   }
private:
   double m_target;
};

struct cbtr_functor_3b
{
   cbtr_functor_3b(double x) : m_target(x) {}
   std::tuple<double, double, double> operator()(double x)
   {
      return std::tuple<double, double, double>(x * x * x - m_target, 3 * x * x, 6 * x);
   }
private:
   double m_target;
};

struct cbtr_functor_3c
{
   cbtr_functor_3c(double x) : m_target(x) {}
   boost::tuple<double, double, double> operator()(double x)
   {
      return boost::tuple<double, double, double>(x * x * x - m_target, 3 * x * x, 6 * x);
   }
private:
   double m_target;
};
struct cbtr_functor_3d
{
   cbtr_functor_3d(double x) : m_target(x) {}
   boost::fusion::tuple<double, double, double> operator()(double x)
   {
      return boost::fusion::tuple<double, double, double>(x * x * x - m_target, 3 * x * x, 6 * x);
   }
private:
   double m_target;
};


BOOST_AUTO_TEST_CASE( test_main )
{
   double x = 27;
   double expected = 3;
   double result;
   double tolerance = std::numeric_limits<double>::epsilon() * 5;
   std::pair<double, double> p;
   //
   // Start by trying the unary functors, bisect first:
   //
   cbtr_functor_1 f1(x);
   boost::math::tools::eps_tolerance<double> t(std::numeric_limits<double>::digits - 1);
   p = boost::math::tools::bisect(f1, 0.0, x, t);
   result = (p.first + p.second) / 2;
   BOOST_CHECK_CLOSE_FRACTION(expected, result, tolerance);
   //
   // bracket_and_solve_root:
   //
   std::uintmax_t max_iter = boost::math::policies::get_max_root_iterations<boost::math::policies::policy<> >();
   p = boost::math::tools::bracket_and_solve_root(f1, x, 2.0, true, t, max_iter);
   result = (p.first + p.second) / 2;
   BOOST_CHECK_CLOSE_FRACTION(expected, result, tolerance);
   //
   // toms748_solve:
   //
   max_iter = boost::math::policies::get_max_root_iterations<boost::math::policies::policy<> >();
   p = boost::math::tools::toms748_solve(f1, 0.0, x, t, max_iter);
   result = (p.first + p.second) / 2;
   BOOST_CHECK_CLOSE_FRACTION(expected, result, tolerance);

   //
   // Now try again with C++11 lambda's
   //
   p = boost::math::tools::bisect([x](double z){ return z * z * z - x; }, 0.0, x, t);
   result = (p.first + p.second) / 2;
   BOOST_CHECK_CLOSE_FRACTION(expected, result, tolerance);
   //
   // bracket_and_solve_root:
   //
   max_iter = boost::math::policies::get_max_root_iterations<boost::math::policies::policy<> >();
   p = boost::math::tools::bracket_and_solve_root([x](double z){ return z * z * z - x; }, x, 2.0, true, t, max_iter);
   result = (p.first + p.second) / 2;
   BOOST_CHECK_CLOSE_FRACTION(expected, result, tolerance);
   //
   // toms748_solve:
   //
   max_iter = boost::math::policies::get_max_root_iterations<boost::math::policies::policy<> >();
   p = boost::math::tools::toms748_solve([x](double z){ return z * z * z - x; }, 0.0, x, t, max_iter);
   result = (p.first + p.second) / 2;
   BOOST_CHECK_CLOSE_FRACTION(expected, result, tolerance);

   cbtr_functor_2a f2(x);
   cbtr_functor_2b f3(x);
   cbtr_functor_2c f4(x);
   cbtr_functor_2d f5(x);

   //
   // Binary Functors - newton_raphson_iterate - test each possible tuple type:
   //
   result = boost::math::tools::newton_raphson_iterate(f2, x, 0.0, x, std::numeric_limits<double>::digits - 1);
   BOOST_CHECK_CLOSE_FRACTION(expected, result, tolerance);
   result = boost::math::tools::newton_raphson_iterate(f3, x, 0.0, x, std::numeric_limits<double>::digits - 1);
   BOOST_CHECK_CLOSE_FRACTION(expected, result, tolerance);
   result = boost::math::tools::newton_raphson_iterate(f4, x, 0.0, x, std::numeric_limits<double>::digits - 1);
   BOOST_CHECK_CLOSE_FRACTION(expected, result, tolerance);
   result = boost::math::tools::newton_raphson_iterate(f5, x, 0.0, x, std::numeric_limits<double>::digits - 1);
   BOOST_CHECK_CLOSE_FRACTION(expected, result, tolerance);
   //
   // And again but with lambdas:
   //
   result = boost::math::tools::newton_raphson_iterate([x](double z){ return std::make_pair(z * z * z - x, 3 * z * z); }, x, 0.0, x, std::numeric_limits<double>::digits - 1);
   BOOST_CHECK_CLOSE_FRACTION(expected, result, tolerance);
   result = boost::math::tools::newton_raphson_iterate([x](double z){ return std::make_tuple(z * z * z - x, 3 * z * z); }, x, 0.0, x, std::numeric_limits<double>::digits - 1);
   BOOST_CHECK_CLOSE_FRACTION(expected, result, tolerance);
   result = boost::math::tools::newton_raphson_iterate([x](double z){ return boost::tuple<double, double>(z * z * z - x, 3 * z * z); }, x, 0.0, x, std::numeric_limits<double>::digits - 1);
   BOOST_CHECK_CLOSE_FRACTION(expected, result, tolerance);
   result = boost::math::tools::newton_raphson_iterate([x](double z){ return boost::fusion::tuple<double, double>(z * z * z - x, 3 * z * z); }, x, 0.0, x, std::numeric_limits<double>::digits - 1);
   BOOST_CHECK_CLOSE_FRACTION(expected, result, tolerance);
   cbtr_functor_3b f6(x);
   cbtr_functor_3c f7(x);
   cbtr_functor_3d f8(x);

   //
   // Ternary functors:
   //
   result = boost::math::tools::halley_iterate(f6, x, 0.0, x, std::numeric_limits<double>::digits - 1);
   BOOST_CHECK_CLOSE_FRACTION(expected, result, tolerance);
   result = boost::math::tools::halley_iterate(f7, x, 0.0, x, std::numeric_limits<double>::digits - 1);
   BOOST_CHECK_CLOSE_FRACTION(expected, result, tolerance);
   result = boost::math::tools::halley_iterate(f8, x, 0.0, x, std::numeric_limits<double>::digits - 1);
   BOOST_CHECK_CLOSE_FRACTION(expected, result, tolerance);
   result = boost::math::tools::halley_iterate([x](double z){ return std::make_tuple(z * z * z - x, 3 * z * z, 6 * z); }, x, 0.0, x, std::numeric_limits<double>::digits - 1);
   BOOST_CHECK_CLOSE_FRACTION(expected, result, tolerance);
   result = boost::math::tools::halley_iterate([x](double z){ return boost::tuple<double, double, double>(z * z * z - x, 3 * z * z, 6 * z); }, x, 0.0, x, std::numeric_limits<double>::digits - 1);
   BOOST_CHECK_CLOSE_FRACTION(expected, result, tolerance);
   result = boost::math::tools::halley_iterate([x](double z){ return boost::fusion::tuple<double, double, double>(z * z * z - x, 3 * z * z, 6 * z); }, x, 0.0, x, std::numeric_limits<double>::digits - 1);
   BOOST_CHECK_CLOSE_FRACTION(expected, result, tolerance);
   result = boost::math::tools::schroder_iterate(f6, x, 0.0, x, std::numeric_limits<double>::digits - 1);
   BOOST_CHECK_CLOSE_FRACTION(expected, result, tolerance);
   result = boost::math::tools::schroder_iterate(f7, x, 0.0, x, std::numeric_limits<double>::digits - 1);
   BOOST_CHECK_CLOSE_FRACTION(expected, result, tolerance);
   result = boost::math::tools::schroder_iterate(f8, x, 0.0, x, std::numeric_limits<double>::digits - 1);
   BOOST_CHECK_CLOSE_FRACTION(expected, result, tolerance);
   result = boost::math::tools::schroder_iterate([x](double z){ return std::make_tuple(z * z * z - x, 3 * z * z, 6 * z); }, x, 0.0, x, std::numeric_limits<double>::digits - 1);
   BOOST_CHECK_CLOSE_FRACTION(expected, result, tolerance);
   result = boost::math::tools::schroder_iterate([x](double z){ return boost::tuple<double, double, double>(z * z * z - x, 3 * z * z, 6 * z); }, x, 0.0, x, std::numeric_limits<double>::digits - 1);
   BOOST_CHECK_CLOSE_FRACTION(expected, result, tolerance);
   result = boost::math::tools::schroder_iterate([x](double z){ return boost::fusion::tuple<double, double, double>(z * z * z - x, 3 * z * z, 6 * z); }, x, 0.0, x, std::numeric_limits<double>::digits - 1);
   BOOST_CHECK_CLOSE_FRACTION(expected, result, tolerance);
} // BOOST_AUTO_TEST_CASE( test_main )


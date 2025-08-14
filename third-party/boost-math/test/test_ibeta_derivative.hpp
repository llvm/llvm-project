// Copyright John Maddock 2006.
// Copyright Paul A. Bristow 2007, 2009
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/concepts/real_concept.hpp>
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/tools/stats.hpp>
#include <boost/math/tools/test.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/array.hpp>
#include "functor.hpp"

#include "handle_test_result.hpp"
#include "table_type.hpp"

#ifndef SC_
#define SC_(x) static_cast<typename table_type<T>::type>(BOOST_JOIN(x, L))
#endif

template <class T>
T ibeta_forwarder(T a, T b, T x)
{
   T derivative;
   boost::math::detail::ibeta_imp(a, b, x, boost::math::policies::policy<>(), false, true, &derivative);
   return derivative;
}

template <class Real, class T>
void do_test_beta(const T& data, const char* type_name, const char* test_name)
{
   typedef Real                   value_type;

   typedef value_type (*pg)(value_type, value_type, value_type);
#if defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = boost::math::ibeta_derivative<value_type, value_type, value_type>;
#else
   pg funcp = boost::math::ibeta_derivative;
#endif

   boost::math::tools::test_result<value_type> result;

#if !(defined(ERROR_REPORTING_MODE) && !defined(BETA_INC_FUNCTION_TO_TEST))
   std::cout << "Testing " << test_name << " with type " << type_name
      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

   //
   // test ibeta_derivative against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(funcp, 0, 1, 2),
      extract_result<Real>(3));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "beta (incomplete)", test_name);
#endif

#if defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   funcp = ibeta_forwarder<value_type>;
#else
   funcp = ibeta_forwarder;
#endif

   if(boost::math::tools::digits<value_type>() > 40)
   {
      //
      // test ibeta_derivative against data:
      //
      result = boost::math::tools::test_hetero<Real>(
         data,
         bind_func<Real>(funcp, 0, 1, 2),
         extract_result<Real>(3));
      handle_test_result(result, data[result.worst()], result.worst(), type_name, "beta (incomplete, internal call test)", test_name);
   }
}

template <class T>
void test_beta(T, const char* name)
{
   //
   // The actual test data is rather verbose, so it's in a separate file
   //
   // The contents are as follows, each row of data contains
   // five items, input value a, input value b, integration limits x, beta(a, b, x) and ibeta(a, b, x):
   //
#if !defined(TEST_DATA) || (TEST_DATA == 1)
#  include "ibeta_derivative_small_data.ipp"

   do_test_beta<T>(ibeta_derivative_small_data, name, "Incomplete Beta Function Derivative: Small Values");
#endif

#if !defined(TEST_DATA) || (TEST_DATA == 2)
#  include "ibeta_derivative_data.ipp"

   do_test_beta<T>(ibeta_derivative_data, name, "Incomplete Beta Function Derivative: Medium Values");

#endif
#ifndef __SUNPRO_CC
#if !defined(TEST_DATA) || (TEST_DATA == 3)
#  include "ibeta_derivative_large_data.ipp"

   do_test_beta<T>(ibeta_derivative_large_data, name, "Incomplete Beta Function Derivative: Large and Diverse Values");
#endif
#endif
#if !defined(TEST_DATA) || (TEST_DATA == 4)
#  include "ibeta_derivative_int_data.ipp"

   do_test_beta<T>(ibeta_derivative_int_data, name, "Incomplete Beta Function Derivative: Small Integer Values");
#endif
}

template <class T>
void test_spots(T)
{
   using std::ldexp;
   T tolerance = boost::math::tools::epsilon<T>() * 40000;
      BOOST_CHECK_CLOSE(
         ::boost::math::ibeta_derivative(
            static_cast<T>(2),
            static_cast<T>(4),
            ldexp(static_cast<T>(1), -557)),
         static_cast<T>(4.23957586190238472641508753637420672781472122471791800210e-167L), tolerance * 4);
      BOOST_CHECK_CLOSE(
         ::boost::math::ibeta_derivative(
            static_cast<T>(2),
            static_cast<T>(4.5),
            ldexp(static_cast<T>(1), -557)),
         static_cast<T>(5.24647512910420109893867082626308082567071751558842352760e-167L), tolerance * 4);

      BOOST_IF_CONSTEXPR(std::numeric_limits<T>::has_quiet_NaN)
      {
         T n = std::numeric_limits<T>::quiet_NaN();
         BOOST_MATH_CHECK_THROW(::boost::math::ibeta_derivative(n, static_cast<T>(2.125), static_cast<T>(0.125)), std::domain_error);
         BOOST_MATH_CHECK_THROW(::boost::math::ibeta_derivative(static_cast<T>(2.125), n, static_cast<T>(0.125)), std::domain_error);
         BOOST_MATH_CHECK_THROW(::boost::math::ibeta_derivative(static_cast<T>(2.125), static_cast<T>(1.125), n), std::domain_error);
      }
      BOOST_IF_CONSTEXPR(std::numeric_limits<T>::has_infinity)
      {
         T n = std::numeric_limits<T>::infinity();
         BOOST_MATH_CHECK_THROW(::boost::math::ibeta_derivative(n, static_cast<T>(2.125), static_cast<T>(0.125)), std::domain_error);
         BOOST_MATH_CHECK_THROW(::boost::math::ibeta_derivative(static_cast<T>(2.125), n, static_cast<T>(0.125)), std::domain_error);
         BOOST_MATH_CHECK_THROW(::boost::math::ibeta_derivative(static_cast<T>(2.125), static_cast<T>(1.125), n), std::domain_error);
         BOOST_MATH_CHECK_THROW(::boost::math::ibeta_derivative(-n, static_cast<T>(2.125), static_cast<T>(0.125)), std::domain_error);
         BOOST_MATH_CHECK_THROW(::boost::math::ibeta_derivative(static_cast<T>(2.125), -n, static_cast<T>(0.125)), std::domain_error);
         BOOST_MATH_CHECK_THROW(::boost::math::ibeta_derivative(static_cast<T>(2.125), static_cast<T>(1.125), -n), std::domain_error);
      }
      //
      // Some additional tests: some of our internal root finding code uses a "back door" into
      // ibeta in order to compute ibeta and it's derivative at the same time, we need to 
      // exercise the special case handling in there as well as in the public interface
      // tested above.
      //
      T derivative = 0;
      boost::math::detail::ibeta_imp(T(1), T(2), T(0), boost::math::policies::policy<>(), false, true, &derivative);
      BOOST_CHECK_EQUAL(derivative, T(1));
      boost::math::detail::ibeta_imp(T(0.5), T(2), T(0), boost::math::policies::policy<>(), false, true, &derivative);
      BOOST_CHECK_GT(derivative, boost::math::tools::max_value<T>() / 3); // any large value will do
      BOOST_CHECK_LT(derivative, boost::math::tools::max_value<T>()); // But not so large that arithmetic overflows.
      boost::math::detail::ibeta_imp(T(2.5), T(2), T(0), boost::math::policies::policy<>(), false, true, &derivative);
      BOOST_CHECK_LT(derivative, boost::math::tools::min_value<T>() * 3); // any small value will do
      BOOST_CHECK_GT(derivative, T(0)); // But not zero.
      T val = boost::math::detail::ibeta_imp(T(0.5f), T(0.5f), T(0.25), boost::math::policies::policy<>(), false, true, &derivative);
      BOOST_CHECK_CLOSE(derivative, static_cast<T>(0.7351051938957227326817686644172925885298486404888542037324880270L), tolerance);
      BOOST_CHECK_CLOSE(val, static_cast<T>(0.3333333333333333333333333333333333333333333333333333333333333333L), tolerance);
      BOOST_CHECK_CLOSE(boost::math::beta(T(0.5f), T(0.5f), T(0.25)), static_cast<T>(1.0471975511965977461542144610931676280657231331250352736583148641L), tolerance);
      //
      // Error handling:
      //
      BOOST_CHECK_THROW(boost::math::ibeta_derivative(T(0), T(2), T(0.5)), std::domain_error);
      BOOST_CHECK_THROW(boost::math::ibeta_derivative(T(-1), T(2), T(0.5)), std::domain_error);
      BOOST_CHECK_THROW(boost::math::ibeta_derivative(T(1), T(0), T(0.5)), std::domain_error);
      BOOST_CHECK_THROW(boost::math::ibeta_derivative(T(1), T(-1), T(0.5)), std::domain_error);
}


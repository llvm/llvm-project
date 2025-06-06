// Copyright John Maddock 2006.
// Copyright Paul A. Bristow 2007, 2009
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef _MSC_VER
# pragma warning (disable : 4996) // POSIX name for this item is deprecated
# pragma warning (disable : 4224) // nonstandard extension used : formal parameter 'arg' was previously defined as a type
# pragma warning (disable : 4180) // qualifier applied to function type has no meaning; ignored
#endif

#ifdef __CUDACC__
#pragma nv_diag_suppress 221
#endif

#include <boost/math/concepts/real_concept.hpp>
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/tools/stats.hpp>
#include "../include_private/boost/math/tools/test.hpp"
#include <boost/math/constants/constants.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/array.hpp>
#include "functor.hpp"

#include "handle_test_result.hpp"
#include "table_type.hpp"

#undef small // VC++ #defines small char !!!!!!

#ifndef SC_
#define SC_(x) static_cast<typename table_type<T>::type>(BOOST_JOIN(x, L))
#endif

template <class Real, class T>
void do_test_beta(const T& data, const char* type_name, const char* test_name)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(BETA_FUNCTION_TO_TEST))
   typedef Real                   value_type;

   typedef value_type (*pg)(value_type, value_type);
#ifdef BETA_FUNCTION_TO_TEST
   pg funcp = BETA_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = boost::math::beta<value_type, value_type>;
#else
   pg funcp = boost::math::beta;
#endif

   boost::math::tools::test_result<value_type> result;

   std::cout << "Testing " << test_name << " with type " << type_name
      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

   //
   // test beta against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data, 
      bind_func<Real>(funcp, 0, 1), 
      extract_result<Real>(2));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "beta", test_name);
   std::cout << std::endl;
#endif
}
template <class T>
void test_beta(T, const char* name)
{
   //
   // The actual test data is rather verbose, so it's in a separate file
   //
   // The contents are as follows, each row of data contains
   // three items, input value a, input value b and beta(a, b):
   // 
#  include "beta_small_data.ipp"

   do_test_beta<T>(beta_small_data, name, "Beta Function: Small Values");

#  include "beta_med_data.ipp"

   do_test_beta<T>(beta_med_data, name, "Beta Function: Medium Values");

#  include "beta_exp_data.ipp"

   do_test_beta<T>(beta_exp_data, name, "Beta Function: Divergent Values");
}

template <class T>
void test_spots(T)
{
   //
   // Basic sanity checks, tolerance is 20 epsilon expressed as a percentage:
   //
   T tolerance = boost::math::tools::epsilon<T>() * 20 * 100;
   T small = boost::math::tools::epsilon<T>() / 1024;
   BOOST_CHECK_CLOSE(::boost::math::beta(static_cast<T>(1), static_cast<T>(1)), static_cast<T>(1), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::beta(static_cast<T>(1), static_cast<T>(4)), static_cast<T>(0.25), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::beta(static_cast<T>(4), static_cast<T>(1)), static_cast<T>(0.25), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::beta(small, static_cast<T>(4)), 1/small, tolerance);
   BOOST_CHECK_CLOSE(::boost::math::beta(static_cast<T>(4), small), 1/small, tolerance);
   BOOST_CHECK_CLOSE(::boost::math::beta(small, static_cast<T>(4)), 1/small, tolerance);
   BOOST_CHECK_CLOSE(::boost::math::beta(small, small / 2), boost::math::tgamma(small) * boost::math::tgamma(small / 2) / boost::math::tgamma(small + small / 2), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::beta(static_cast<T>(4), static_cast<T>(20)), static_cast<T>(0.00002823263692828910220214568040654997176736L), tolerance);
   if (boost::math::tools::digits<T>() < 100)
   {
      // Inexact input, so disable for ultra precise long doubles:
      BOOST_CHECK_CLOSE(::boost::math::beta(static_cast<T>(0.0125L), static_cast<T>(0.000023L)), static_cast<T>(43558.24045647538375006349016083320744662L), tolerance * 2);
   }

   #ifndef BOOST_MATH_NO_EXCEPTIONS
   BOOST_CHECK_THROW(boost::math::beta(static_cast<T>(0), static_cast<T>(1)), std::domain_error);
   BOOST_CHECK_THROW(boost::math::beta(static_cast<T>(-1), static_cast<T>(1)), std::domain_error);
   BOOST_CHECK_THROW(boost::math::beta(static_cast<T>(1), static_cast<T>(-1)), std::domain_error);
   BOOST_CHECK_THROW(boost::math::beta(static_cast<T>(1), static_cast<T>(0)), std::domain_error);
   #endif
}


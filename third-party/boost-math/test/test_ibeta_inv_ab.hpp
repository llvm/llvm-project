// Copyright John Maddock 2006.
// Copyright Paul A. Bristow 2007, 2009
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error

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

#ifdef TEST_GSL
#include <gsl/gsl_errno.h>
#include <gsl/gsl_message.h>
#endif

#include "handle_test_result.hpp"
#include "table_type.hpp"

#ifndef SC_
#define SC_(x) static_cast<typename table_type<T>::type>(BOOST_JOIN(x, L))
#endif

template <class Real, class T>
void test_inverses(const T& data)
{
   using namespace std;
   //typedef typename T::value_type row_type;
   typedef Real                   value_type;

   value_type precision = static_cast<value_type>(ldexp(1.0, 1-boost::math::policies::digits<value_type, boost::math::policies::policy<> >()/2)) * 100;
   if(boost::math::policies::digits<value_type, boost::math::policies::policy<> >() < 50)
      precision = 1;   // 1% or two decimal digits, all we can hope for when the input is truncated

   for(unsigned i = 0; i < data.size(); ++i)
   {
      //
      // These inverse tests are thrown off if the output of the
      // incomplete beta is too close to 1: basically there is insuffient
      // information left in the value we're using as input to the inverse
      // to be able to get back to the original value.
      //
      if(Real(data[i][5]) == 0)
      {
         BOOST_CHECK_EQUAL(boost::math::ibeta_inva(Real(data[i][1]), Real(data[i][2]), Real(data[i][5])), std::numeric_limits<value_type>::has_infinity ? std::numeric_limits<value_type>::infinity() : boost::math::tools::max_value<value_type>());
         BOOST_CHECK_EQUAL(boost::math::ibeta_invb(Real(data[i][0]), Real(data[i][2]), Real(data[i][5])), boost::math::tools::min_value<value_type>());
      }
      else if((1 - Real(data[i][5]) > 0.001) 
         && (fabs(Real(data[i][5])) > 2 * boost::math::tools::min_value<value_type>()) 
         && (fabs(Real(data[i][5])) > 2 * boost::math::tools::min_value<double>()))
      {
         value_type inv = boost::math::ibeta_inva(Real(data[i][1]), Real(data[i][2]), Real(data[i][5]));
         BOOST_CHECK_CLOSE(Real(data[i][0]), inv, precision);
         inv = boost::math::ibeta_invb(Real(data[i][0]), Real(data[i][2]), Real(data[i][5]));
         BOOST_CHECK_CLOSE(Real(data[i][1]), inv, precision);
      }
      else if(1 == Real(data[i][5]))
      {
         BOOST_CHECK_EQUAL(boost::math::ibeta_inva(Real(data[i][1]), Real(data[i][2]), Real(data[i][5])), boost::math::tools::min_value<value_type>());
         BOOST_CHECK_EQUAL(boost::math::ibeta_invb(Real(data[i][0]), Real(data[i][2]), Real(data[i][5])), std::numeric_limits<value_type>::has_infinity ? std::numeric_limits<value_type>::infinity() : boost::math::tools::max_value<value_type>());
      }

      if(Real(data[i][6]) == 0)
      {
         BOOST_CHECK_EQUAL(boost::math::ibetac_inva(Real(data[i][1]), Real(data[i][2]), Real(data[i][6])), boost::math::tools::min_value<value_type>());
         BOOST_CHECK_EQUAL(boost::math::ibetac_invb(Real(data[i][0]), Real(data[i][2]), Real(data[i][6])), std::numeric_limits<value_type>::has_infinity ? std::numeric_limits<value_type>::infinity() : boost::math::tools::max_value<value_type>());
      }
      else if((1 - Real(data[i][6]) > 0.001) 
         && (fabs(Real(data[i][6])) > 2 * boost::math::tools::min_value<value_type>()) 
         && (fabs(Real(data[i][6])) > 2 * boost::math::tools::min_value<double>()))
      {
         value_type inv = boost::math::ibetac_inva(Real(data[i][1]), Real(data[i][2]), Real(data[i][6]));
         BOOST_CHECK_CLOSE(Real(data[i][0]), inv, precision);
         inv = boost::math::ibetac_invb(Real(data[i][0]), Real(data[i][2]), Real(data[i][6]));
         BOOST_CHECK_CLOSE(Real(data[i][1]), inv, precision);
      }
      else if(Real(data[i][6]) == 1)
      {
         BOOST_CHECK_EQUAL(boost::math::ibetac_inva(Real(data[i][1]), Real(data[i][2]), Real(data[i][6])), std::numeric_limits<value_type>::has_infinity ? std::numeric_limits<value_type>::infinity() : boost::math::tools::max_value<value_type>());
         BOOST_CHECK_EQUAL(boost::math::ibetac_invb(Real(data[i][0]), Real(data[i][2]), Real(data[i][6])), boost::math::tools::min_value<value_type>());
      }
   }
}

template <class Real, class T>
void test_inverses2(const T& data, const char* type_name, const char* test_name)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(IBETA_INVA_FUNCTION_TO_TEST))
   //typedef typename T::value_type row_type;
   typedef Real                   value_type;

   typedef value_type (*pg)(value_type, value_type, value_type);
#ifdef IBETA_INVA_FUNCTION_TO_TEST
   pg funcp = IBETA_INVA_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = boost::math::ibeta_inva<value_type, value_type, value_type>;
#else
   pg funcp = boost::math::ibeta_inva;
#endif

   boost::math::tools::test_result<value_type> result;

   std::cout << "Testing " << test_name << " with type " << type_name
      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

   //
   // test ibeta_inva(T, T, T) against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(funcp, 0, 1, 2),
      extract_result<Real>(3));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "ibeta_inva", test_name);
   //
   // test ibetac_inva(T, T, T) against data:
   //
#ifdef IBETAC_INVA_FUNCTION_TO_TEST
   funcp = IBETAC_INVA_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   funcp = boost::math::ibetac_inva<value_type, value_type, value_type>;
#else
   funcp = boost::math::ibetac_inva;
#endif
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(funcp, 0, 1, 2),
      extract_result<Real>(4));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "ibetac_inva", test_name);
   //
   // test ibeta_invb(T, T, T) against data:
   //
#ifdef IBETA_INVB_FUNCTION_TO_TEST
   funcp = IBETA_INVB_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   funcp = boost::math::ibeta_invb<value_type, value_type, value_type>;
#else
   funcp = boost::math::ibeta_invb;
#endif
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(funcp, 0, 1, 2),
      extract_result<Real>(5));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "ibeta_invb", test_name);
   //
   // test ibetac_invb(T, T, T) against data:
   //
#ifdef IBETAC_INVB_FUNCTION_TO_TEST
   funcp = IBETAC_INVB_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   funcp = boost::math::ibetac_invb<value_type, value_type, value_type>;
#else
   funcp = boost::math::ibetac_invb;
#endif
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(funcp, 0, 1, 2),
      extract_result<Real>(6));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "ibetac_invb", test_name);
#endif
}

template <class T>
void test_beta(T, const char* name)
{
#if !defined(ERROR_REPORTING_MODE)
   //
   // The actual test data is rather verbose, so it's in a separate file
   //
   // The contents are as follows, each row of data contains
   // five items, input value a, input value b, integration limits x, beta(a, b, x) and ibeta(a, b, x):
   //
   std::cout << "Running sanity checks for type " << name << std::endl;

#if !defined(TEST_DATA) || (TEST_DATA == 1)
#  include "ibeta_small_data.ipp"

   test_inverses<T>(ibeta_small_data);
#endif

#if !defined(TEST_DATA) || (TEST_DATA == 2)
#  include "ibeta_data.ipp"

   test_inverses<T>(ibeta_data);
#endif

#if !defined(TEST_DATA) || (TEST_DATA == 3)
#  include "ibeta_large_data.ipp"

   test_inverses<T>(ibeta_large_data);
#endif
#endif

#if !defined(TEST_REAL_CONCEPT) || defined(FULL_TEST) || (TEST_DATA == 4)
   if(boost::is_floating_point<T>::value){
   //
   // This accuracy test is normally only enabled for "real"
   // floating point types and not for class real_concept.
   // The reason is that these tests are exceptionally slow
   // to complete when T doesn't have Lanczos support defined for it.
   //
#  include "ibeta_inva_data.ipp"

   test_inverses2<T>(ibeta_inva_data, name, "Inverse incomplete beta");
   }
#endif
   //
   // Special spot tests and bug reports:
   //
   if (std::numeric_limits<T>::has_quiet_NaN)
   {
      T n = std::numeric_limits<T>::quiet_NaN();
      BOOST_MATH_CHECK_THROW(::boost::math::ibeta_inva(n, static_cast<T>(2.125), static_cast<T>(0.125)), std::domain_error);
      BOOST_MATH_CHECK_THROW(::boost::math::ibeta_inva(static_cast<T>(2.125), n, static_cast<T>(0.125)), std::domain_error);
      BOOST_MATH_CHECK_THROW(::boost::math::ibeta_inva(static_cast<T>(2.125), static_cast<T>(1.125), n), std::domain_error);
      BOOST_MATH_CHECK_THROW(::boost::math::ibeta_invb(n, static_cast<T>(2.125), static_cast<T>(0.125)), std::domain_error);
      BOOST_MATH_CHECK_THROW(::boost::math::ibeta_invb(static_cast<T>(2.125), n, static_cast<T>(0.125)), std::domain_error);
      BOOST_MATH_CHECK_THROW(::boost::math::ibeta_invb(static_cast<T>(2.125), static_cast<T>(1.125), n), std::domain_error);
   }
   if (std::numeric_limits<T>::has_infinity)
   {
      T n = std::numeric_limits<T>::infinity();
      BOOST_MATH_CHECK_THROW(::boost::math::ibeta_invb(n, static_cast<T>(2.125), static_cast<T>(0.125)), std::domain_error);
      BOOST_MATH_CHECK_THROW(::boost::math::ibeta_invb(static_cast<T>(2.125), n, static_cast<T>(0.125)), std::domain_error);
      BOOST_MATH_CHECK_THROW(::boost::math::ibeta_invb(static_cast<T>(2.125), static_cast<T>(1.125), n), std::domain_error);
      BOOST_MATH_CHECK_THROW(::boost::math::ibeta_invb(static_cast<T>(-n), static_cast<T>(2.125), static_cast<T>(0.125)), std::domain_error);
      BOOST_MATH_CHECK_THROW(::boost::math::ibeta_invb(static_cast<T>(2.125), static_cast<T>(-n), static_cast<T>(0.125)), std::domain_error);
      BOOST_MATH_CHECK_THROW(::boost::math::ibeta_invb(static_cast<T>(2.125), static_cast<T>(1.125), static_cast<T>(-n)), std::domain_error);
   }

}


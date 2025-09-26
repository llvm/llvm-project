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
#include <boost/math/special_functions/next.hpp>  // for has_denorm_now
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/tools/stats.hpp>
#include "../include_private/boost/math/tools/test.hpp"
#include <boost/math/constants/constants.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/array.hpp>
#include "functor.hpp"

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
         BOOST_CHECK_EQUAL(boost::math::ibeta_inv(Real(data[i][0]), Real(data[i][1]), Real(data[i][5])), value_type(0));
      else if((1 - Real(data[i][5]) > 0.001) 
         && (fabs(Real(data[i][5])) > 2 * boost::math::tools::min_value<value_type>()) 
         && (fabs(Real(data[i][5])) > 2 * boost::math::tools::min_value<double>()))
      {
         value_type inv = boost::math::ibeta_inv(Real(data[i][0]), Real(data[i][1]), Real(data[i][5]));
         BOOST_CHECK_CLOSE(Real(data[i][2]), inv, precision);
      }
      else if(1 == Real(data[i][5]))
         BOOST_CHECK_EQUAL(boost::math::ibeta_inv(Real(data[i][0]), Real(data[i][1]), Real(data[i][5])), value_type(1));

      if(Real(data[i][6]) == 0)
         BOOST_CHECK_EQUAL(boost::math::ibetac_inv(Real(data[i][0]), Real(data[i][1]), Real(data[i][6])), value_type(1));
      else if((1 - Real(data[i][6]) > 0.001) 
         && (fabs(Real(data[i][6])) > 2 * boost::math::tools::min_value<value_type>()) 
         && (fabs(Real(data[i][6])) > 2 * boost::math::tools::min_value<double>()))
      {
         value_type inv = boost::math::ibetac_inv(Real(data[i][0]), Real(data[i][1]), Real(data[i][6]));
         BOOST_CHECK_CLOSE(Real(data[i][2]), inv, precision);
      }
      else if(Real(data[i][6]) == 1)
         BOOST_CHECK_EQUAL(boost::math::ibetac_inv(Real(data[i][0]), Real(data[i][1]), Real(data[i][6])), value_type(0));
   }
}

template <class Real, class T>
void test_inverses2(const T& data, const char* type_name, const char* test_name)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(IBETA_INV_FUNCTION_TO_TEST))
   //typedef typename T::value_type row_type;
   typedef Real                   value_type;

   typedef value_type (*pg)(value_type, value_type, value_type);
#ifdef IBETA_INV_FUNCTION_TO_TEST
   pg funcp = IBETA_INV_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = boost::math::ibeta_inv<value_type, value_type, value_type>;
#else
   pg funcp = boost::math::ibeta_inv;
#endif

   boost::math::tools::test_result<value_type> result;

   std::cout << "Testing " << test_name << " with type " << type_name
      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

   //
   // test ibeta_inv(T, T, T) against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(funcp, 0, 1, 2),
      extract_result<Real>(3));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "ibeta_inv", test_name);
   //
   // test ibetac_inv(T, T, T) against data:
   //
#ifdef IBETAC_INV_FUNCTION_TO_TEST
   funcp = IBETAC_INV_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   funcp = boost::math::ibetac_inv<value_type, value_type, value_type>;
#else
   funcp = boost::math::ibetac_inv;
#endif
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(funcp, 0, 1, 2),
      extract_result<Real>(4));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "ibetac_inv", test_name);
#endif
}


template <class T>
void test_beta(T, const char* name)
{
#if !defined(ERROR_REPORTING_MODE)
   (void)name;
   //
   // The actual test data is rather verbose, so it's in a separate file
   //
   // The contents are as follows, each row of data contains
   // five items, input value a, input value b, integration limits x, beta(a, b, x) and ibeta(a, b, x):
   //
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

#if !defined(TEST_DATA) || (TEST_DATA == 4)
#  include "ibeta_inv_data.ipp"

   test_inverses2<T>(ibeta_inv_data, name, "Inverse incomplete beta");
#endif
}

template <class T>
void test_spots(T)
{
   BOOST_MATH_STD_USING
   //
   // basic sanity checks, tolerance is 100 epsilon expressed as a percentage:
   //
   T tolerance = boost::math::tools::epsilon<T>() * 10000;
   BOOST_CHECK_CLOSE(
      ::boost::math::ibeta_inv(
         static_cast<T>(1),
         static_cast<T>(2),
         static_cast<T>(0.5)),
      static_cast<T>(0.29289321881345247559915563789515096071516406231153L), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::ibeta_inv(
         static_cast<T>(3),
         static_cast<T>(0.5),
         static_cast<T>(0.5)),
      static_cast<T>(0.92096723292382700385142816696980724853063433975470L), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::ibeta_inv(
         static_cast<T>(20.125),
         static_cast<T>(0.5),
         static_cast<T>(0.5)),
      static_cast<T>(0.98862133312917003480022776106012775747685870929920L), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::ibeta_inv(
         static_cast<T>(40),
         static_cast<T>(80),
         static_cast<T>(0.5)),
      static_cast<T>(0.33240456430025026300937492802591128972548660643778L), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::ibeta_inv(
         static_cast<T>(40),
         static_cast<T>(0.5),
         ldexp(T(1), -30)),
      static_cast<T>(0.624305407878048788716096298053941618358257550305573588792717L), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::ibeta_inv(
         static_cast<T>(40),
         static_cast<T>(0.5),
         static_cast<T>(1 - ldexp(T(1), -30))),
      static_cast<T>(0.99999999999999999998286262026583217516676792408012252456039L), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::ibeta_inv(
         static_cast<T>(0.5),
         static_cast<T>(40),
         static_cast<T>(ldexp(T(1), -30))),
      static_cast<T>(1.713737973416782483323207591987747543960774485649459249e-20L), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::ibeta_inv(
         static_cast<T>(0.5),
         static_cast<T>(0.75),
         static_cast<T>(ldexp(T(1), -30))),
      static_cast<T>(1.245132488513853853809715434621955746959615015005382639e-18L), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::ibeta_inv(
         static_cast<T>(0.5),
         static_cast<T>(0.5),
         static_cast<T>(0.25)),
      static_cast<T>(0.1464466094067262377995778189475754803575820311557629L), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::ibeta_inv(
         static_cast<T>(0.5),
         static_cast<T>(0.5),
         static_cast<T>(0.75)),
      static_cast<T>(0.853553390593273762200422181052424519642417968844237018294169L), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::ibeta_inv(
         static_cast<T>(1),
         static_cast<T>(5),
         static_cast<T>(0.125)),
      static_cast<T>(0.026352819384831863473794894078665766580641189002729204514544L), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::ibeta_inv(
         static_cast<T>(5),
         static_cast<T>(1),
         static_cast<T>(0.125)),
      static_cast<T>(0.659753955386447129687000985614820066516734506596709340752903L), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::ibeta_inv(
         static_cast<T>(1),
         static_cast<T>(0.125),
         static_cast<T>(0.125)),
      static_cast<T>(0.656391084194183349609374999999999999999999999999999999999999L), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::ibeta_inv(
         static_cast<T>(0.125),
         static_cast<T>(1),
         static_cast<T>(0.125)),
      static_cast<T>(5.960464477539062500000e-8), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::ibetac_inv(
         static_cast<T>(5),
         static_cast<T>(1),
         static_cast<T>(0.125)),
      static_cast<T>(0.973647180615168136526205105921334233419358810997270795485455L), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::ibetac_inv(
         static_cast<T>(1),
         static_cast<T>(5),
         static_cast<T>(0.125)),
      static_cast<T>(0.340246044613552870312999014385179933483265493403290659247096L), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::ibetac_inv(
         static_cast<T>(0.125),
         static_cast<T>(1),
         static_cast<T>(0.125)),
      static_cast<T>(0.343608915805816650390625000000000000000000000000000000000000L), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::ibetac_inv(
         static_cast<T>(1),
         static_cast<T>(0.125),
         static_cast<T>(0.125)),
      static_cast<T>(0.99999994039535522460937500000000000000000000000L), tolerance);
   //
   // Bug cases, issue 873:
   //
   if ((std::numeric_limits<T>::max)() > static_cast<T>(1e50))
   {
      BOOST_CHECK_CLOSE(
         ::boost::math::ibeta_inv(
            static_cast<T>(1e50L),
            static_cast<T>(10),
            static_cast<T>(1) / static_cast<T>(10)),
         static_cast<T>(1), tolerance);
      BOOST_CHECK_CLOSE(
         ::boost::math::ibetac_inv(
            static_cast<T>(1e50L),
            static_cast<T>(10),
            static_cast<T>(1) / static_cast<T>(10)),
         static_cast<T>(1), tolerance);
   }
   if (std::numeric_limits<T>::has_quiet_NaN)
   {
      T n = std::numeric_limits<T>::quiet_NaN();
      BOOST_MATH_CHECK_THROW(::boost::math::ibeta_inv(n, static_cast<T>(2.125), static_cast<T>(0.125)), std::domain_error);
      BOOST_MATH_CHECK_THROW(::boost::math::ibeta_inv(static_cast<T>(2.125), n, static_cast<T>(0.125)), std::domain_error);
      BOOST_MATH_CHECK_THROW(::boost::math::ibeta_inv(static_cast<T>(2.125), static_cast<T>(1.125), n), std::domain_error);
   }
   if (std::numeric_limits<T>::has_infinity)
   {
      T n = std::numeric_limits<T>::infinity();
      BOOST_MATH_CHECK_THROW(::boost::math::ibeta_inv(n, static_cast<T>(2.125), static_cast<T>(0.125)), std::domain_error);
      BOOST_MATH_CHECK_THROW(::boost::math::ibeta_inv(static_cast<T>(2.125), n, static_cast<T>(0.125)), std::domain_error);
      BOOST_MATH_CHECK_THROW(::boost::math::ibeta_inv(static_cast<T>(2.125), static_cast<T>(1.125), n), std::domain_error);
      BOOST_MATH_CHECK_THROW(::boost::math::ibeta_inv(-n, static_cast<T>(2.125), static_cast<T>(0.125)), std::domain_error);
      BOOST_MATH_CHECK_THROW(::boost::math::ibeta_inv(static_cast<T>(2.125), -n, static_cast<T>(0.125)), std::domain_error);
      BOOST_MATH_CHECK_THROW(::boost::math::ibeta_inv(static_cast<T>(2.125), static_cast<T>(1.125), -n), std::domain_error);
   }
   #ifndef SYCL_LANGUAGE_VERSION
   if (boost::math::detail::has_denorm_now<T>())
   {
      T m = std::numeric_limits<T>::denorm_min();
      T small = 2 * (std::numeric_limits<T>::min)();
      BOOST_CHECK((boost::math::isfinite)(boost::math::ibeta_inv(m, static_cast<T>(2.125), static_cast<T>(0.125))));
      BOOST_CHECK((boost::math::isfinite)(boost::math::ibeta_inv(m, m, static_cast<T>(0.125))));
      BOOST_CHECK_LT(boost::math::ibeta_inv(m, static_cast<T>(12.125), static_cast<T>(0.125)), small);
      BOOST_CHECK((boost::math::isfinite)(boost::math::ibeta_inv(static_cast<T>(2.125), m, static_cast<T>(0.125))));
      BOOST_CHECK((boost::math::isfinite)(boost::math::ibeta_inv(static_cast<T>(12.125), m, static_cast<T>(0.125))));
      BOOST_CHECK((boost::math::isfinite)(boost::math::ibeta_inv(m, m, static_cast<T>(0.125))));
   }
   #endif
}


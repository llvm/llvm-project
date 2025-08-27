// Copyright John Maddock 2006.
// Copyright Paul A. Bristow 2007, 2009
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/concepts/real_concept.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/special_functions/gamma.hpp>
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/results_collector.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
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

#define BOOST_CHECK_CLOSE_EX(a, b, prec, i) \
   {\
      unsigned int failures = boost::unit_test::results_collector.results( boost::unit_test::framework::current_test_case().p_id ).p_assertions_failed;\
      BOOST_CHECK_CLOSE(a, b, prec); \
      if(failures != boost::unit_test::results_collector.results( boost::unit_test::framework::current_test_case().p_id ).p_assertions_failed)\
      {\
         std::cerr << "Failure was at row " << i << std::endl;\
         std::cerr << std::setprecision(35); \
         std::cerr << "{ " << data[i][0] << " , " << data[i][1] << " , " << data[i][2];\
         std::cerr << " , " << data[i][3] << " , " << data[i][4] << " , " << data[i][5] << " } " << std::endl;\
      }\
   }

template <class Real, class T>
void do_test_gamma_2(const T& data, const char* type_name, const char* test_name)
{
   //
   // test gamma_p_inv(T, T) against data:
   //
   using namespace std;
   typedef Real                   value_type;

   std::cout << test_name << " with type " << type_name << std::endl;

   //
   // These sanity checks test for a round trip accuracy of one half
   // of the bits in T, unless T is type float, in which case we check
   // for just one decimal digit.  The problem here is the sensitivity
   // of the functions, not their accuracy.  This test data was generated
   // for the forward functions, which means that when it is used as
   // the input to the inverses then it is necessarily inexact.  This rounding
   // of the input is what makes the data unsuitable for use as an accuracy check,
   // and also demonstrates that you can't in general round-trip these functions.
   // It is however a useful sanity check.
   //
   value_type precision = static_cast<value_type>(ldexp(1.0, 1-boost::math::policies::digits<value_type, boost::math::policies::policy<> >()/2)) * 100;
   if(boost::math::policies::digits<value_type, boost::math::policies::policy<> >() < 50)
      precision = 1;   // 1% or two decimal digits, all we can hope for when the input is truncated to float

   for(unsigned i = 0; i < data.size(); ++i)
   {
      //
      // These inverse tests are thrown off if the output of the
      // incomplete gamma is too close to 1: basically there is insuffient
      // information left in the value we're using as input to the inverse
      // to be able to get back to the original value.
      //
      if(Real(data[i][5]) == 0)
         BOOST_CHECK_EQUAL(boost::math::gamma_p_inv(Real(data[i][0]), Real(data[i][5])), value_type(0));
      else if((1 - Real(data[i][5]) > 0.001) 
         && (fabs(Real(data[i][5])) > 2 * boost::math::tools::min_value<value_type>()) 
         && (fabs(Real(data[i][5])) > 2 * boost::math::tools::min_value<double>()))
      {
         value_type inv = boost::math::gamma_p_inv(Real(data[i][0]), Real(data[i][5]));
         BOOST_CHECK_CLOSE_EX(Real(data[i][1]), inv, precision, i);
      }
      else if(1 == Real(data[i][5]))
         BOOST_CHECK_EQUAL(boost::math::gamma_p_inv(Real(data[i][0]), Real(data[i][5])), std::numeric_limits<value_type>::has_infinity ? std::numeric_limits<value_type>::infinity() : boost::math::tools::max_value<value_type>());
      else
      {
         // not enough bits in our input to get back to x, but we should be in
         // the same ball park:
         value_type inv = boost::math::gamma_p_inv(Real(data[i][0]), Real(data[i][5]));
         BOOST_CHECK_CLOSE_EX(Real(data[i][1]), inv, 100000, i);
      }

      if(Real(data[i][3]) == 0)
         BOOST_CHECK_EQUAL(boost::math::gamma_q_inv(Real(data[i][0]), Real(data[i][3])), std::numeric_limits<value_type>::has_infinity ? std::numeric_limits<value_type>::infinity() : boost::math::tools::max_value<value_type>());
      else if((1 - Real(data[i][3]) > 0.001) && (fabs(Real(data[i][3])) > 2 * boost::math::tools::min_value<value_type>()))
      {
         value_type inv = boost::math::gamma_q_inv(Real(data[i][0]), Real(data[i][3]));
         BOOST_CHECK_CLOSE_EX(Real(data[i][1]), inv, precision, i);
      }
      else if(1 == Real(data[i][3]))
         BOOST_CHECK_EQUAL(boost::math::gamma_q_inv(Real(data[i][0]), Real(data[i][3])), value_type(0));
      else if(fabs(Real(data[i][3])) > 2 * boost::math::tools::min_value<value_type>())
      {
         // not enough bits in our input to get back to x, but we should be in
         // the same ball park:
         value_type inv = boost::math::gamma_q_inv(Real(data[i][0]), Real(data[i][3]));
         BOOST_CHECK_CLOSE_EX(Real(data[i][1]), inv, 100, i);
      }
   }
   std::cout << std::endl;
}

template <class Real, class T>
void do_test_gamma_inv(const T& data, const char* type_name, const char* test_name)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(GAMMAP_INV_FUNCTION_TO_TEST))
   typedef Real                   value_type;

   typedef value_type (*pg)(value_type, value_type);
#ifdef GAMMAP_INV_FUNCTION_TO_TEST
   pg funcp = GAMMAP_INV_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = boost::math::gamma_p_inv<value_type, value_type>;
#else
   pg funcp = boost::math::gamma_p_inv;
#endif

   boost::math::tools::test_result<value_type> result;

   std::cout << "Testing " << test_name << " with type " << type_name
      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

   //
   // test gamma_p_inv(T, T) against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(funcp, 0, 1),
      extract_result<Real>(2));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "gamma_p_inv", test_name);
   //
   // test gamma_q_inv(T, T) against data:
   //
#ifdef GAMMAQ_INV_FUNCTION_TO_TEST
   funcp = GAMMAQ_INV_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   funcp = boost::math::gamma_q_inv<value_type, value_type>;
#else
   funcp = boost::math::gamma_q_inv;
#endif
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(funcp, 0, 1),
      extract_result<Real>(3));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "gamma_q_inv", test_name);
#endif
}

template <class T>
void test_gamma(T, const char* name)
{
#if !defined(TEST_UDT) && !defined(ERROR_REPORTING_MODE)
   //
   // The actual test data is rather verbose, so it's in a separate file
   //
   // First the data for the incomplete gamma function, each
   // row has the following 6 entries:
   // Parameter a, parameter z,
   // Expected tgamma(a, z), Expected gamma_q(a, z)
   // Expected tgamma_lower(a, z), Expected gamma_p(a, z)
   //
#  include "igamma_med_data.ipp"

   do_test_gamma_2<T>(igamma_med_data, name, "Running round trip sanity checks on incomplete gamma medium sized values");

#  include "igamma_small_data.ipp"

   do_test_gamma_2<T>(igamma_small_data, name, "Running round trip sanity checks on incomplete gamma small values");

#  include "igamma_big_data.ipp"

   do_test_gamma_2<T>(igamma_big_data, name, "Running round trip sanity checks on incomplete gamma large values");

#endif

#  include "gamma_inv_data.ipp"

   do_test_gamma_inv<T>(gamma_inv_data, name, "incomplete gamma inverse(a, z) medium values");

#  include "gamma_inv_big_data.ipp"

   do_test_gamma_inv<T>(gamma_inv_big_data, name, "incomplete gamma inverse(a, z) large values");

#  include "gamma_inv_small_data.ipp"

   do_test_gamma_inv<T>(gamma_inv_small_data, name, "incomplete gamma inverse(a, z) small values");
}

template <class T>
void test_spots(T, const char* type_name)
{
   std::cout << "Running spot checks for type " << type_name << std::endl;
   //
   // basic sanity checks, tolerance is 150 epsilon expressed as a percentage:
   //
   T tolerance = boost::math::tools::epsilon<T>() * 15000;
   if(tolerance < 1e-25f)
      tolerance = 1e-25f;  // limit of test data?
   BOOST_CHECK_CLOSE(::boost::math::gamma_q_inv(static_cast<T>(1)/100, static_cast<T>(1.0/128)), static_cast<T>(0.35767144525455121503672919307647515332256996883787L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::gamma_q_inv(static_cast<T>(1)/100, static_cast<T>(0.5)), static_cast<T>(4.4655350189103486773248562646452806745879516124613e-31L), tolerance*10);
   //
   // We can't test in this region against Mathworld's data as the results produced
   // by functions.wolfram.com appear to be in error, and do *not* round trip with
   // their own version of gamma_q.  Using our output from the inverse as input to 
   // their version of gamma_q *does* round trip however.  It should be pointed out
   // that the functions in this area are very sensitive with nearly infinite
   // first derivatives, it's also questionable how useful these functions are
   // in this part of the domain.
   //
   //BOOST_CHECK_CLOSE(::boost::math::gamma_q_inv(static_cast<T>(1e-2), static_cast<T>(1.0-1.0/128)), static_cast<T>(3.8106736649978161389878528903698068142257930575497e-181L), tolerance);
   //
   BOOST_CHECK_CLOSE(::boost::math::gamma_q_inv(static_cast<T>(0.5), static_cast<T>(1.0/128)), static_cast<T>(3.5379794687984498627918583429482809311448951189097L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::gamma_q_inv(static_cast<T>(0.5), static_cast<T>(1.0/2)), static_cast<T>(0.22746821155978637597125832348982469815821055329511L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::gamma_q_inv(static_cast<T>(0.5), static_cast<T>(1.0-1.0/128)), static_cast<T>(0.000047938431649305382237483273209405461203600840052182L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::gamma_q_inv(static_cast<T>(10), static_cast<T>(1.0/128)), static_cast<T>(19.221865946801723949866005318845155649972164294057L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::gamma_q_inv(static_cast<T>(10), static_cast<T>(1.0/2)), static_cast<T>(9.6687146147141311517500637401166726067778162022664L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::gamma_q_inv(static_cast<T>(10), static_cast<T>(1.0-1.0/128)), static_cast<T>(3.9754602513640844712089002210120603689809432130520L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::gamma_q_inv(static_cast<T>(10000), static_cast<T>(1.0/128)), static_cast<T>(10243.369973939134157953734588122880006091919872879L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::gamma_q_inv(static_cast<T>(10000), static_cast<T>(1.0/2)), static_cast<T>(9999.6666686420474237369661574633153551436435884101L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::gamma_q_inv(static_cast<T>(10000), static_cast<T>(1.0-1.0/128)), static_cast<T>(9759.8597223369324083191194574874497413261589080204L), tolerance);
}


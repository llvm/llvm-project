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
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/tools/stats.hpp>
#include <boost/math/tools/test.hpp>
#include <boost/array.hpp>
#include "functor.hpp"

#include "handle_test_result.hpp"
#include "table_type.hpp"

#ifndef SC_
#define SC_(x) static_cast<typename table_type<T>::type>(BOOST_JOIN(x, L))
#endif

template <class Real>
struct negative_tgamma_ratio
{
   template <class Row>
   Real operator()(const Row& row)
   {
#ifdef TGAMMA_DELTA_RATIO_FUNCTION_TO_TEST
      return TGAMMA_DELTA_RATIO_FUNCTION_TO_TEST(Real(row[0]), Real(-Real(row[1])));
#else
      return boost::math::tgamma_delta_ratio(Real(row[0]), Real(-Real(row[1])));
#endif
   }
};

template <class Real, class T>
void do_test_tgamma_delta_ratio(const T& data, const char* type_name, const char* test_name)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(TGAMMA_DELTA_RATIO_FUNCTION_TO_TEST))
   typedef Real                   value_type;

   typedef value_type (*pg)(value_type, value_type);
#ifdef TGAMMA_DELTA_RATIO_FUNCTION_TO_TEST
   pg funcp = TGAMMA_DELTA_RATIO_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = boost::math::tgamma_delta_ratio<value_type, value_type>;
#else
   pg funcp = boost::math::tgamma_delta_ratio;
#endif

   boost::math::tools::test_result<value_type> result;

   std::cout << "Testing " << test_name << " with type " << type_name
      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

   //
   // test tgamma_delta_ratio against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(funcp, 0, 1),
      extract_result<Real>(2));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "tgamma_delta_ratio", test_name);
   result = boost::math::tools::test_hetero<Real>(
      data,
      negative_tgamma_ratio<Real>(),
      extract_result<Real>(3));
   std::string s(test_name);
   s += " (negative delta)";
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "tgamma_delta_ratio", s.c_str());
#endif
}

template <class Real, class T>
void do_test_tgamma_ratio(const T& data, const char* type_name, const char* test_name)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(TGAMMA_RATIO_FUNCTION_TO_TEST))
   typedef Real                   value_type;

   typedef value_type (*pg)(value_type, value_type);
#ifdef TGAMMA_RATIO_FUNCTION_TO_TEST
   pg funcp = TGAMMA_RATIO_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = boost::math::tgamma_ratio<value_type, value_type>;
#else
   pg funcp = boost::math::tgamma_ratio;
#endif

   boost::math::tools::test_result<value_type> result;

   std::cout << "Testing " << test_name << " with type " << type_name
      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

   //
   // test tgamma_ratio against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(funcp, 0, 1),
      extract_result<Real>(2));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "tgamma_ratio", test_name);
#endif
}

template <class T>
void test_tgamma_ratio(T, const char* name)
{
   //
   // The actual test data is rather verbose, so it's in a separate file
   //
#  include "tgamma_delta_ratio_data.ipp"

   do_test_tgamma_delta_ratio<T>(tgamma_delta_ratio_data, name, "tgamma + small delta ratios");

#  include "tgamma_delta_ratio_int.ipp"

   do_test_tgamma_delta_ratio<T>(tgamma_delta_ratio_int, name, "tgamma + small integer ratios");

#  include "tgamma_delta_ratio_int2.ipp"

   do_test_tgamma_delta_ratio<T>(tgamma_delta_ratio_int2, name, "integer tgamma ratios");

#  include "tgamma_ratio_data.ipp"

   do_test_tgamma_ratio<T>(tgamma_ratio_data, name, "tgamma ratios");

}

template <class T>
void test_spots(T, const char*)
{
#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:4127 4756)
#endif
   //
   // A few special spot tests:
   //
   BOOST_MATH_STD_USING
   T tol = boost::math::tools::epsilon<T>() * 20;
   if(std::numeric_limits<T>::max_exponent > 200)
   {
      BOOST_CHECK_CLOSE_FRACTION(boost::math::tgamma_ratio(T(ldexp(T(1), -500)), T(180.25)), T(8.0113754557649679470816892372669519037339812035512e-178L), 3 * tol);
      BOOST_CHECK_CLOSE_FRACTION(boost::math::tgamma_ratio(T(ldexp(T(1), -525)), T(192.25)), T(1.5966560279353205461166489184101261541784867035063e-197L), 3 * tol);
      BOOST_CHECK_CLOSE_FRACTION(boost::math::tgamma_ratio(T(182.25), T(ldexp(T(1), -500))), T(4.077990437521002194346763299159975185747917450788e+181L), 3 * tol);
      BOOST_CHECK_CLOSE_FRACTION(boost::math::tgamma_ratio(T(193.25), T(ldexp(T(1), -525))), T(1.2040790040958522422697601672703926839178050326148e+199L), 3 * tol);
      BOOST_CHECK_CLOSE_FRACTION(boost::math::tgamma_ratio(T(193.25), T(194.75)), T(0.00037151765099653237632823607820104961270831942138159L), 3 * tol);
   }
   BOOST_MATH_CHECK_THROW(boost::math::tgamma_ratio(T(0), T(2)), std::domain_error);
   BOOST_MATH_CHECK_THROW(boost::math::tgamma_ratio(T(2), T(0)), std::domain_error);
   BOOST_MATH_CHECK_THROW(boost::math::tgamma_ratio(T(-1), T(2)), std::domain_error);
   BOOST_MATH_CHECK_THROW(boost::math::tgamma_ratio(T(2), T(-1)), std::domain_error);
   if(std::numeric_limits<T>::has_infinity)
   {
      BOOST_MATH_CHECK_THROW(boost::math::tgamma_ratio(std::numeric_limits<T>::infinity(), T(2)), std::domain_error);
      BOOST_MATH_CHECK_THROW(boost::math::tgamma_ratio(T(2), std::numeric_limits<T>::infinity()), std::domain_error);
   }
   //
   // Some bug cases from Rocco Romeo:
   //
   if(std::numeric_limits<T>::min_exponent < -1020)
   {
      BOOST_CHECK_CLOSE_FRACTION(boost::math::tgamma_ratio(T(ldexp(T(1), -1020)), T(100)), T(1.20390418056093374068585549133304106854441830616070800417660e151L), tol);
      BOOST_CHECK_CLOSE_FRACTION(boost::math::tgamma_ratio(T(ldexp(T(1), -1020)), T(150)), T(2.94980580122226729924781231239336413648584663386992050529324e46L), tol);
      BOOST_CHECK_CLOSE_FRACTION(boost::math::tgamma_ratio(T(ldexp(T(1), -1020)), T(180)), T(1.00669209319561468911303652019446665496398881230516805140750e-20L), tol);
      BOOST_CHECK_CLOSE_FRACTION(boost::math::tgamma_ratio(T(ldexp(T(1), -1020)), T(220)), T(1.08230263539550701700187215488533416834407799907721731317227e-112L), tol);
      BOOST_CHECK_CLOSE_FRACTION(boost::math::tgamma_ratio(T(ldexp(T(1), -1020)), T(260)), T(7.62689807594728483940172477902929825624752380292252137809206e-208L), tol);
      BOOST_CHECK_CLOSE_FRACTION(boost::math::tgamma_ratio(T(ldexp(T(1), -1020)), T(290)), T(5.40206998243175672775582485422795773284966068149812072521290e-281L), tol);
      BOOST_CHECK_CLOSE_FRACTION(boost::math::tgamma_delta_ratio(T(ldexp(T(1), -1020)), T(ldexp(T(1), -1020))), T(2), tol);
      if(0 != ldexp(T(1), -1074))
      {
         // This is denorm_min at double precision:
         BOOST_CHECK_CLOSE_FRACTION(boost::math::tgamma_ratio(T(ldexp(T(1), -1074)), T(200)), T(5.13282785052571536804189023927976812551830809667482691717029e-50L), tol * 50);
         BOOST_CHECK_CLOSE_FRACTION(boost::math::tgamma_ratio(T(200), T(ldexp(T(1), -1074))), T(1.94824379293682687942882944294875087145333536754699303593931e49L), tol * 10);
         BOOST_CHECK_CLOSE_FRACTION(boost::math::tgamma_delta_ratio(T(ldexp(T(1), -1074)), T(200)), T(5.13282785052571536804189023927976812551830809667482691717029e-50L), tol * 10);
         BOOST_CHECK_CLOSE_FRACTION(boost::math::tgamma_delta_ratio(T(200), T(ldexp(T(1), -1074))), T(1), tol);
      }
   }
}

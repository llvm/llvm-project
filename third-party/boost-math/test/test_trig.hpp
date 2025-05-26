// Copyright John Maddock 2014.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/concepts/real_concept.hpp>
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
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
void do_test_trig(const T& data, const char* type_name, const char* test_name)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(SIN_PI_RATIO_FUNCTION_TO_TEST))
   typedef Real                   value_type;

   typedef value_type (*pg)(value_type);
#ifdef SIN_PI_RATIO_FUNCTION_TO_TEST
   pg funcp = SIN_PI_RATIO_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = boost::math::sin_pi<value_type>;
#else
   pg funcp = boost::math::sin_pi;
#endif

   boost::math::tools::test_result<value_type> result;

   std::cout << "Testing " << test_name << " with type " << type_name
      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

   //
   // test sin_pi against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(funcp, 0),
      extract_result<Real>(1));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "sin_pi", test_name);
   //
   // test cos_pi against data:
   //
#ifdef COS_PI_RATIO_FUNCTION_TO_TEST
   funcp = COS_PI_RATIO_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   funcp = boost::math::cos_pi<value_type>;
#else
   funcp = boost::math::cos_pi;
#endif
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(funcp, 0),
      extract_result<Real>(2));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "cos_pi", test_name);
   std::cout << std::endl;
#endif
}

template <class T>
void test_trig(T, const char* name)
{
   //
   // The actual test data is rather verbose, so it's in a separate file
   //
   // The contents are as follows, each row of data contains
   // three items, input value a, input value b and erf(a, b):
   //
#  include "trig_data.ipp"

   do_test_trig<T>(trig_data, name, "sin_pi and cos_pi");

#  include "trig_data2.ipp"

   do_test_trig<T>(trig_data2, name, "sin_pi and cos_pi near integers and half integers");
}


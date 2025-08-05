// Copyright John Maddock 2006.
// Copyright Paul A. Bristow 2007, 2009
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error

#include <boost/math/concepts/real_concept.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/tools/stats.hpp>
#include <boost/math/tools/test.hpp>
#include <boost/math/tools/big_constant.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/array.hpp>
#include "functor.hpp"

#include "handle_test_result.hpp"
#include "table_type.hpp"

#include <boost/math/special_functions/hypergeometric_1F1.hpp>
#include <boost/math/quadrature/exp_sinh.hpp>

#ifdef _MSC_VER
#pragma warning(disable:4127)
#endif

template <class Real, class T>
void do_test_1F1(const T& data, const char* type_name, const char* test_name)
{
   typedef Real                   value_type;

   typedef value_type(*pg)(value_type, value_type, value_type);
#if defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = boost::math::log_hypergeometric_1F1<value_type, value_type>;
#else
   pg funcp = boost::math::log_hypergeometric_1F1;
#endif

   boost::math::tools::test_result<value_type> result;

   std::cout << "Testing " << test_name << " with type " << type_name
      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

   //
   // test hypergeometric_2F0 against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(funcp, 0, 1, 2),
      extract_result<Real>(3));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "log_hypergeometric_1F1", test_name);
   std::cout << std::endl;
}

#ifndef SC_
#define SC_(x) BOOST_MATH_BIG_CONSTANT(T, 1000000, x)
#endif

template <class T>
void test_spots1(T, const char* type_name)
{
#include "hypergeometric_1f1_log_large.ipp"

   do_test_1F1<T>(hypergeometric_1f1_log_large, type_name, "Large random values - log");
}

template <class T>
void test_spots2(T, const char* type_name)
{
#include "hypergeometric_1f1_log_large_unsolved.ipp"

   do_test_1F1<T>(hypergeometric_1f1_log_large_unsolved, type_name, "Large random values - log - unsolved");
}

template <class T>
void test_spots_bugs(T, const char* type_name)
{
   static const std::array<std::array<T, 4>, 7> hypergeometric_1F1_bugs = { {
    // Found while investigating https://github.com/boostorg/math/issues/1034
    {{ 21156.0f, 21156.0f, 11322, SC_(11322.0)}},
    {{ 21156.0f, 21155.0f, 11322, SC_(11322.428655862323560632951631114666466652986288119296800531328684)}},
    {{ 21156.0f, 21154.0f, 11322, SC_(11322.857338938931770780542471014439235046236959098048505808665509)}},
    {{ 21156.0f, 21154.5f, 11322, SC_(11322.642993998700652342915766513423502332460377941025163971463001)}},
    {{ 21156.0f, 21155.0f - 1.0f / 128, 11322, SC_(11322.43200484338133063401686149227756707)}},
    {{ 21156.0f, 21154.5f - 1.0f / 128, 11322, SC_(11322.646343086066466097278446364687150256282052775170519455915995)}},
    {{ 21156.0f, 21154.0f + 1.0f / 128, 11322, SC_(11322.85398974691465958225700429672975704)}},
   } };

   do_test_1F1<T>(hypergeometric_1F1_bugs, type_name, "Large random values - log - bug cases");
}

template <class T>
void test_spots(T z, const char* type_name)
{
   test_spots1(z, type_name);
   test_spots_bugs(z, type_name);
#ifdef TEST_UNSOLVED
   test_spots2(z, type_name);
#endif
}


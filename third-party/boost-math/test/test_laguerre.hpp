// Copyright John Maddock 2006.
// Copyright Paul A. Bristow 2007, 2009
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/concepts/real_concept.hpp>
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/array.hpp>
#include "functor.hpp"

#include "handle_test_result.hpp"
#include "table_type.hpp"

#ifndef SC_
#define SC_(x) static_cast<typename table_type<T>::type>(BOOST_JOIN(x, L))
#endif

template <class Real, class T>
void do_test_laguerre2(const T& data, const char* type_name, const char* test_name)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(LAGUERRE_FUNCTION_TO_TEST))
   typedef Real                   value_type;

   typedef value_type (*pg)(unsigned, value_type);
#ifdef LAGUERRE_FUNCTION_TO_TEST
   pg funcp = LAGUERRE_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = boost::math::laguerre<value_type>;
#else
   pg funcp = boost::math::laguerre;
#endif

   boost::math::tools::test_result<value_type> result;

   std::cout << "Testing " << test_name << " with type " << type_name
      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

   //
   // test laguerre against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data, 
      bind_func_int1<Real>(funcp, 0, 1), 
      extract_result<Real>(2));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "laguerre(n, x)", test_name);

   std::cout << std::endl;
#endif
}

template <class Real, class T>
void do_test_laguerre3(const T& data, const char* type_name, const char* test_name)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(ASSOC_LAGUERRE_FUNCTION_TO_TEST))
   typedef Real                   value_type;

   typedef value_type (*pg)(unsigned, unsigned, value_type);
#ifdef ASSOC_LAGUERRE_FUNCTION_TO_TEST
   pg funcp = ASSOC_LAGUERRE_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = boost::math::laguerre<unsigned, value_type>;
#else
   pg funcp = boost::math::laguerre;
#endif

   boost::math::tools::test_result<value_type> result;

   std::cout << "Testing " << test_name << " with type " << type_name
      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

   //
   // test laguerre against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data, 
      bind_func_int2<Real>(funcp, 0, 1, 2), 
      extract_result<Real>(3));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "laguerre(n, m, x)", test_name);
   std::cout << std::endl;
#endif
}

template <class T>
void test_laguerre(T, const char* name)
{
   //
   // The actual test data is rather verbose, so it's in a separate file
   //
   // The contents are as follows, each row of data contains
   // three items, input value a, input value b and erf(a, b):
   // 
#  include "laguerre2.ipp"

   do_test_laguerre2<T>(laguerre2, name, "Laguerre Polynomials");

#  include "laguerre3.ipp"

   do_test_laguerre3<T>(laguerre3, name, "Associated Laguerre Polynomials");
}

template <class T>
void test_spots(T, const char* t)
{
   std::cout << "Testing basic sanity checks for type " << t << std::endl;
   //
   // basic sanity checks, tolerance is 100 epsilon:
   //
   T tolerance = boost::math::tools::epsilon<T>() * 100;
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::laguerre(1, static_cast<T>(0.5L)), static_cast<T>(0.5L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::laguerre(4, static_cast<T>(0.5L)), static_cast<T>(-0.3307291666666666666666666666666666666667L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::laguerre(7, static_cast<T>(0.5L)), static_cast<T>(-0.5183392237103174603174603174603174603175L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::laguerre(20, static_cast<T>(0.5L)), static_cast<T>(0.3120174870800154148915399248893113634676L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::laguerre(50, static_cast<T>(0.5L)), static_cast<T>(-0.3181388060269979064951118308575628226834L), tolerance);

   BOOST_CHECK_CLOSE_FRACTION(::boost::math::laguerre(1, static_cast<T>(-0.5L)), static_cast<T>(1.5L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::laguerre(4, static_cast<T>(-0.5L)), static_cast<T>(3.835937500000000000000000000000000000000L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::laguerre(7, static_cast<T>(-0.5L)), static_cast<T>(7.950934709821428571428571428571428571429L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::laguerre(20, static_cast<T>(-0.5L)), static_cast<T>(76.12915699869631476833699787070874048223L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::laguerre(50, static_cast<T>(-0.5L)), static_cast<T>(2307.428631277506570629232863491518399720L), tolerance);

   BOOST_CHECK_CLOSE_FRACTION(::boost::math::laguerre(1, static_cast<T>(4.5L)), static_cast<T>(-3.500000000000000000000000000000000000000L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::laguerre(4, static_cast<T>(4.5L)), static_cast<T>(0.08593750000000000000000000000000000000000L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::laguerre(7, static_cast<T>(4.5L)), static_cast<T>(-1.036928013392857142857142857142857142857L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::laguerre(20, static_cast<T>(4.5L)), static_cast<T>(1.437239150257817378525582974722170737587L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::laguerre(50, static_cast<T>(4.5L)), static_cast<T>(-0.7795068145562651416494321484050019245248L), tolerance);

   BOOST_CHECK_CLOSE_FRACTION(::boost::math::laguerre(4, 5, static_cast<T>(0.5L)), static_cast<T>(88.31510416666666666666666666666666666667L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::laguerre(10, 0, static_cast<T>(2.5L)), static_cast<T>(-0.8802526766660982969576719576719576719577L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::laguerre(10, 1, static_cast<T>(4.5L)), static_cast<T>(1.564311458042689732142857142857142857143L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::laguerre(10, 6, static_cast<T>(8.5L)), static_cast<T>(20.51596541066649098875661375661375661376L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::laguerre(10, 12, static_cast<T>(12.5L)), static_cast<T>(-199.5560968456234671241181657848324514991L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::laguerre(50, 40, static_cast<T>(12.5L)), static_cast<T>(-4.996769495006119488583146995907246595400e16L), tolerance);
}


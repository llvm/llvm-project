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
#include <boost/math/special_functions/ellint_rj.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/array.hpp>
#include <boost/random.hpp>
#include "functor.hpp"

#include "handle_test_result.hpp"
#include "table_type.hpp"

#ifndef SC_
#define SC_(x) static_cast<typename table_type<T>::type>(BOOST_JOIN(x, L))
#endif

template <class Real, typename T>
void do_test_ellint_rf(T& data, const char* type_name, const char* test)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(ELLINT_RF_FUNCTION_TO_TEST))
   typedef Real                   value_type;

   std::cout << "Testing: " << test << std::endl;

#ifdef ELLINT_RF_FUNCTION_TO_TEST
   value_type(*fp)(value_type, value_type, value_type) = ELLINT_RF_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
    value_type (*fp)(value_type, value_type, value_type) = boost::math::ellint_rf<value_type, value_type, value_type>;
#else
    value_type (*fp)(value_type, value_type, value_type) = boost::math::ellint_rf;
#endif
    boost::math::tools::test_result<value_type> result;
 
    result = boost::math::tools::test_hetero<Real>(
      data, 
      bind_func<Real>(fp, 0, 1, 2),
      extract_result<Real>(3));
   handle_test_result(result, data[result.worst()], result.worst(), 
      type_name, "ellint_rf", test);

   std::cout << std::endl;
#endif
}

template <class Real, typename T>
void do_test_ellint_rc(T& data, const char* type_name, const char* test)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(ELLINT_RC_FUNCTION_TO_TEST))
   typedef Real                   value_type;

   std::cout << "Testing: " << test << std::endl;

#ifdef ELLINT_RC_FUNCTION_TO_TEST
   value_type(*fp)(value_type, value_type) = ELLINT_RC_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
    value_type (*fp)(value_type, value_type) = boost::math::ellint_rc<value_type, value_type>;
#else
    value_type (*fp)(value_type, value_type) = boost::math::ellint_rc;
#endif
    boost::math::tools::test_result<value_type> result;
 
    result = boost::math::tools::test_hetero<Real>(
      data, 
      bind_func<Real>(fp, 0, 1),
      extract_result<Real>(2));
      handle_test_result(result, data[result.worst()], result.worst(), 
      type_name, "ellint_rc", test);

   std::cout << std::endl;
#endif
}

template <class Real, typename T>
void do_test_ellint_rj(T& data, const char* type_name, const char* test)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(ELLINT_RJ_FUNCTION_TO_TEST))
   typedef Real                   value_type;

   std::cout << "Testing: " << test << std::endl;

#ifdef ELLINT_RJ_FUNCTION_TO_TEST
   value_type(*fp)(value_type, value_type, value_type, value_type) = ELLINT_RJ_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
    value_type (*fp)(value_type, value_type, value_type, value_type) = boost::math::ellint_rj<value_type, value_type, value_type, value_type>;
#else
    value_type (*fp)(value_type, value_type, value_type, value_type) = boost::math::ellint_rj;
#endif
    boost::math::tools::test_result<value_type> result;
 
    result = boost::math::tools::test_hetero<Real>(
      data, 
      bind_func<Real>(fp, 0, 1, 2, 3),
      extract_result<Real>(4));
      handle_test_result(result, data[result.worst()], result.worst(), 
      type_name, "ellint_rj", test);

   std::cout << std::endl;
#endif
}

template <class Real, typename T>
void do_test_ellint_rd(T& data, const char* type_name, const char* test)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(ELLINT_RD_FUNCTION_TO_TEST))
   typedef Real                   value_type;

   std::cout << "Testing: " << test << std::endl;

#ifdef ELLINT_RD_FUNCTION_TO_TEST
   value_type(*fp)(value_type, value_type, value_type) = ELLINT_RD_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
    value_type (*fp)(value_type, value_type, value_type) = boost::math::ellint_rd<value_type, value_type, value_type>;
#else
    value_type (*fp)(value_type, value_type, value_type) = boost::math::ellint_rd;
#endif
    boost::math::tools::test_result<value_type> result;
 
    result = boost::math::tools::test_hetero<Real>(
      data, 
      bind_func<Real>(fp, 0, 1, 2),
      extract_result<Real>(3));
    handle_test_result(result, data[result.worst()], result.worst(), 
      type_name, "ellint_rd", test);

   std::cout << std::endl;
#endif
}

template <class Real, typename T>
void do_test_ellint_rg(T& data, const char* type_name, const char* test)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(ELLINT_RD_FUNCTION_TO_TEST))
   typedef Real                   value_type;

   std::cout << "Testing: " << test << std::endl;

#ifdef ELLINT_RG_FUNCTION_TO_TEST
   value_type(*fp)(value_type, value_type, value_type) = ELLINT_RG_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   value_type(*fp)(value_type, value_type, value_type) = boost::math::ellint_rg<value_type, value_type, value_type>;
#else
   value_type(*fp)(value_type, value_type, value_type) = boost::math::ellint_rg;
#endif
   boost::math::tools::test_result<value_type> result;

   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(fp, 0, 1, 2),
      extract_result<Real>(3));
   handle_test_result(result, data[result.worst()], result.worst(),
      type_name, "ellint_rg", test);

   std::cout << std::endl;
#endif
}

#if !defined(TEST1) && !defined(TEST2) && !defined(TEST3) && !defined(TEST4)
#define TEST1
#define TEST2
#define TEST3
#define TEST4
#endif

#ifdef TEST1

template <typename T>
void t1(T, const char* type_name)
{
#include "ellint_rf_data.ipp"

   do_test_ellint_rf<T>(ellint_rf_data, type_name, "RF: Random data");
}

template <typename T>
void t2(T, const char* type_name)
{
#include "ellint_rf_xxx.ipp"

   do_test_ellint_rf<T>(ellint_rf_xxx, type_name, "RF: x = y = z");
}

template <typename T>
void t3(T, const char* type_name)
{
#include "ellint_rf_xyy.ipp"

   do_test_ellint_rf<T>(ellint_rf_xyy, type_name, "RF: x = y or y = z or x = z");
}

template <typename T>
void t4(T, const char* type_name)
{
#include "ellint_rf_0yy.ipp"

   do_test_ellint_rf<T>(ellint_rf_0yy, type_name, "RF: x = 0, y = z");
}

template <typename T>
void t5(T, const char* type_name)
{
#include "ellint_rf_xy0.ipp"

   do_test_ellint_rf<T>(ellint_rf_xy0, type_name, "RF: z = 0");
}

#endif
#ifdef TEST2

template <typename T>
void t6(T, const char* type_name)
{
#include "ellint_rc_data.ipp"

   do_test_ellint_rc<T>(ellint_rc_data, type_name, "RC: Random data");

   //
   // Error handling:
   //
   BOOST_CHECK_THROW(boost::math::ellint_rc(T(-1), T(1)), std::domain_error);
   BOOST_CHECK_THROW(boost::math::ellint_rc(T(1), T(0)), std::domain_error);
}

template <typename T>
void t7(T, const char* type_name)
{
#include "ellint_rj_data.ipp"

   do_test_ellint_rj<T>(ellint_rj_data, type_name, "RJ: Random data");
}

template <typename T>
void t8(T, const char* type_name)
{
#include "ellint_rj_e4.ipp"

   do_test_ellint_rj<T>(ellint_rj_e4, type_name, "RJ: 4 Equal Values");
}

template <typename T>
void t9(T, const char* type_name)
{
#include "ellint_rj_e3.ipp"

   do_test_ellint_rj<T>(ellint_rj_e3, type_name, "RJ: 3 Equal Values");
}

template <typename T>
void t10(T, const char* type_name)
{
#include "ellint_rj_e2.ipp"

   do_test_ellint_rj<T>(ellint_rj_e2, type_name, "RJ: 2 Equal Values");
}

template <typename T>
void t11(T, const char* type_name)
{
#include "ellint_rj_zp.ipp"

   do_test_ellint_rj<T>(ellint_rj_zp, type_name, "RJ: Equal z and p");
}

#endif
#ifdef TEST3

template <typename T>
void t12(T, const char* type_name)
{
#include "ellint_rd_data.ipp"

   do_test_ellint_rd<T>(ellint_rd_data, type_name, "RD: Random data");

   BOOST_CHECK_THROW(boost::math::ellint_rd(T(-1), T(1), T(2)), std::domain_error);
   BOOST_CHECK_THROW(boost::math::ellint_rd(T(1), T(-1), T(2)), std::domain_error);
   BOOST_CHECK_THROW(boost::math::ellint_rd(T(1), T(2), T(0)), std::domain_error);
   BOOST_CHECK_THROW(boost::math::ellint_rd(T(0), T(0), T(2)), std::domain_error);
   BOOST_CHECK_EQUAL(boost::math::ellint_rd(T(0.5), T(0), T(0.75)), boost::math::ellint_rd(T(0), T(0.5), T(0.75)));
}

template <typename T>
void t13(T, const char* type_name)
{
#include "ellint_rd_xyy.ipp"

   do_test_ellint_rd<T>(ellint_rd_xyy, type_name, "RD: y = z");
}

template <typename T>
void t14(T, const char* type_name)
{
#include "ellint_rd_xxz.ipp"

   do_test_ellint_rd<T>(ellint_rd_xxz, type_name, "RD: x = y");
}

template <typename T>
void t15(T, const char* type_name)
{
#include "ellint_rd_0yy.ipp"

   do_test_ellint_rd<T>(ellint_rd_0yy, type_name, "RD: x = 0, y = z");
}

template <typename T>
void t16(T, const char* type_name)
{
#include "ellint_rd_xxx.ipp"

   do_test_ellint_rd<T>(ellint_rd_xxx, type_name, "RD: x = y = z");
}

template <typename T>
void t17(T, const char* type_name)
{
#include "ellint_rd_0xy.ipp"

   do_test_ellint_rd<T>(ellint_rd_0xy, type_name, "RD: x = 0");
}

#endif
#ifdef TEST4

template <typename T>
void t18(T, const char* type_name)
{
#include "ellint_rg.ipp"

   do_test_ellint_rg<T>(ellint_rg, type_name, "RG: Random Data");
}

template <typename T>
void t19(T, const char* type_name)
{
#include "ellint_rg_00x.ipp"

   do_test_ellint_rg<T>(ellint_rg_00x, type_name, "RG: two values 0");
}

template <typename T>
void t20(T, const char* type_name)
{
#include "ellint_rg_xxx.ipp"

   do_test_ellint_rg<T>(ellint_rg_xxx, type_name, "RG: All values the same or zero");
}

template <typename T>
void t21(T, const char* type_name)
{
#include "ellint_rg_xyy.ipp"

   do_test_ellint_rg<T>(ellint_rg_xyy, type_name, "RG: two values the same");
}

template <typename T>
void t22(T, const char* type_name)
{
#include "ellint_rg_xy0.ipp"

   do_test_ellint_rg<T>(ellint_rg_xy0, type_name, "RG: one value zero");
}

#endif

template <typename T>
void test_spots(T val, const char* type_name)
{
#ifndef TEST_UDT
   using namespace boost::math;
   using namespace std;
   // Spot values from Numerical Computation of Real or Complex 
   // Elliptic Integrals, B. C. Carlson: http://arxiv.org/abs/math.CA/9409227
   // RF:
   T tolerance = (std::max)(T(1e-13f), tools::epsilon<T>() * 5) * 100; // Note 5eps expressed as a percentage!!!
   T eps2 = 5 * tools::epsilon<T>();
   BOOST_CHECK_CLOSE(ellint_rf(T(1), T(2), T(0)), T(1.3110287771461), tolerance);
   BOOST_CHECK_CLOSE(ellint_rf(T(0.5), T(1), T(0)), T(1.8540746773014), tolerance);
   BOOST_CHECK_CLOSE(ellint_rf(T(2), T(3), T(4)), T(0.58408284167715), tolerance);

   BOOST_CHECK_THROW(ellint_rf(T(-1), T(1), T(1)), std::domain_error);
   BOOST_CHECK_THROW(ellint_rf(T(1), T(-1), T(1)), std::domain_error);
   BOOST_CHECK_THROW(ellint_rf(T(1), T(1), T(-1)), std::domain_error);
   BOOST_CHECK_THROW(ellint_rf(T(0), T(0), T(1)), std::domain_error);
   BOOST_CHECK_THROW(ellint_rf(T(1), T(0), T(0)), std::domain_error);
   BOOST_CHECK_THROW(ellint_rf(T(0), T(1), T(0)), std::domain_error);
   BOOST_CHECK_THROW(ellint_rf(T(0), T(0), T(0)), std::domain_error);

   BOOST_CHECK_EQUAL(ellint_rf(T(0), T(2), T(2)), ellint_rf(T(2), T(0), T(2)));
   BOOST_CHECK_EQUAL(ellint_rf(T(2), T(2), T(0)), ellint_rf(T(2), T(0), T(2)));
   BOOST_CHECK_EQUAL(ellint_rf(T(0), T(2), T(3)), ellint_rf(T(2), T(3), T(0)));
   BOOST_CHECK_EQUAL(ellint_rf(T(0), T(2), T(3)), ellint_rf(T(3), T(2), T(0)));
   BOOST_CHECK_EQUAL(ellint_rf(T(0), T(2), T(3)), ellint_rf(T(3), T(0), T(2)));
   BOOST_CHECK_EQUAL(ellint_rf(T(0), T(2), T(3)), ellint_rf(T(2), T(0), T(3)));

   // RC:
   BOOST_CHECK_CLOSE_FRACTION(ellint_rc(T(0), T(1)/4), boost::math::constants::pi<T>(), eps2);
   BOOST_CHECK_CLOSE_FRACTION(ellint_rc(T(9)/4, T(2)), boost::math::constants::ln_two<T>(), eps2);
   BOOST_CHECK_CLOSE_FRACTION(ellint_rc(T(1) / 4, T(-2)), boost::math::constants::ln_two<T>() / 3, eps2);

   BOOST_CHECK_CLOSE_FRACTION(boost::math::detail::ellint_rc1p_imp(T(-2), boost::math::policies::policy<>()), ellint_rc(T(1), T(-1)), eps2);
   BOOST_CHECK_CLOSE_FRACTION(boost::math::detail::ellint_rc1p_imp(T(-0.75), boost::math::policies::policy<>()), ellint_rc(T(1), T(0.25)), eps2);

   // RJ:
   BOOST_CHECK_CLOSE(ellint_rj(T(0), T(1), T(2), T(3)), T(0.77688623778582), tolerance);
   BOOST_CHECK_CLOSE(ellint_rj(T(2), T(3), T(4), T(5)), T(0.14297579667157), tolerance);
   BOOST_CHECK_CLOSE(ellint_rj(T(2), T(3), T(4), T(-0.5)), T(0.24723819703052), tolerance);
   BOOST_CHECK_CLOSE(ellint_rj(T(2), T(3), T(4), T(-5)), T(-0.12711230042964), tolerance);

   BOOST_CHECK_THROW(ellint_rj(T(-1), T(1), T(2), T(3)), std::domain_error);
   BOOST_CHECK_THROW(ellint_rj(T(1), T(-1), T(2), T(3)), std::domain_error);
   BOOST_CHECK_THROW(ellint_rj(T(1), T(2), T(-2), T(3)), std::domain_error);
   BOOST_CHECK_THROW(ellint_rj(T(1), T(2), T(2), T(0)), std::domain_error);
   BOOST_CHECK_THROW(ellint_rj(T(0), T(0), T(2), T(3)), std::domain_error);
   BOOST_CHECK_THROW(ellint_rj(T(2), T(0), T(0), T(0)), std::domain_error);
   BOOST_CHECK_THROW(ellint_rj(T(0), T(2), T(0), T(0)), std::domain_error);

   // RD:
   BOOST_CHECK_CLOSE(ellint_rd(T(0), T(2), T(1)), T(1.7972103521034), tolerance);
   BOOST_CHECK_CLOSE(ellint_rd(T(2), T(3), T(4)), T(0.16510527294261), tolerance);

   // RG
   BOOST_CHECK_THROW(ellint_rg(T(-1), T(1), T(2)), std::domain_error);
   BOOST_CHECK_THROW(ellint_rg(T(-1), T(-1), T(2)), std::domain_error);
   BOOST_CHECK_THROW(ellint_rg(T(-1), T(2), T(-1)), std::domain_error);
   BOOST_CHECK_EQUAL(ellint_rg(T(0), T(2), T(2)), ellint_rg(T(2), T(2), T(0)));
   BOOST_CHECK_EQUAL(ellint_rg(T(0), T(2), T(2)), ellint_rg(T(2), T(0), T(2)));

   // Sanity/consistency checks from Numerical Computation of Real or Complex 
   // Elliptic Integrals, B. C. Carlson: http://arxiv.org/abs/math.CA/9409227
   boost::mt19937 ran;
   boost::uniform_real<float> ur(0, 1000);
   T eps40 = 40 * tools::epsilon<T>();

   for(unsigned i = 0; i < 1000; ++i)
   {
      T x = ur(ran);
      T y = ur(ran);
      T z = ur(ran);
      T lambda = ur(ran);
      T mu = x * y / lambda;
      // RF, eq 49:
      T s1 = ellint_rf(x+lambda, y+lambda, lambda) + 
         ellint_rf(x + mu, y + mu, mu);
      T s2 = ellint_rf(x, y, T(0));
      BOOST_CHECK_CLOSE_FRACTION(s1, s2, eps40);
      // RC is degenerate case of RF:
      s1 = ellint_rc(x, y);
      s2 = ellint_rf(x, y, y);
      BOOST_CHECK_CLOSE_FRACTION(s1, s2, eps40);
      // RC, eq 50 (Note have to assume y = x):
      T mu2 = x * x / lambda;
      s1 = ellint_rc(lambda, x+lambda) 
         + ellint_rc(mu2, x + mu2);
      s2 = ellint_rc(T(0), x);
      BOOST_CHECK_CLOSE_FRACTION(s1, s2, eps40);
      /*
      T p = ????; // no closed form for a, b and p???
      s1 = ellint_rj(x+lambda, y+lambda, lambda, p+lambda)
         + ellint_rj(x+mu, y+mu, mu, p+mu);
      s2 = ellint_rj(x, y, T(0), p)
         - 3 * ellint_rc(a, b);
      */
      // RD, eq 53:
      s1 = ellint_rd(lambda, x+lambda, y+lambda)
         + ellint_rd(mu, x+mu, y+mu);
      s2 = ellint_rd(T(0), x, y)
         - 3 / (y * sqrt(x+y+lambda+mu));
      BOOST_CHECK_CLOSE_FRACTION(s1, s2, eps40);
      // RD is degenerate case of RJ:
      s1 = ellint_rd(x, y, z);
      s2 = ellint_rj(x, y, z, z);
      BOOST_CHECK_CLOSE_FRACTION(s1, s2, eps40);
   }
#endif
   //
   // Now random spot values:
   //
#ifdef TEST1
   t1(val, type_name);
   t2(val, type_name);
   t3(val, type_name);
   t4(val, type_name);
   t5(val, type_name);
#endif
#ifdef TEST2
   t6(val, type_name);
   t7(val, type_name);
   t8(val, type_name);
   t9(val, type_name);
   t10(val, type_name);
   t11(val, type_name);
#endif
#ifdef TEST3
   t12(val, type_name);
   t13(val, type_name);
   t14(val, type_name);
   t15(val, type_name);
   t16(val, type_name);
   t17(val, type_name);
#endif
#ifdef TEST4
   t18(val, type_name);
   t19(val, type_name);
   t20(val, type_name);
   t21(val, type_name);
   t22(val, type_name);
#endif
}


//  Copyright John Maddock 2015.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef _MSC_VER
#  pragma warning (disable : 4224)
#endif

#include <boost/array.hpp>
#include <boost/lexical_cast.hpp>
#include "../../test/table_type.hpp"
#include "table_helper.hpp"
#include "performance.hpp"
#include <iostream>

#define evaluate_polynomial_c_imp evaluate_polynomial_c_imp_1
#undef BOOST_MATH_TOOLS_POLY_EVAL_20_HPP
#include <boost/math/tools/detail/polynomial_horner1_20.hpp>
#undef evaluate_polynomial_c_imp
#undef BOOST_MATH_TOOLS_POLY_EVAL_20_HPP
#define evaluate_polynomial_c_imp evaluate_polynomial_c_imp_2
#include <boost/math/tools/detail/polynomial_horner2_20.hpp>
#undef evaluate_polynomial_c_imp
#undef BOOST_MATH_TOOLS_POLY_EVAL_20_HPP
#define evaluate_polynomial_c_imp evaluate_polynomial_c_imp_3
#include <boost/math/tools/detail/polynomial_horner3_20.hpp>
#undef evaluate_polynomial_c_imp

#undef BOOST_MATH_TOOLS_POLY_RAT_20_HPP
#define evaluate_rational_c_imp evaluate_rational_c_imp_1
#include <boost/math/tools/detail/rational_horner1_20.hpp>
#undef evaluate_rational_c_imp
#undef BOOST_MATH_TOOLS_POLY_RAT_20_HPP
#define evaluate_rational_c_imp evaluate_rational_c_imp_2
#include <boost/math/tools/detail/rational_horner2_20.hpp>
#undef evaluate_rational_c_imp
#undef BOOST_MATH_TOOLS_RAT_EVAL_20_HPP
#define evaluate_rational_c_imp evaluate_rational_c_imp_3
#include <boost/math/tools/detail/rational_horner3_20.hpp>
#undef evaluate_rational_c_imp
#undef BOOST_MATH_TOOLS_POLY_RAT_20_HPP

static const double num[21] = {
   static_cast<double>(56906521.91347156388090791033559122686859L),
   static_cast<double>(103794043.1163445451906271053616070238554L),
   static_cast<double>(86363131.28813859145546927288977868422342L),
   static_cast<double>(43338889.32467613834773723740590533316085L),
   static_cast<double>(14605578.08768506808414169982791359218571L),
   static_cast<double>(3481712.15498064590882071018964774556468L),
   static_cast<double>(601859.6171681098786670226533699352302507L),
   static_cast<double>(75999.29304014542649875303443598909137092L),
   static_cast<double>(6955.999602515376140356310115515198987526L),
   static_cast<double>(449.9445569063168119446858607650988409623L),
   static_cast<double>(19.51992788247617482847860966235652136208L),
   static_cast<double>(0.5098416655656676188125178644804694509993L),
   static_cast<double>(0.006061842346248906525783753964555936883222L),
   0.0
};
static const double denom[20] = {
   static_cast<double>(0u),
   static_cast<double>(39916800u),
   static_cast<double>(120543840u),
   static_cast<double>(150917976u),
   static_cast<double>(105258076u),
   static_cast<double>(45995730u),
   static_cast<double>(13339535u),
   static_cast<double>(2637558u),
   static_cast<double>(357423u),
   static_cast<double>(32670u),
   static_cast<double>(1925u),
   static_cast<double>(66u),
   static_cast<double>(1u),
   0.0
};
static const std::uint32_t denom_int[20] = {
   static_cast<std::uint32_t>(0u),
   static_cast<std::uint32_t>(39916800u),
   static_cast<std::uint32_t>(120543840u),
   static_cast<std::uint32_t>(150917976u),
   static_cast<std::uint32_t>(105258076u),
   static_cast<std::uint32_t>(45995730u),
   static_cast<std::uint32_t>(13339535u),
   static_cast<std::uint32_t>(2637558u),
   static_cast<std::uint32_t>(357423u),
   static_cast<std::uint32_t>(32670u),
   static_cast<std::uint32_t>(1925u),
   static_cast<std::uint32_t>(66u),
   static_cast<std::uint32_t>(1u),
   0
};

std::string make_order_string(int n)
{
   std::string result = boost::lexical_cast<std::string>(n);
   if (result.size() < 2)
      result.insert(result.begin(), ' ');
   return result;
}

void test_poly_1(const std::integral_constant<int, 1>&)
{
}

template <int N>
void test_poly_1(const std::integral_constant<int, N>&)
{
   test_poly_1(std::integral_constant<int, N - 1>());

   double time = exec_timed_test([](const std::vector<double>& v) 
   {
      double result = 0;
      for (unsigned i = 0; i < 10; ++i)
         result += boost::math::tools::detail::evaluate_polynomial_c_imp_1(denom, v[0] + i, static_cast<std::integral_constant<int, N>*>(0));
      return result;
   });
   report_execution_time(time, std::string("Polynomial Method Comparison with ") + compiler_name() + std::string(" on ") + platform_name(), "Order " + make_order_string(N), "Method 1[br](Double Coefficients)");

   time = exec_timed_test([](const std::vector<double>& v)
   {
      double result = 0;
      for (unsigned i = 0; i < 10; ++i)
         result += boost::math::tools::detail::evaluate_polynomial_c_imp_1(denom_int, v[0] + i, static_cast<std::integral_constant<int, N>*>(0));
      return result;
   });
   report_execution_time(time, std::string("Polynomial Method Comparison with ") + compiler_name() + std::string(" on ") + platform_name(), "Order " + make_order_string(N), "Method 1[br](Integer Coefficients)");
}


void test_poly_2(const std::integral_constant<int, 1>&)
{
}

template <int N>
void test_poly_2(const std::integral_constant<int, N>&)
{
   test_poly_2(std::integral_constant<int, N - 1>());

   double time = exec_timed_test([](const std::vector<double>& v) 
   {  
      double result = 0; 
      for (unsigned i = 0; i < 10; ++i)
         result += boost::math::tools::detail::evaluate_polynomial_c_imp_2(denom, v[0] + i, static_cast<std::integral_constant<int, N>*>(0));
      return result;
   });
   report_execution_time(time, std::string("Polynomial Method Comparison with ") + compiler_name() + std::string(" on ") + platform_name(), "Order " + make_order_string(N), "Method 2[br](Double Coefficients)");

   time = exec_timed_test([](const std::vector<double>& v) 
   {  
      double result = 0; 
      for (unsigned i = 0; i < 10; ++i)
         result += boost::math::tools::detail::evaluate_polynomial_c_imp_2(denom_int, v[0] + i, static_cast<std::integral_constant<int, N>*>(0));
      return result;
   });
   report_execution_time(time, std::string("Polynomial Method Comparison with ") + compiler_name() + std::string(" on ") + platform_name(), "Order " + make_order_string(N), "Method 2[br](Integer Coefficients)");
}

void test_poly_3(const std::integral_constant<int, 1>&)
{
}

template <int N>
void test_poly_3(const std::integral_constant<int, N>&)
{
   test_poly_3(std::integral_constant<int, N - 1>());

   double time = exec_timed_test([](const std::vector<double>& v) {  double result = 0;
   for (unsigned i = 0; i < 10; ++i)
      result += boost::math::tools::detail::evaluate_polynomial_c_imp_3(denom, v[0] + i, static_cast<std::integral_constant<int, N>*>(0));
   return result;
   });
   report_execution_time(time, std::string("Polynomial Method Comparison with ") + compiler_name() + std::string(" on ") + platform_name(), "Order " + make_order_string(N), "Method 3[br](Double Coefficients)");

   time = exec_timed_test([](const std::vector<double>& v) {  double result = 0;
   for (unsigned i = 0; i < 10; ++i)
      result += boost::math::tools::detail::evaluate_polynomial_c_imp_3(denom_int, v[0] + i, static_cast<std::integral_constant<int, N>*>(0));
   return result;
   });
   report_execution_time(time, std::string("Polynomial Method Comparison with ") + compiler_name() + std::string(" on ") + platform_name(), "Order " + make_order_string(N), "Method 3[br](Integer Coefficients)");
}

template <class T, class U>
U evaluate_polynomial_0(const T* poly, U const& z, std::size_t count)
{
   U sum = static_cast<U>(poly[count - 1]);
   for (int i = static_cast<int>(count) - 2; i >= 0; --i)
   {
      sum *= z;
      sum += static_cast<U>(poly[i]);
   }
   return sum;
}

void test_rat_1(const std::integral_constant<int, 1>&)
{
}

template <int N>
void test_rat_1(const std::integral_constant<int, N>&)
{
   test_rat_1(std::integral_constant<int, N - 1>());

   double time = exec_timed_test([](const std::vector<double>& v)
   {
      double result = 0;
      for (unsigned i = 0; i < 10; ++i)
         result += boost::math::tools::detail::evaluate_rational_c_imp_1(num, denom, v[0] + i, static_cast<std::integral_constant<int, N>*>(0));
      return result;
   });
   report_execution_time(time, std::string("Rational Method Comparison with ") + compiler_name() + std::string(" on ") + platform_name(), "Order " + make_order_string(N), "Method 1[br](Double Coefficients)");

   time = exec_timed_test([](const std::vector<double>& v)
   {
      double result = 0;
      for (unsigned i = 0; i < 10; ++i)
         result += boost::math::tools::detail::evaluate_rational_c_imp_1(num, denom_int, v[0] + i, static_cast<std::integral_constant<int, N>*>(0));
      return result;
   });
   report_execution_time(time, std::string("Rational Method Comparison with ") + compiler_name() + std::string(" on ") + platform_name(), "Order " + make_order_string(N), "Method 1[br](Integer Coefficients)");
}

void test_rat_2(const std::integral_constant<int, 1>&)
{
}

template <int N>
void test_rat_2(const std::integral_constant<int, N>&)
{
   test_rat_2(std::integral_constant<int, N - 1>());

   double time = exec_timed_test([](const std::vector<double>& v)
   {
      double result = 0;
      for (unsigned i = 0; i < 10; ++i)
         result += boost::math::tools::detail::evaluate_rational_c_imp_2(num, denom, v[0] + i, static_cast<std::integral_constant<int, N>*>(0));
      return result;
   });
   report_execution_time(time, std::string("Rational Method Comparison with ") + compiler_name() + std::string(" on ") + platform_name(), "Order " + make_order_string(N), "Method 2[br](Double Coefficients)");

   time = exec_timed_test([](const std::vector<double>& v)
   {
      double result = 0;
      for (unsigned i = 0; i < 10; ++i)
         result += boost::math::tools::detail::evaluate_rational_c_imp_2(num, denom_int, v[0] + i, static_cast<std::integral_constant<int, N>*>(0));
      return result;
   });
   report_execution_time(time, std::string("Rational Method Comparison with ") + compiler_name() + std::string(" on ") + platform_name(), "Order " + make_order_string(N), "Method 2[br](Integer Coefficients)");
}

void test_rat_3(const std::integral_constant<int, 1>&)
{
}

template <int N>
void test_rat_3(const std::integral_constant<int, N>&)
{
   test_rat_3(std::integral_constant<int, N - 1>());

   double time = exec_timed_test([](const std::vector<double>& v)
   {
      double result = 0;
      for (unsigned i = 0; i < 10; ++i)
         result += boost::math::tools::detail::evaluate_rational_c_imp_3(num, denom, v[0] + i, static_cast<std::integral_constant<int, N>*>(0));
      return result;
   });
   report_execution_time(time, std::string("Rational Method Comparison with ") + compiler_name() + std::string(" on ") + platform_name(), "Order " + make_order_string(N), "Method 3[br](Double Coefficients)");

   time = exec_timed_test([](const std::vector<double>& v)
   {
      double result = 0;
      for (unsigned i = 0; i < 10; ++i)
         result += boost::math::tools::detail::evaluate_rational_c_imp_3(num, denom_int, v[0] + i, static_cast<std::integral_constant<int, N>*>(0));
      return result;
   });
   report_execution_time(time, std::string("Rational Method Comparison with ") + compiler_name() + std::string(" on ") + platform_name(), "Order " + make_order_string(N), "Method 3[br](Integer Coefficients)");
}

template <class T, class U, class V>
V evaluate_rational_0(const T* num, const U* denom, const V& z_, std::size_t count)
{
   V z(z_);
   V s1, s2;
   if (z <= 1)
   {
      s1 = static_cast<V>(num[count - 1]);
      s2 = static_cast<V>(denom[count - 1]);
      for (int i = (int)count - 2; i >= 0; --i)
      {
         s1 *= z;
         s2 *= z;
         s1 += num[i];
         s2 += denom[i];
      }
   }
   else
   {
      z = 1 / z;
      s1 = static_cast<V>(num[0]);
      s2 = static_cast<V>(denom[0]);
      for (unsigned i = 1; i < count; ++i)
      {
         s1 *= z;
         s2 *= z;
         s1 += num[i];
         s2 += denom[i];
      }
   }
   return s1 / s2;
}


int main()
{
   double val = 0.001;
   while (val < 1)
   {
      std::vector<double> v;
      v.push_back(val);
      data.push_back(v);
      val *= 1.1;
   }

   for (unsigned i = 3; i <= 20; ++i)
   {
      double time = exec_timed_test([&](const std::vector<double>& v) {  
         double result = 0;
         for (unsigned j = 0; j < 10; ++j)
            result += evaluate_polynomial_0(denom, v[0] + j, i);
         return result;
      });
      report_execution_time(time, std::string("Polynomial Method Comparison with ") + compiler_name() + std::string(" on ") + platform_name(), "Order " + make_order_string(i), "Method 0[br](Double Coefficients)");

      time = exec_timed_test([&](const std::vector<double>& v) {
         double result = 0;
         for (unsigned j = 0; j < 10; ++j)
            result += evaluate_polynomial_0(denom_int, v[0] + j, i);
         return result;
      });
      report_execution_time(time, std::string("Polynomial Method Comparison with ") + compiler_name() + std::string(" on ") + platform_name(), "Order " + make_order_string(i), "Method 0[br](Integer Coefficients)");
   }

   test_poly_1(std::integral_constant<int, 20>());
   test_poly_2(std::integral_constant<int, 20>());
   test_poly_3(std::integral_constant<int, 20>());

   for (unsigned i = 3; i <= 20; ++i)
   {
      double time = exec_timed_test([&](const std::vector<double>& v) {
         double result = 0;
         for (unsigned j = 0; j < 10; ++j)
            result += evaluate_rational_0(num, denom, v[0] + j, i);
         return result;
      });
      report_execution_time(time, std::string("Rational Method Comparison with ") + compiler_name() + std::string(" on ") + platform_name(), "Order " + make_order_string(i), "Method 0[br](Double Coefficients)");

      time = exec_timed_test([&](const std::vector<double>& v) {
         double result = 0;
         for (unsigned j = 0; j < 10; ++j)
            result += evaluate_rational_0(num, denom_int, v[0] + j, i);
         return result;
      });
      report_execution_time(time, std::string("Rational Method Comparison with ") + compiler_name() + std::string(" on ") + platform_name(), "Order " + make_order_string(i), "Method 0[br](Integer Coefficients)");
   }

   test_rat_1(std::integral_constant<int, 20>());
   test_rat_2(std::integral_constant<int, 20>());
   test_rat_3(std::integral_constant<int, 20>());

   return 0;
}


// Copyright John Maddock 2023.
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/tools/promotion.hpp>
#ifndef BOOST_MATH_STANDALONE
#include <boost/multiprecision/cpp_bin_float.hpp>
#endif
#include <type_traits>

#if __has_include(<stdfloat>)
#  include <stdfloat>
#endif

int main() 
{
   using boost::math::tools::promote_args_t;

#if defined(__cpp_lib_type_trait_variable_templates) && (__cpp_lib_type_trait_variable_templates >= 201510L) && defined(__cpp_static_assert) && (__cpp_static_assert >= 201411L)

   static_assert(std::is_same_v<promote_args_t<long double, float>, long double>);
   static_assert(std::is_same_v<promote_args_t<long double, double>, long double>);
   static_assert(std::is_same_v<promote_args_t<long double, long double>, long double>);
#ifdef __STDCPP_FLOAT16_T__
   static_assert(std::is_same_v<promote_args_t<long double, std::float16_t>, long double>);
#endif
#ifdef __STDCPP_FLOAT32_T__
   static_assert(std::is_same_v<promote_args_t<long double, std::float32_t>, long double>);
#endif
#ifdef __STDCPP_FLOAT64_T__
#if LDBL_MANT_DIG > 53
   static_assert(std::is_same_v<promote_args_t<long double, std::float64_t>, long double>);
#else
   static_assert(std::is_same_v<promote_args_t<long double, std::float64_t>, std::float64_t>);
#endif
#endif
#ifdef __STDCPP_FLOAT128_T__
#if LDBL_MANT_DIG > 113
   static_assert(std::is_same_v<promote_args_t<long double, std::float128_t>, long double>);
#else
   static_assert(std::is_same_v<promote_args_t<long double, std::float128_t>, std::float128_t>);
#endif
#endif
   
   static_assert(std::is_same_v<promote_args_t<double, float>, double>);
   static_assert(std::is_same_v<promote_args_t<double, double>, double>);
   static_assert(std::is_same_v<promote_args_t<double, long double>, long double>);
#ifdef __STDCPP_FLOAT16_T__
   static_assert(std::is_same_v<promote_args_t<double, std::float16_t>, double>);
#endif
#ifdef __STDCPP_FLOAT32_T__
   static_assert(std::is_same_v<promote_args_t<double, std::float32_t>, double>);
#endif
#ifdef __STDCPP_FLOAT64_T__
#if DBL_MANT_DIG > 53
   static_assert(std::is_same_v<promote_args_t<double, std::float64_t>, double>);
#else
   static_assert(std::is_same_v<promote_args_t<double, std::float64_t>, std::float64_t>);
#endif
#endif
#ifdef __STDCPP_FLOAT128_T__
#if DBL_MANT_DIG > 113
   static_assert(std::is_same_v<promote_args_t<double, std::float128_t>, double>);
#else
   static_assert(std::is_same_v<promote_args_t<double, std::float128_t>, std::float128_t>);
#endif
#endif
   
   static_assert(std::is_same_v<promote_args_t<float, float>, float>);
   static_assert(std::is_same_v<promote_args_t<float, double>, double>);
   static_assert(std::is_same_v<promote_args_t<float, long double>, long double>);
#ifdef __STDCPP_FLOAT16_T__
   static_assert(std::is_same_v<promote_args_t<float, std::float16_t>, float>);
#endif
#ifdef __STDCPP_FLOAT32_T__
   static_assert(std::is_same_v<promote_args_t<float, std::float32_t>, std::float32_t>);
#endif
#ifdef __STDCPP_FLOAT64_T__
   static_assert(std::is_same_v<promote_args_t<float, std::float64_t>, std::float64_t>);
#endif
#ifdef __STDCPP_FLOAT128_T__
   static_assert(std::is_same_v<promote_args_t<float, std::float128_t>, std::float128_t>);
#endif
   
#ifdef __STDCPP_FLOAT16_T__
   static_assert(std::is_same_v<promote_args_t<std::float16_t, float>, float>);
   static_assert(std::is_same_v<promote_args_t<std::float16_t, double>, double>);
   static_assert(std::is_same_v<promote_args_t<std::float16_t, long double>, long double>);
#ifdef __STDCPP_FLOAT16_T__
   static_assert(std::is_same_v<promote_args_t<std::float16_t, std::float16_t>, float>);
#endif
#ifdef __STDCPP_FLOAT32_T__
   static_assert(std::is_same_v<promote_args_t<std::float16_t, std::float32_t>, std::float32_t>);
#endif
#ifdef __STDCPP_FLOAT64_T__
   static_assert(std::is_same_v<promote_args_t<std::float16_t, std::float64_t>, std::float64_t>);
#endif
#ifdef __STDCPP_FLOAT128_T__
   static_assert(std::is_same_v<promote_args_t<std::float16_t, std::float128_t>, std::float128_t>);
#endif
#endif   

#ifdef __STDCPP_FLOAT32_T__
   static_assert(std::is_same_v<promote_args_t<std::float32_t, float>, std::float32_t>);
   static_assert(std::is_same_v<promote_args_t<std::float32_t, double>, double>);
   static_assert(std::is_same_v<promote_args_t<std::float32_t, long double>, long double>);
#ifdef __STDCPP_FLOAT16_T__
   static_assert(std::is_same_v<promote_args_t<std::float32_t, std::float16_t>, std::float32_t>);
#endif
#ifdef __STDCPP_FLOAT32_T__
   static_assert(std::is_same_v<promote_args_t<std::float32_t, std::float32_t>, std::float32_t>);
#endif
#ifdef __STDCPP_FLOAT64_T__
   static_assert(std::is_same_v<promote_args_t<std::float32_t, std::float64_t>, std::float64_t>);
#endif
#ifdef __STDCPP_FLOAT128_T__
   static_assert(std::is_same_v<promote_args_t<std::float32_t, std::float128_t>, std::float128_t>);
#endif
#endif   

#ifdef __STDCPP_FLOAT64_T__
   static_assert(std::is_same_v<promote_args_t<std::float64_t, float>, std::float64_t>);
   static_assert(std::is_same_v<promote_args_t<std::float64_t, double>, std::float64_t>);
#if LDBL_MANT_DIG > 53
   static_assert(std::is_same_v<promote_args_t<std::float64_t, long double>, long double>);
#else
   static_assert(std::is_same_v<promote_args_t<std::float64_t, long double>, std::float64_t>);
#endif
#ifdef __STDCPP_FLOAT16_T__
   static_assert(std::is_same_v<promote_args_t<std::float64_t, std::float16_t>, std::float64_t>);
#endif
#ifdef __STDCPP_FLOAT32_T__
   static_assert(std::is_same_v<promote_args_t<std::float64_t, std::float32_t>, std::float64_t>);
#endif
#ifdef __STDCPP_FLOAT64_T__
   static_assert(std::is_same_v<promote_args_t<std::float64_t, std::float64_t>, std::float64_t>);
#endif
#ifdef __STDCPP_FLOAT128_T__
   static_assert(std::is_same_v<promote_args_t<std::float64_t, std::float128_t>, std::float128_t>);
#endif
#endif   

#ifdef __STDCPP_FLOAT128_T__
   static_assert(std::is_same_v<promote_args_t<std::float128_t, float>, std::float128_t>);
   static_assert(std::is_same_v<promote_args_t<std::float128_t, double>, std::float128_t>);
#if LDBL_MANT_DIG > 113
   static_assert(std::is_same_v<promote_args_t<std::float128_t, long double>, long double>);
#else
   static_assert(std::is_same_v<promote_args_t<std::float128_t, long double>, std::float128_t>);
#endif
#ifdef __STDCPP_FLOAT16_T__
   static_assert(std::is_same_v<promote_args_t<std::float128_t, std::float16_t>, std::float128_t>);
#endif
#ifdef __STDCPP_FLOAT32_T__
   static_assert(std::is_same_v<promote_args_t<std::float128_t, std::float32_t>, std::float128_t>);
#endif
#ifdef __STDCPP_FLOAT64_T__
   static_assert(std::is_same_v<promote_args_t<std::float128_t, std::float64_t>, std::float128_t>);
#endif
#ifdef __STDCPP_FLOAT128_T__
   static_assert(std::is_same_v<promote_args_t<std::float128_t, std::float128_t>, std::float128_t>);
#endif
#endif   

#ifndef BOOST_MATH_STANDALONE
   static_assert(std::is_same_v<promote_args_t<float, boost::multiprecision::cpp_bin_float_50>, boost::multiprecision::cpp_bin_float_50>);
   static_assert(std::is_same_v<promote_args_t<double, boost::multiprecision::cpp_bin_float_50>, boost::multiprecision::cpp_bin_float_50>);
   static_assert(std::is_same_v<promote_args_t<long double, boost::multiprecision::cpp_bin_float_50>, boost::multiprecision::cpp_bin_float_50>);
#ifdef __STDCPP_FLOAT16_T__
   static_assert(std::is_same_v<promote_args_t<std::float16_t, boost::multiprecision::cpp_bin_float_50>, boost::multiprecision::cpp_bin_float_50>);
#endif
#ifdef __STDCPP_FLOAT32_T__
   static_assert(std::is_same_v<promote_args_t<std::float32_t, boost::multiprecision::cpp_bin_float_50>, boost::multiprecision::cpp_bin_float_50>);
#endif
#ifdef __STDCPP_FLOAT64_T__
   static_assert(std::is_same_v<promote_args_t<std::float64_t, boost::multiprecision::cpp_bin_float_50>, boost::multiprecision::cpp_bin_float_50>);
#endif
#ifdef __STDCPP_FLOAT128_T__
   static_assert(std::is_same_v<promote_args_t<std::float128_t, boost::multiprecision::cpp_bin_float_50>, boost::multiprecision::cpp_bin_float_50>);
#endif
#endif // BOOST_MATH_STANDALONE

#endif

   return 0;
}

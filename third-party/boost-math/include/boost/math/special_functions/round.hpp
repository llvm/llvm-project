//  Copyright John Maddock 2007.
//  Copyright Matt Borland 2023.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_ROUND_HPP
#define BOOST_MATH_ROUND_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/tools/config.hpp>

#ifndef BOOST_MATH_HAS_NVRTC

#include <boost/math/ccmath/detail/config.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <type_traits>
#include <limits>
#include <cmath>

#if !defined(BOOST_MATH_NO_CCMATH) && !defined(BOOST_MATH_NO_CONSTEXPR_DETECTION)
#include <boost/math/ccmath/ldexp.hpp>
#    define BOOST_MATH_HAS_CONSTEXPR_LDEXP
#endif

namespace boost{ namespace math{

namespace detail{

template <class T, class Policy>
BOOST_MATH_GPU_ENABLED inline tools::promote_args_t<T> round(const T& v, const Policy& pol, const std::false_type&)
{
   BOOST_MATH_STD_USING
   using result_type = tools::promote_args_t<T>;

   if(!(boost::math::isfinite)(v))
   {
      return policies::raise_rounding_error("boost::math::round<%1%>(%1%)", nullptr, static_cast<result_type>(v), static_cast<result_type>(v), pol);
   }
   //
   // The logic here is rather convoluted, but avoids a number of traps,
   // see discussion here https://github.com/boostorg/math/pull/8
   //
   if (T(-0.5) < v && v < T(0.5))
   {
      // special case to avoid rounding error on the direct
      // predecessor of +0.5 resp. the direct successor of -0.5 in
      // IEEE floating point types
      return static_cast<result_type>(0);
   }
   else if (v > 0)
   {
      // subtract v from ceil(v) first in order to avoid rounding
      // errors on largest representable integer numbers
      result_type c(ceil(v));
      return T(0.5) < c - v ? c - 1 : c;
   }
   else
   {
      // see former branch
      result_type f(floor(v));
      return T(0.5) < v - f ? f + 1 : f;
   }
}
template <class T, class Policy>
BOOST_MATH_GPU_ENABLED inline tools::promote_args_t<T> round(const T& v, const Policy&, const std::true_type&)
{
   return v;
}

} // namespace detail

template <class T, class Policy>
BOOST_MATH_GPU_ENABLED inline tools::promote_args_t<T> round(const T& v, const Policy& pol)
{
   return detail::round(v, pol, std::integral_constant<bool, detail::is_integer_for_rounding<T>::value>());
}
template <class T>
BOOST_MATH_GPU_ENABLED inline tools::promote_args_t<T> round(const T& v)
{
   return round(v, policies::policy<>());
}
//
// The following functions will not compile unless T has an
// implicit conversion to the integer types.  For user-defined
// number types this will likely not be the case.  In that case
// these functions should either be specialized for the UDT in
// question, or else overloads should be placed in the same
// namespace as the UDT: these will then be found via argument
// dependent lookup.  See our concept archetypes for examples.
//
// Non-standard numeric limits syntax "(std::numeric_limits<int>::max)()"
// is to avoid macro substiution from MSVC
// https://stackoverflow.com/questions/27442885/syntax-error-with-stdnumeric-limitsmax
//
template <class T, class Policy>
inline int iround(const T& v, const Policy& pol)
{
   BOOST_MATH_STD_USING
   using result_type = tools::promote_args_t<T>;

   result_type r = boost::math::round(v, pol);

   #if defined(BOOST_MATH_HAS_CONSTEXPR_LDEXP) && !defined(BOOST_MATH_HAS_GPU_SUPPORT)
   if constexpr (std::is_arithmetic_v<result_type>
                 #ifdef BOOST_MATH_FLOAT128_TYPE
                 && !std::is_same_v<BOOST_MATH_FLOAT128_TYPE, result_type>
                 #endif
                )
   {
      constexpr result_type max_val = boost::math::ccmath::ldexp(static_cast<result_type>(1), std::numeric_limits<int>::digits);
      
      if (r >= max_val || r < -max_val)
      {
         return static_cast<int>(boost::math::policies::raise_rounding_error("boost::math::iround<%1%>(%1%)", nullptr, v, static_cast<int>(0), pol));
      }
   }
   else
   {
      static const result_type max_val = ldexp(static_cast<result_type>(1), std::numeric_limits<int>::digits);
   
      if (r >= max_val || r < -max_val)
      {
         return static_cast<int>(boost::math::policies::raise_rounding_error("boost::math::iround<%1%>(%1%)", nullptr, v, static_cast<int>(0), pol));
      }
   }
   #else
   BOOST_MATH_STATIC_LOCAL_VARIABLE const result_type max_val = ldexp(static_cast<result_type>(1), std::numeric_limits<int>::digits);

   if (r >= max_val || r < -max_val)
   {
      return static_cast<int>(boost::math::policies::raise_rounding_error("boost::math::iround<%1%>(%1%)", nullptr, v, static_cast<int>(0), pol));
   }
   #endif

   return static_cast<int>(r);
}
template <class T>
BOOST_MATH_GPU_ENABLED inline int iround(const T& v)
{
   return iround(v, policies::policy<>());
}

template <class T, class Policy>
BOOST_MATH_GPU_ENABLED inline long lround(const T& v, const Policy& pol)
{
   BOOST_MATH_STD_USING
   using result_type = tools::promote_args_t<T>;

   result_type r = boost::math::round(v, pol);
   
   #if defined(BOOST_MATH_HAS_CONSTEXPR_LDEXP) && !defined(BOOST_MATH_HAS_GPU_SUPPORT)
   if constexpr (std::is_arithmetic_v<result_type>
                 #ifdef BOOST_MATH_FLOAT128_TYPE
                 && !std::is_same_v<BOOST_MATH_FLOAT128_TYPE, result_type>
                 #endif
                )
   {
      constexpr result_type max_val = boost::math::ccmath::ldexp(static_cast<result_type>(1), std::numeric_limits<long>::digits);
      
      if (r >= max_val || r < -max_val)
      {
         return static_cast<long>(boost::math::policies::raise_rounding_error("boost::math::lround<%1%>(%1%)", nullptr, v, static_cast<long>(0), pol));
      }
   }
   else
   {
      static const result_type max_val = ldexp(static_cast<result_type>(1), std::numeric_limits<long>::digits);
   
      if (r >= max_val || r < -max_val)
      {
         return static_cast<long>(boost::math::policies::raise_rounding_error("boost::math::lround<%1%>(%1%)", nullptr, v, static_cast<long>(0), pol));
      }
   }
   #else
   BOOST_MATH_STATIC_LOCAL_VARIABLE const result_type max_val = ldexp(static_cast<result_type>(1), std::numeric_limits<long>::digits);

   if (r >= max_val || r < -max_val)
   {
      return static_cast<long>(boost::math::policies::raise_rounding_error("boost::math::lround<%1%>(%1%)", nullptr, v, static_cast<long>(0), pol));
   }
   #endif

   return static_cast<long>(r);
}
template <class T>
BOOST_MATH_GPU_ENABLED inline long lround(const T& v)
{
   return lround(v, policies::policy<>());
}

template <class T, class Policy>
BOOST_MATH_GPU_ENABLED inline long long llround(const T& v, const Policy& pol)
{
   BOOST_MATH_STD_USING
   using result_type = boost::math::tools::promote_args_t<T>;

   result_type r = boost::math::round(v, pol);

   #if defined(BOOST_MATH_HAS_CONSTEXPR_LDEXP) && !defined(BOOST_MATH_HAS_GPU_SUPPORT)
   if constexpr (std::is_arithmetic_v<result_type>
                 #ifdef BOOST_MATH_FLOAT128_TYPE
                 && !std::is_same_v<BOOST_MATH_FLOAT128_TYPE, result_type>
                 #endif
                )
   {
      constexpr result_type max_val = boost::math::ccmath::ldexp(static_cast<result_type>(1), std::numeric_limits<long long>::digits);
      
      if (r >= max_val || r < -max_val)
      {
         return static_cast<long long>(boost::math::policies::raise_rounding_error("boost::math::llround<%1%>(%1%)", nullptr, v, static_cast<long long>(0), pol));
      }
   }
   else
   {
      static const result_type max_val = ldexp(static_cast<result_type>(1), std::numeric_limits<long long>::digits);
   
      if (r >= max_val || r < -max_val)
      {
         return static_cast<long long>(boost::math::policies::raise_rounding_error("boost::math::llround<%1%>(%1%)", nullptr, v, static_cast<long long>(0), pol));
      }
   }
   #else
   BOOST_MATH_STATIC_LOCAL_VARIABLE const result_type max_val = ldexp(static_cast<result_type>(1), std::numeric_limits<long long>::digits);

   if (r >= max_val || r < -max_val)
   {
      return static_cast<long long>(boost::math::policies::raise_rounding_error("boost::math::llround<%1%>(%1%)", nullptr, v, static_cast<long long>(0), pol));
   }
   #endif

   return static_cast<long long>(r);
}
template <class T>
BOOST_MATH_GPU_ENABLED inline long long llround(const T& v)
{
   return llround(v, policies::policy<>());
}

}} // namespaces

#else // Specialized NVRTC overloads

namespace boost {
namespace math {

template <typename T>
BOOST_MATH_GPU_ENABLED T round(T x)
{
   return ::round(x);
}

template <>
BOOST_MATH_GPU_ENABLED float round(float x)
{
   return ::roundf(x);
}

template <typename T, typename Policy>
BOOST_MATH_GPU_ENABLED T round(T x, const Policy&)
{
   return ::round(x);
}

template <typename Policy>
BOOST_MATH_GPU_ENABLED float round(float x, const Policy&)
{
   return ::roundf(x);
}

template <typename T>
BOOST_MATH_GPU_ENABLED int iround(T x)
{
   return static_cast<int>(::lround(x));
}

template <>
BOOST_MATH_GPU_ENABLED int iround(float x)
{
   return static_cast<int>(::lroundf(x));
}

template <typename T, typename Policy>
BOOST_MATH_GPU_ENABLED int iround(T x, const Policy&)
{
   return static_cast<int>(::lround(x));
}

template <typename Policy>
BOOST_MATH_GPU_ENABLED int iround(float x, const Policy&)
{
   return static_cast<int>(::lroundf(x));
}

template <typename T>
BOOST_MATH_GPU_ENABLED long lround(T x)
{
   return ::lround(x);
}

template <>
BOOST_MATH_GPU_ENABLED long lround(float x)
{
   return ::lroundf(x);
}

template <typename T, typename Policy>
BOOST_MATH_GPU_ENABLED long lround(T x, const Policy&)
{
   return ::lround(x);
}

template <typename Policy>
BOOST_MATH_GPU_ENABLED long lround(float x, const Policy&)
{
   return ::lroundf(x);
}

template <typename T>
BOOST_MATH_GPU_ENABLED long long llround(T x)
{
   return ::llround(x);
}

template <>
BOOST_MATH_GPU_ENABLED long long llround(float x)
{
   return ::llroundf(x);
}

template <typename T, typename Policy>
BOOST_MATH_GPU_ENABLED long long llround(T x, const Policy&)
{
   return ::llround(x);
}

template <typename Policy>
BOOST_MATH_GPU_ENABLED long long llround(float x, const Policy&)
{
   return ::llroundf(x);
}

} // Namespace math
} // Namespace boost

#endif // BOOST_MATH_HAS_NVRTC

#endif // BOOST_MATH_ROUND_HPP

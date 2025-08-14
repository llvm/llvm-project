//  (C) Copyright John Maddock 2006.
//  (C) Copyright Matt Borland 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_EXPM1_INCLUDED
#define BOOST_MATH_EXPM1_INCLUDED

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/tools/config.hpp>

#ifndef BOOST_MATH_HAS_NVRTC

#include <boost/math/tools/series.hpp>
#include <boost/math/tools/precision.hpp>
#include <boost/math/tools/big_constant.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/tools/rational.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/tools/assert.hpp>
#include <boost/math/tools/numeric_limits.hpp>
#include <boost/math/tools/type_traits.hpp>
#include <boost/math/tools/cstdint.hpp>

#if defined(__GNUC__) && defined(BOOST_MATH_USE_FLOAT128)
//
// This is the only way we can avoid
// warning: non-standard suffix on floating constant [-Wpedantic]
// when building with -Wall -pedantic.  Neither __extension__
// nor #pragma diagnostic ignored work :(
//
#pragma GCC system_header
#endif

namespace boost{ namespace math{

namespace detail
{
  // Functor expm1_series returns the next term in the Taylor series
  // x^k / k!
  // each time that operator() is invoked.
  //
  template <class T>
  struct expm1_series
  {
     typedef T result_type;

     BOOST_MATH_GPU_ENABLED expm1_series(T x)
        : k(0), m_x(x), m_term(1) {}

     BOOST_MATH_GPU_ENABLED T operator()()
     {
        ++k;
        m_term *= m_x;
        m_term /= k;
        return m_term;
     }

     BOOST_MATH_GPU_ENABLED int count()const
     {
        return k;
     }

  private:
     int k;
     const T m_x;
     T m_term;
     expm1_series(const expm1_series&) = delete;
     expm1_series& operator=(const expm1_series&) = delete;
  };

template <class T, class Policy, class tag>
struct expm1_initializer
{
   struct init
   {
      BOOST_MATH_GPU_ENABLED init()
      {
         do_init(tag());
      }
      template <int N>
      BOOST_MATH_GPU_ENABLED static void do_init(const boost::math::integral_constant<int, N>&){}
      BOOST_MATH_GPU_ENABLED static void do_init(const boost::math::integral_constant<int, 64>&)
      {
         expm1(T(0.5));
      }
      BOOST_MATH_GPU_ENABLED static void do_init(const boost::math::integral_constant<int, 113>&)
      {
         expm1(T(0.5));
      }
      BOOST_MATH_GPU_ENABLED void force_instantiate()const{}
   };
   BOOST_MATH_STATIC const init initializer;
   BOOST_MATH_GPU_ENABLED static void force_instantiate()
   {
      #ifndef BOOST_MATH_HAS_GPU_SUPPORT
      initializer.force_instantiate();
      #endif
   }
};

template <class T, class Policy, class tag>
const typename expm1_initializer<T, Policy, tag>::init expm1_initializer<T, Policy, tag>::initializer;

//
// Algorithm expm1 is part of C99, but is not yet provided by many compilers.
//
// This version uses a Taylor series expansion for 0.5 > |x| > epsilon.
//
template <class T, class Policy>
T expm1_imp(T x, const boost::math::integral_constant<int, 0>&, const Policy& pol)
{
   BOOST_MATH_STD_USING

   T a = fabs(x);
   if((boost::math::isnan)(a))
   {
      return policies::raise_domain_error<T>("boost::math::expm1<%1%>(%1%)", "expm1 requires a finite argument, but got %1%", a, pol);
   }
   if(a > T(0.5f))
   {
      if(a >= tools::log_max_value<T>())
      {
         if(x > 0)
            return policies::raise_overflow_error<T>("boost::math::expm1<%1%>(%1%)", nullptr, pol);
         return -1;
      }
      return exp(x) - T(1);
   }
   if(a < tools::epsilon<T>())
      return x;
   detail::expm1_series<T> s(x);
   boost::math::uintmax_t max_iter = policies::get_max_series_iterations<Policy>();

   T result = tools::sum_series(s, policies::get_epsilon<T, Policy>(), max_iter);

   policies::check_series_iterations<T>("boost::math::expm1<%1%>(%1%)", max_iter, pol);
   return result;
}

template <class T, class P>
BOOST_MATH_GPU_ENABLED T expm1_imp(T x, const boost::math::integral_constant<int, 53>&, const P& pol)
{
   BOOST_MATH_STD_USING

   T a = fabs(x);
   if(a > T(0.5L))
   {
      if(a >= tools::log_max_value<T>())
      {
         if(x > 0)
            return policies::raise_overflow_error<T>("boost::math::expm1<%1%>(%1%)", nullptr, pol);
         return -1;
      }
      return exp(x) - T(1);
   }
   if(a < tools::epsilon<T>())
      return x;

   BOOST_MATH_STATIC const float Y = 0.10281276702880859e1f;
   BOOST_MATH_STATIC const T n[] = { static_cast<T>(-0.28127670288085937e-1), static_cast<T>(0.51278186299064534e0), static_cast<T>(-0.6310029069350198e-1), static_cast<T>(0.11638457975729296e-1), static_cast<T>(-0.52143390687521003e-3), static_cast<T>(0.21491399776965688e-4) };
   BOOST_MATH_STATIC const T d[] = { 1, static_cast<T>(-0.45442309511354755e0), static_cast<T>(0.90850389570911714e-1), static_cast<T>(-0.10088963629815502e-1), static_cast<T>(0.63003407478692265e-3), static_cast<T>(-0.17976570003654402e-4) };

   T result = x * Y + x * tools::evaluate_polynomial(n, x) / tools::evaluate_polynomial(d, x);
   return result;
}

template <class T, class P>
BOOST_MATH_GPU_ENABLED T expm1_imp(T x, const boost::math::integral_constant<int, 64>&, const P& pol)
{
   BOOST_MATH_STD_USING

   T a = fabs(x);
   if(a > T(0.5L))
   {
      if(a >= tools::log_max_value<T>())
      {
         if(x > 0)
            return policies::raise_overflow_error<T>("boost::math::expm1<%1%>(%1%)", nullptr, pol);
         return -1;
      }
      return exp(x) - T(1);
   }
   if(a < tools::epsilon<T>())
      return x;

   BOOST_MATH_STATIC const float Y = 0.10281276702880859375e1f;
   BOOST_MATH_STATIC const T n[] = {
      BOOST_MATH_BIG_CONSTANT(T, 64, -0.281276702880859375e-1),
       BOOST_MATH_BIG_CONSTANT(T, 64, 0.512980290285154286358e0),
       BOOST_MATH_BIG_CONSTANT(T, 64, -0.667758794592881019644e-1),
       BOOST_MATH_BIG_CONSTANT(T, 64, 0.131432469658444745835e-1),
       BOOST_MATH_BIG_CONSTANT(T, 64, -0.72303795326880286965e-3),
       BOOST_MATH_BIG_CONSTANT(T, 64, 0.447441185192951335042e-4),
       BOOST_MATH_BIG_CONSTANT(T, 64, -0.714539134024984593011e-6)
   };
   BOOST_MATH_STATIC const T d[] = {
      BOOST_MATH_BIG_CONSTANT(T, 64, 1.0),
      BOOST_MATH_BIG_CONSTANT(T, 64, -0.461477618025562520389e0),
      BOOST_MATH_BIG_CONSTANT(T, 64, 0.961237488025708540713e-1),
      BOOST_MATH_BIG_CONSTANT(T, 64, -0.116483957658204450739e-1),
      BOOST_MATH_BIG_CONSTANT(T, 64, 0.873308008461557544458e-3),
      BOOST_MATH_BIG_CONSTANT(T, 64, -0.387922804997682392562e-4),
      BOOST_MATH_BIG_CONSTANT(T, 64, 0.807473180049193557294e-6)
   };

   T result = x * Y + x * tools::evaluate_polynomial(n, x) / tools::evaluate_polynomial(d, x);
   return result;
}

template <class T, class P>
BOOST_MATH_GPU_ENABLED T expm1_imp(T x, const boost::math::integral_constant<int, 113>&, const P& pol)
{
   BOOST_MATH_STD_USING

   T a = fabs(x);
   if(a > T(0.5L))
   {
      if(a >= tools::log_max_value<T>())
      {
         if(x > 0)
            return policies::raise_overflow_error<T>("boost::math::expm1<%1%>(%1%)", nullptr, pol);
         return -1;
      }
      return exp(x) - T(1);
   }
   if(a < tools::epsilon<T>())
      return x;

   static const float Y = 0.10281276702880859375e1f;
   static const T n[] = {
      BOOST_MATH_BIG_CONSTANT(T, 113, -0.28127670288085937499999999999999999854e-1),
      BOOST_MATH_BIG_CONSTANT(T, 113, 0.51278156911210477556524452177540792214e0),
      BOOST_MATH_BIG_CONSTANT(T, 113, -0.63263178520747096729500254678819588223e-1),
      BOOST_MATH_BIG_CONSTANT(T, 113, 0.14703285606874250425508446801230572252e-1),
      BOOST_MATH_BIG_CONSTANT(T, 113, -0.8675686051689527802425310407898459386e-3),
      BOOST_MATH_BIG_CONSTANT(T, 113, 0.88126359618291165384647080266133492399e-4),
      BOOST_MATH_BIG_CONSTANT(T, 113, -0.25963087867706310844432390015463138953e-5),
      BOOST_MATH_BIG_CONSTANT(T, 113, 0.14226691087800461778631773363204081194e-6),
      BOOST_MATH_BIG_CONSTANT(T, 113, -0.15995603306536496772374181066765665596e-8),
      BOOST_MATH_BIG_CONSTANT(T, 113, 0.45261820069007790520447958280473183582e-10)
   };
   static const T d[] = {
      BOOST_MATH_BIG_CONSTANT(T, 113, 1.0),
      BOOST_MATH_BIG_CONSTANT(T, 113, -0.45441264709074310514348137469214538853e0),
      BOOST_MATH_BIG_CONSTANT(T, 113, 0.96827131936192217313133611655555298106e-1),
      BOOST_MATH_BIG_CONSTANT(T, 113, -0.12745248725908178612540554584374876219e-1),
      BOOST_MATH_BIG_CONSTANT(T, 113, 0.11473613871583259821612766907781095472e-2),
      BOOST_MATH_BIG_CONSTANT(T, 113, -0.73704168477258911962046591907690764416e-4),
      BOOST_MATH_BIG_CONSTANT(T, 113, 0.34087499397791555759285503797256103259e-5),
      BOOST_MATH_BIG_CONSTANT(T, 113, -0.11114024704296196166272091230695179724e-6),
      BOOST_MATH_BIG_CONSTANT(T, 113, 0.23987051614110848595909588343223896577e-8),
      BOOST_MATH_BIG_CONSTANT(T, 113, -0.29477341859111589208776402638429026517e-10),
      BOOST_MATH_BIG_CONSTANT(T, 113, 0.13222065991022301420255904060628100924e-12)
   };

   T result = x * Y + x * tools::evaluate_polynomial(n, x) / tools::evaluate_polynomial(d, x);
   return result;
}

} // namespace detail

template <class T, class Policy>
BOOST_MATH_GPU_ENABLED inline typename tools::promote_args<T>::type expm1(T x, const Policy& /* pol */)
{
   typedef typename tools::promote_args<T>::type result_type;
   typedef typename policies::evaluation<result_type, Policy>::type value_type;
   typedef typename policies::precision<result_type, Policy>::type precision_type;
   typedef typename policies::normalise<
      Policy,
      policies::promote_float<false>,
      policies::promote_double<false>,
      policies::discrete_quantile<>,
      policies::assert_undefined<> >::type forwarding_policy;

   typedef boost::math::integral_constant<int,
      precision_type::value <= 0 ? 0 :
      precision_type::value <= 53 ? 53 :
      precision_type::value <= 64 ? 64 :
      precision_type::value <= 113 ? 113 : 0
   > tag_type;

   detail::expm1_initializer<value_type, forwarding_policy, tag_type>::force_instantiate();

   return policies::checked_narrowing_cast<result_type, forwarding_policy>(detail::expm1_imp(
      static_cast<value_type>(x),
      tag_type(), forwarding_policy()), "boost::math::expm1<%1%>(%1%)");
}

#ifdef expm1
#  ifndef BOOST_HAS_expm1
#     define BOOST_HAS_expm1
#  endif
#  undef expm1
#endif

#if defined(BOOST_HAS_EXPM1) && !(defined(__osf__) && defined(__DECCXX_VER))
#  ifdef BOOST_MATH_USE_C99
BOOST_MATH_GPU_ENABLED inline float expm1(float x, const policies::policy<>&){ return ::expm1f(x); }
#     ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
inline long double expm1(long double x, const policies::policy<>&){ return ::expm1l(x); }
#     endif
#  else
inline float expm1(float x, const policies::policy<>&){ return static_cast<float>(::expm1(x)); }
#  endif
BOOST_MATH_GPU_ENABLED inline double expm1(double x, const policies::policy<>&){ return ::expm1(x); }
#endif

template <class T>
BOOST_MATH_GPU_ENABLED inline typename tools::promote_args<T>::type expm1(T x)
{
   return expm1(x, policies::policy<>());
}

} // namespace math
} // namespace boost

#else // Special handling for NVRTC 

namespace boost {
namespace math {

template <typename T>
BOOST_MATH_GPU_ENABLED auto expm1(T x)
{
   return ::expm1(x);
}

template <>
BOOST_MATH_GPU_ENABLED auto expm1(float x)
{
   return ::expm1f(x);
}

template <typename T, typename Policy>
BOOST_MATH_GPU_ENABLED auto expm1(T x, const Policy&)
{
   return ::expm1(x);
}

template <typename Policy>
BOOST_MATH_GPU_ENABLED auto expm1(float x, const Policy&)
{
   return ::expm1f(x);
}

} // Namespace math
} // Namespace boost

#endif // BOOST_MATH_HAS_NVRTC

#endif // BOOST_MATH_HYPOT_INCLUDED





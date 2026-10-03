//  (C) Copyright John Maddock 2005-2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_HYPOT_INCLUDED
#define BOOST_MATH_HYPOT_INCLUDED

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/tools/config.hpp>
#include <boost/math/tools/precision.hpp>
#include <boost/math/tools/numeric_limits.hpp>
#include <boost/math/tools/type_traits.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/tools/utility.hpp>
#include <boost/math/tools/numeric_limits.hpp>

namespace boost{ namespace math{ namespace detail{

template <class T, class Policy>
BOOST_MATH_GPU_ENABLED T hypot_imp(T x, T y, const Policy& pol)
{
   //
   // Normalize x and y, so that both are positive and x >= y:
   //
   BOOST_MATH_STD_USING

   x = fabs(x);
   y = fabs(y);

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4127)
#endif
   // special case, see C99 Annex F:
   if(boost::math::numeric_limits<T>::has_infinity
      && ((x == boost::math::numeric_limits<T>::infinity())
      || (y == boost::math::numeric_limits<T>::infinity())))
      return policies::raise_overflow_error<T>("boost::math::hypot<%1%>(%1%,%1%)", nullptr, pol);
#ifdef _MSC_VER
#pragma warning(pop)
#endif

   if(y > x)
      BOOST_MATH_GPU_SAFE_SWAP(x, y);

   if(x * tools::epsilon<T>() >= y)
      return x;

   T rat = y / x;
   return x * sqrt(1 + rat*rat);
} // template <class T> T hypot(T x, T y)

template <class T, class Policy>
BOOST_MATH_GPU_ENABLED T hypot_imp(T x, T y, T z, const Policy& pol)
{
   BOOST_MATH_STD_USING

   x = fabs(x);
   y = fabs(y);
   z = fabs(z);

   #ifdef _MSC_VER
   #pragma warning(push)
   #pragma warning(disable: 4127)
   #endif
   // special case, see C99 Annex F:
   BOOST_MATH_IF_CONSTEXPR (boost::math::numeric_limits<T>::has_infinity)
   {
      if(((x == boost::math::numeric_limits<T>::infinity())
         || (y == boost::math::numeric_limits<T>::infinity())
         || (z == boost::math::numeric_limits<T>::infinity())))
         return policies::raise_overflow_error<T>("boost::math::hypot<%1%>(%1%,%1%,%1%)", nullptr, pol);
   }
   #ifdef _MSC_VER
   #pragma warning(pop)
   #endif

   const T a {(max)((max)(x, y), z)};

   if (a == T(0))
   {
      return a;
   }

   const T x_div_a {x / a};
   const T y_div_a {y / a};
   const T z_div_a {z / a};

   return a * sqrt(x_div_a * x_div_a
                 + y_div_a * y_div_a
                 + z_div_a * z_div_a);

}

}

template <class T1, class T2>
BOOST_MATH_GPU_ENABLED inline typename tools::promote_args<T1, T2>::type
   hypot(T1 x, T2 y)
{
   typedef typename tools::promote_args<T1, T2>::type result_type;
   return detail::hypot_imp(
      static_cast<result_type>(x), static_cast<result_type>(y), policies::policy<>());
}

template <class T1, class T2, class Policy, boost::math::enable_if_t<policies::is_policy_v<Policy>, bool>>
BOOST_MATH_GPU_ENABLED inline typename tools::promote_args<T1, T2>::type
   hypot(T1 x, T2 y, const Policy& pol)
{
   typedef typename tools::promote_args<T1, T2>::type result_type;
   return detail::hypot_imp(
      static_cast<result_type>(x), static_cast<result_type>(y), pol);
}

template <class T1, class T2, class T3, boost::math::enable_if_t<!policies::is_policy_v<T3>, bool>>
BOOST_MATH_GPU_ENABLED inline tools::promote_args_t<T1, T2, T3>
   hypot(T1 x, T2 y, T3 z)
{
   using result_type = tools::promote_args_t<T1, T2, T3>;
   return detail::hypot_imp(static_cast<result_type>(x),
                            static_cast<result_type>(y),
                            static_cast<result_type>(z),
                            policies::policy<>());
}

template <class T1, class T2, class T3, class Policy>
BOOST_MATH_GPU_ENABLED inline tools::promote_args_t<T1, T2, T3>
   hypot(T1 x, T2 y, T3 z, const Policy& pol)
{
   using result_type = tools::promote_args_t<T1, T2, T3>;
   return detail::hypot_imp(static_cast<result_type>(x),
                            static_cast<result_type>(y),
                            static_cast<result_type>(z),
                            pol);
}

} // namespace math
} // namespace boost

#endif // BOOST_MATH_HYPOT_INCLUDED




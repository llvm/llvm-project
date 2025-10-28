//  (C) Copyright John Maddock 2006.
//  (C) Copyright Matt Borland 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_POWM1
#define BOOST_MATH_POWM1

#ifdef _MSC_VER
#pragma once
#pragma warning(push)
#pragma warning(disable:4702) // Unreachable code (release mode only warning)
#endif

#include <boost/math/tools/config.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/special_functions/log1p.hpp>
#include <boost/math/special_functions/expm1.hpp>
#include <boost/math/special_functions/trunc.hpp>
#include <boost/math/special_functions/sign.hpp>
#include <boost/math/tools/assert.hpp>

namespace boost{ namespace math{ namespace detail{

template <class T, class Policy>
BOOST_MATH_GPU_ENABLED inline T powm1_imp(const T x, const T y, const Policy& pol)
{
   BOOST_MATH_STD_USING
   constexpr auto function = "boost::math::powm1<%1%>(%1%, %1%)";

   if ((fabs(y * (x - 1)) < T(0.5)) || (fabs(y) < T(0.2)))
   {
      // We don't have any good/quick approximation for log(x) * y
      // so just try it and see:
      T l = y * log(x);
      if (l < T(0.5))
         return boost::math::expm1(l, pol);
      if (l > boost::math::tools::log_max_value<T>())
         return boost::math::policies::raise_overflow_error<T>(function, nullptr, pol);
      // fall through....
   }
   
   T result = pow(x, y) - 1;
   if((boost::math::isinf)(result))
      return result < 0 ? -boost::math::policies::raise_overflow_error<T>(function, nullptr, pol) : boost::math::policies::raise_overflow_error<T>(function, nullptr, pol);
   if((boost::math::isnan)(result))
      return boost::math::policies::raise_domain_error<T>(function, "Result of pow is complex or undefined", x, pol);
   return result;
}

template <class T, class Policy>
BOOST_MATH_GPU_ENABLED inline T powm1_imp_dispatch(const T x, const T y, const Policy& pol)
{
   BOOST_MATH_STD_USING

   if ((boost::math::signbit)(x)) // Need to error check -0 here as well
   {
      constexpr auto function = "boost::math::powm1<%1%>(%1%, %1%)";

      // y had better be an integer:
      if (boost::math::trunc(y) != y)
         return boost::math::policies::raise_domain_error<T>(function, "For non-integral exponent, expected base > 0 but got %1%", x, pol);
      if (boost::math::trunc(y / 2) == y / 2)
         return powm1_imp(T(-x), T(y), pol);
   }

   return powm1_imp(T(x), T(y), pol);
}

} // detail

template <class T1, class T2>
BOOST_MATH_GPU_ENABLED inline typename tools::promote_args<T1, T2>::type
   powm1(const T1 a, const T2 z)
{
   typedef typename tools::promote_args<T1, T2>::type result_type;
   return detail::powm1_imp_dispatch(static_cast<result_type>(a), static_cast<result_type>(z), policies::policy<>());
}

template <class T1, class T2, class Policy>
BOOST_MATH_GPU_ENABLED inline typename tools::promote_args<T1, T2>::type
   powm1(const T1 a, const T2 z, const Policy& pol)
{
   typedef typename tools::promote_args<T1, T2>::type result_type;
   return detail::powm1_imp_dispatch(static_cast<result_type>(a), static_cast<result_type>(z), pol);
}

} // namespace math
} // namespace boost

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif // BOOST_MATH_POWM1






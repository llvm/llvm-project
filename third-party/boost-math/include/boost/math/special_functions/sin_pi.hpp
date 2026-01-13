//  Copyright (c) 2007 John Maddock
//  Copyright (c) 2024 Matt Borland
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_SIN_PI_HPP
#define BOOST_MATH_SIN_PI_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/tools/config.hpp>

#ifndef BOOST_MATH_HAS_NVRTC

#include <cmath>
#include <limits>
#include <type_traits>
#include <boost/math/tools/numeric_limits.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/special_functions/trunc.hpp>
#include <boost/math/tools/promotion.hpp>
#include <boost/math/constants/constants.hpp>

namespace boost{ namespace math{ namespace detail{

template <class T, class Policy>
BOOST_MATH_GPU_ENABLED inline T sin_pi_imp(T x, const Policy&)
{
   BOOST_MATH_STD_USING // ADL of std names
   // sin of pi*x:
   if(x < T(0.5))
      return sin(constants::pi<T>() * x);
   bool invert;
   if(x < 1)
   {
      invert = true;
      x = -x;
   }
   else
      invert = false;

   T rem = floor(x);
   if(abs(floor(rem/2)*2 - rem) > boost::math::numeric_limits<T>::epsilon())
   {
      invert = !invert;
   }
   rem = x - rem;
   if(rem > 0.5f)
      rem = 1 - rem;
   if(rem == 0.5f)
      return static_cast<T>(invert ? -1 : 1);
   
   rem = sin(constants::pi<T>() * rem);
   return invert ? T(-rem) : rem;
}

template <class T, class Policy>
BOOST_MATH_GPU_ENABLED inline T sin_pi_dispatch(T x, const Policy& pol)
{
   if (x < T(0))
   {
      return -sin_pi_imp(T(-x), pol);
   }
   else
   {
      return sin_pi_imp(T(x), pol);
   }
}

} // namespace detail

template <class T, class Policy>
BOOST_MATH_GPU_ENABLED inline typename tools::promote_args<T>::type sin_pi(T x, const Policy&)
{
   typedef typename tools::promote_args<T>::type result_type;
   typedef typename policies::evaluation<result_type, Policy>::type value_type;
   typedef typename policies::normalise<
      Policy,
      policies::promote_float<false>,
      policies::promote_double<false>,
      policies::discrete_quantile<>,
      policies::assert_undefined<>,
      // We want to ignore overflows since the result is in [-1,1] and the 
      // check slows the code down considerably.
      policies::overflow_error<policies::ignore_error> >::type forwarding_policy;
   return policies::checked_narrowing_cast<result_type, forwarding_policy>(boost::math::detail::sin_pi_dispatch<value_type>(x, forwarding_policy()), "sin_pi");
}

template <class T>
inline typename tools::promote_args<T>::type sin_pi(T x)
{
   return boost::math::sin_pi(x, policies::policy<>());
}

} // namespace math
} // namespace boost

#else // Special handling for NVRTC

namespace boost {
namespace math {

template <typename T>
BOOST_MATH_GPU_ENABLED auto sin_pi(T x)
{
   return ::sinpi(x);
}

template <>
BOOST_MATH_GPU_ENABLED auto sin_pi(float x)
{
   return ::sinpif(x);
}

template <typename T, typename Policy>
BOOST_MATH_GPU_ENABLED auto sin_pi(T x, const Policy&)
{
   return ::sinpi(x);
}

template <typename Policy>
BOOST_MATH_GPU_ENABLED auto sin_pi(float x, const Policy&)
{
   return ::sinpif(x);
}

} // namespace math
} // namespace boost

#endif // BOOST_MATH_HAS_NVRTC

#endif


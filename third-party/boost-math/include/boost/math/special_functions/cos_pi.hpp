//  Copyright (c) 2007 John Maddock
//  Copyright (c) 2024 Matt Borland
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_COS_PI_HPP
#define BOOST_MATH_COS_PI_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/tools/config.hpp>

#ifndef BOOST_MATH_HAS_NVRTC

#include <cmath>
#include <limits>
#include <boost/math/tools/numeric_limits.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/special_functions/trunc.hpp>
#include <boost/math/tools/promotion.hpp>
#include <boost/math/constants/constants.hpp>

namespace boost{ namespace math{ namespace detail{

template <class T, class Policy>
BOOST_MATH_GPU_ENABLED T cos_pi_imp(T x, const Policy&)
{
   BOOST_MATH_STD_USING // ADL of std names
   // cos of pi*x:
   bool invert = false;
   if(fabs(x) < T(0.25))
      return cos(constants::pi<T>() * x);

   if(x < 0)
   {
      x = -x;
   }
   T rem = floor(x);
   if(abs(floor(rem/2)*2 - rem) > boost::math::numeric_limits<T>::epsilon())
   {
      invert = !invert;
   }
   rem = x - rem;
   if(rem > 0.5f)
   {
      rem = 1 - rem;
      invert = !invert;
   }
   if(rem == 0.5f)
      return 0;
   
   if(rem > 0.25f)
   {
      rem = 0.5f - rem;
      rem = sin(constants::pi<T>() * rem);
   }
   else
      rem = cos(constants::pi<T>() * rem);
   return invert ? T(-rem) : rem;
}

} // namespace detail

template <class T, class Policy>
BOOST_MATH_GPU_ENABLED inline typename tools::promote_args<T>::type cos_pi(T x, const Policy&)
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
   return policies::checked_narrowing_cast<result_type, forwarding_policy>(boost::math::detail::cos_pi_imp<value_type>(x, forwarding_policy()), "cos_pi");
}

template <class T>
BOOST_MATH_GPU_ENABLED inline typename tools::promote_args<T>::type cos_pi(T x)
{
   return boost::math::cos_pi(x, policies::policy<>());
}

} // namespace math
} // namespace boost

#else // Special handling for NVRTC

namespace boost {
namespace math {

template <typename T>
BOOST_MATH_GPU_ENABLED auto cos_pi(T x)
{
   return ::cospi(x);
}

template <>
BOOST_MATH_GPU_ENABLED auto cos_pi(float x)
{
   return ::cospif(x);
}

template <typename T, typename Policy>
BOOST_MATH_GPU_ENABLED auto cos_pi(T x, const Policy&)
{
   return ::cospi(x);
}

template <typename Policy>
BOOST_MATH_GPU_ENABLED auto cos_pi(float x, const Policy&)
{
   return ::cospif(x);
}

} // namespace math
} // namespace boost

#endif // BOOST_MATH_HAS_NVRTC

#endif


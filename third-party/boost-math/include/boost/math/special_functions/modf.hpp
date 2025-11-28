//  Copyright John Maddock 2007.
//  Copyright Matt Borland 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_MODF_HPP
#define BOOST_MATH_MODF_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/tools/config.hpp>
#include <boost/math/special_functions/trunc.hpp>
#include <boost/math/policies/policy.hpp>

#ifndef BOOST_MATH_HAS_NVRTC
#include <boost/math/special_functions/math_fwd.hpp>
#endif

namespace boost{ namespace math{

template <class T, class Policy>
BOOST_MATH_GPU_ENABLED inline T modf(const T& v, T* ipart, const Policy& pol)
{
   *ipart = trunc(v, pol);
   return v - *ipart;
}
template <class T>
BOOST_MATH_GPU_ENABLED inline T modf(const T& v, T* ipart)
{
   return modf(v, ipart, policies::policy<>());
}

template <class T, class Policy>
BOOST_MATH_GPU_ENABLED inline T modf(const T& v, int* ipart, const Policy& pol)
{
   *ipart = itrunc(v, pol);
   return v - *ipart;
}
template <class T>
BOOST_MATH_GPU_ENABLED inline T modf(const T& v, int* ipart)
{
   return modf(v, ipart, policies::policy<>());
}

template <class T, class Policy>
BOOST_MATH_GPU_ENABLED inline T modf(const T& v, long* ipart, const Policy& pol)
{
   *ipart = ltrunc(v, pol);
   return v - *ipart;
}
template <class T>
BOOST_MATH_GPU_ENABLED inline T modf(const T& v, long* ipart)
{
   return modf(v, ipart, policies::policy<>());
}

template <class T, class Policy>
BOOST_MATH_GPU_ENABLED inline T modf(const T& v, long long* ipart, const Policy& pol)
{
   *ipart = lltrunc(v, pol);
   return v - *ipart;
}
template <class T>
BOOST_MATH_GPU_ENABLED inline T modf(const T& v, long long* ipart)
{
   return modf(v, ipart, policies::policy<>());
}

}} // namespaces

#endif // BOOST_MATH_MODF_HPP

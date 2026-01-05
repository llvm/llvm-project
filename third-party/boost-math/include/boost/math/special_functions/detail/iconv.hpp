//  Copyright (c) 2009 John Maddock
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_ICONV_HPP
#define BOOST_MATH_ICONV_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/tools/config.hpp>
#include <boost/math/tools/type_traits.hpp>
#include <boost/math/special_functions/round.hpp>

namespace boost { namespace math { namespace detail{

template <class T, class Policy>
BOOST_MATH_GPU_ENABLED inline int iconv_imp(T v, Policy const&, boost::math::true_type const&)
{
   return static_cast<int>(v);
}

template <class T, class Policy>
BOOST_MATH_GPU_ENABLED inline int iconv_imp(T v, Policy const& pol, boost::math::false_type const&)
{
   BOOST_MATH_STD_USING
   return iround(v, pol);
}

template <class T, class Policy>
BOOST_MATH_GPU_ENABLED inline int iconv(T v, Policy const& pol)
{
   typedef typename boost::math::is_convertible<T, int>::type tag_type;
   return iconv_imp(v, pol, tag_type());
}


}}} // namespaces

#endif // BOOST_MATH_ICONV_HPP


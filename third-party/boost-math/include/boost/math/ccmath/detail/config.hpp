//  (C) Copyright John Maddock 2023.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Core configuration for ccmath functions, basically will they work or not?

#ifndef BOOST_MATH_CCMATH_DETAIL_CONFIG
#define BOOST_MATH_CCMATH_DETAIL_CONFIG

#include <cmath>
#include <type_traits>
#include <limits>
#include <boost/math/tools/is_constant_evaluated.hpp>
#include <boost/math/tools/is_standalone.hpp>

#ifndef BOOST_MATH_STANDALONE

#include <boost/config.hpp>
#ifdef BOOST_MATH_NO_CXX17_IF_CONSTEXPR
#  define BOOST_MATH_NO_CCMATH
#endif

#else // BOOST_MATH_STANDALONE

#if defined(_MSC_VER)

#if defined(_MSVC_LANG) && (_MSVC_LANG < 201703)
#  define BOOST_MATH_NO_CCMATH
#endif

#else // _MSC_VER

#if (__cplusplus < 201703)
#  define BOOST_MATH_NO_CCMATH
#endif

#endif

#endif

#ifndef _MSC_VER
//
// Don't check here for msvc as they didn't get std lib configuration macros at the same time as C++17 <type_traits>
//
#if (defined(__cpp_lib_bool_constant) && __cpp_lib_bool_constant < 201505L) && !defined(BOOST_MATH_NO_CCMATH)
#  define BOOST_MATH_NO_CCMATH
#endif
#endif


#endif

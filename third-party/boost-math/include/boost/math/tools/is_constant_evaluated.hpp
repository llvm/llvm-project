//  Copyright John Maddock 2011-2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TOOLS_IS_CONSTANT_EVALUATED_HPP
#define BOOST_MATH_TOOLS_IS_CONSTANT_EVALUATED_HPP

#include <boost/math/tools/config.hpp>

#ifdef __has_include
# if __has_include(<version>)
#  include <version>
#  ifdef __cpp_lib_is_constant_evaluated
#   include <type_traits>
#   define BOOST_MATH_HAS_IS_CONSTANT_EVALUATED
#  endif
# endif
#endif

#ifdef __has_builtin
#  if __has_builtin(__builtin_is_constant_evaluated) && !defined(BOOST_MATH_NO_CXX14_CONSTEXPR) && !defined(BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX)
#    define BOOST_MATH_HAS_BUILTIN_IS_CONSTANT_EVALUATED
#  endif
#endif
//
// MSVC also supports __builtin_is_constant_evaluated if it's recent enough:
//
#if defined(_MSC_FULL_VER) && (_MSC_FULL_VER >= 192528326)
#  define BOOST_MATH_HAS_BUILTIN_IS_CONSTANT_EVALUATED
#endif
//
// As does GCC-9:
//
#if !defined(BOOST_MATH_NO_CXX14_CONSTEXPR) && (__GNUC__ >= 9) && !defined(BOOST_MATH_HAS_BUILTIN_IS_CONSTANT_EVALUATED)
#  define BOOST_MATH_HAS_BUILTIN_IS_CONSTANT_EVALUATED
#endif

#if defined(BOOST_MATH_HAS_IS_CONSTANT_EVALUATED) && !defined(BOOST_MATH_NO_CXX14_CONSTEXPR)
#  define BOOST_MATH_IS_CONSTANT_EVALUATED(x) std::is_constant_evaluated()
#elif defined(BOOST_MATH_HAS_BUILTIN_IS_CONSTANT_EVALUATED)
#  define BOOST_MATH_IS_CONSTANT_EVALUATED(x) __builtin_is_constant_evaluated()
#elif !defined(BOOST_MATH_NO_CXX14_CONSTEXPR) && (__GNUC__ >= 6)
#  define BOOST_MATH_IS_CONSTANT_EVALUATED(x) __builtin_constant_p(x)
#  define BOOST_MATH_USING_BUILTIN_CONSTANT_P
#else
#  define BOOST_MATH_IS_CONSTANT_EVALUATED(x) false
#  define BOOST_MATH_NO_CONSTEXPR_DETECTION
#endif

#endif // BOOST_MATH_TOOLS_IS_CONSTANT_EVALUATED_HPP

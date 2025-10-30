//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TOOLS_THROW_EXCEPTION_HPP
#define BOOST_MATH_TOOLS_THROW_EXCEPTION_HPP

#include <boost/math/tools/is_standalone.hpp>

#ifndef BOOST_MATH_STANDALONE

#if defined(_MSC_VER) || defined(__GNUC__)
# pragma push_macro( "I" )
# undef I
#endif

#include <boost/throw_exception.hpp>
#define BOOST_MATH_THROW_EXCEPTION(expr) boost::throw_exception(expr);

#if defined(_MSC_VER) || defined(__GNUC__)
# pragma pop_macro( "I" )
#endif

#else // Standalone mode - use standard library facilities

#ifdef _MSC_VER
#  ifdef _CPPUNWIND
#    define BOOST_MATH_THROW_EXCEPTION(expr) throw expr;
#  else
#    define BOOST_MATH_THROW_EXCEPTION(expr)
#  endif
#else
#  ifdef __EXCEPTIONS
#    define BOOST_MATH_THROW_EXCEPTION(expr) throw expr;
#  else
#    define BOOST_MATH_THROW_EXCEPTION(expr)
#  endif
#endif

#endif // BOOST_MATH_STANDALONE

#endif // BOOST_MATH_TOOLS_THROW_EXCEPTION_HPP

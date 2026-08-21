//  (C) Copyright Antony Polukhin 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TOOLS_NOTHROW_HPP
#define BOOST_MATH_TOOLS_NOTHROW_HPP

#include <boost/math/tools/is_standalone.hpp>

#ifndef BOOST_MATH_STANDALONE

#include <boost/config.hpp>

#define BOOST_MATH_NOTHROW BOOST_NOEXCEPT_OR_NOTHROW

#else // Standalone mode - use noexcept or throw()

#if __cplusplus >= 201103L
#define BOOST_MATH_NOTHROW noexcept
#else
#define BOOST_MATH_NOTHROW throw()
#endif

#endif

#endif // BOOST_MATH_TOOLS_NOTHROW_HPP

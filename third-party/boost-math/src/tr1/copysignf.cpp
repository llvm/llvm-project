//  Copyright John Maddock 2008.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0.  (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#  include <pch.hpp>
#ifndef BOOST_MATH_TR1_SOURCE
#  define BOOST_MATH_TR1_SOURCE
#endif
#include <boost/math/tr1.hpp>
#include <boost/math/special_functions/sign.hpp>
#include "c_policy.hpp"

extern "C" float BOOST_MATH_TR1_DECL boost_copysignf BOOST_MATH_PREVENT_MACRO_SUBSTITUTION(float x, float y) BOOST_MATH_C99_THROW_SPEC
{
   return boost::math::copysign BOOST_MATH_PREVENT_MACRO_SUBSTITUTION(x, y);
}




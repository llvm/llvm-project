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
#include <boost/math/special_functions/next.hpp>
#include "c_policy.hpp"

namespace boost{ namespace math{ namespace tr1{

extern "C" float BOOST_MATH_TR1_DECL boost_nexttowardf BOOST_MATH_PREVENT_MACRO_SUBSTITUTION(float x, long double y) BOOST_MATH_C99_THROW_SPEC
{
#ifdef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   return  (float)c_policies::nextafter BOOST_MATH_PREVENT_MACRO_SUBSTITUTION((double)x, (double)y);
#else
   return  (float)c_policies::nextafter BOOST_MATH_PREVENT_MACRO_SUBSTITUTION((long double)x, y);
#endif
}

}}}



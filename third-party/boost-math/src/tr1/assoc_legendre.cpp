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
#include <boost/math/special_functions/legendre.hpp>
#include "c_policy.hpp"

extern "C" double BOOST_MATH_TR1_DECL boost_assoc_legendre BOOST_MATH_PREVENT_MACRO_SUBSTITUTION(unsigned l, unsigned m, double x) BOOST_MATH_C99_THROW_SPEC
{
   return (m&1 ? -1 : 1) * c_policies::legendre_p BOOST_MATH_PREVENT_MACRO_SUBSTITUTION(l, m, x);
}



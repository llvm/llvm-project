
//  Copyright John Maddock 2007-8.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//
// This tests two things: that mpfr::mpreal meets our
// conceptual requirements, and that we can instantiate
// all our distributions and special functions on this type.
//
#define BOOST_MATH_ASSERT_UNDEFINED_POLICY false
#define TEST_MPFR

#include <boost/math/bindings/mpreal.hpp>
#include <boost/math/concepts/real_type_concept.hpp>
#include "compile_test/instantiate.hpp"
#include <boost/math/tools/assert.hpp>

//static_assert((boost::is_same<mpfr::mpreal, boost::math::tools::promote_args<mpfr::mpreal>::type >::value));

void foo()
{
   instantiate(mpfr::mpreal());
}

int main()
{
   BOOST_CONCEPT_ASSERT((boost::math::concepts::RealTypeConcept<mpfr::mpreal>));
   return 0;
}



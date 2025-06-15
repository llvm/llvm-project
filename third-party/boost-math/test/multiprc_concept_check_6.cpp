
//  Copyright John Maddock 2020.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//
// This tests two things: that multiprecision::number meets our
// conceptual requirements, and that we can instantiate
// all our distributions and special functions on this type.
//
#define BOOST_MATH_ASSERT_UNDEFINED_POLICY false
#define TEST_GROUP_7

#ifdef BOOST_MATH_RUN_MP_TESTS

#ifdef _MSC_VER
#  pragma warning(disable:4800)
#  pragma warning(disable:4512)
#  pragma warning(disable:4127)
#  pragma warning(disable:4512)
#  pragma warning(disable:4503) // decorated name length exceeded, name was truncated
#endif

#include <boost/multiprecision/cpp_dec_float.hpp>
#include "compile_test/instantiate.hpp"

using namespace boost::multiprecision;

typedef number<cpp_dec_float<50>, et_on> test_type;

// We get sporadic internal compiler errors from gcc-7.x when CI testing
// that don't appear to be reproducible locally.  gcc-6.x and gcc-8.x are fine
// so for now it's a <shrug> and move on...
#if ! (defined(BOOST_GCC) && (__GNUC__ == 7))

void foo()
{
   instantiate(test_type());
}

#endif

int main()
{
   //BOOST_CONCEPT_ASSERT((boost::math::concepts::RealTypeConcept<test_type>));
}

#else

int main()
{
   return 0;
}

#endif

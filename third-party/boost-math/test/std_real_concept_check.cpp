//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_MATH_ASSERT_UNDEFINED_POLICY false

#include <boost/math/concepts/std_real_concept.hpp>
#include <boost/math/concepts/distributions.hpp>

#include "compile_test/instantiate.hpp"

//
// The purpose of this test is to verify that our code compiles
// cleanly with a type whose std lib functions are in namespace
// std and can *not* be found by ADL.  This verifies that we're
// not finding std lib functions that are in the global namespace
// for example calling ::pow(double) rather than std::pow(long double).
// This is a silent error that does the wrong thing at runtime, and
// of course we can't call std::pow() directly because we want
// the functions to be found by ADL when that's appropriate.
//
// Furthermore our code does different things internally depending
// on numeric_limits<>::digits, so there are some macros that can
// be defined that cause our concept-archetype to emulate various
// floating point types:
//
// EMULATE32: 32-bit float
// EMULATE64: 64-bit double
// EMULATE80: 80-bit long double
// EMULATE128: 128-bit long double
//
// In order to ensure total code coverage this file must be
// compiled with each of the above macros in turn, and then 
// without any of the above as well!
//

#define NULL_MACRO /**/
#ifdef EMULATE32
namespace std{
template<>
struct numeric_limits<boost::math::concepts::std_real_concept>
{
   static const bool is_specialized = true;
   static boost::math::concepts::std_real_concept min NULL_MACRO() BOOST_NOEXCEPT_OR_NOTHROW;
   static boost::math::concepts::std_real_concept max NULL_MACRO() BOOST_NOEXCEPT_OR_NOTHROW;
   static const int digits = 24;
   static const int digits10 = 6;
   static const bool is_signed = true;
   static const bool is_integer = false;
   static const bool is_exact = false;
   static const int radix = 2;
   static boost::math::concepts::std_real_concept epsilon() BOOST_NOEXCEPT_OR_NOTHROW;
   static boost::math::concepts::std_real_concept round_error() BOOST_NOEXCEPT_OR_NOTHROW;
   static const int min_exponent = -125;
   static const int min_exponent10 = -37;
   static const int max_exponent = 128;
   static const int max_exponent10 = 38;
   static const bool has_infinity = true;
   static const bool has_quiet_NaN = true;
   static const bool has_signaling_NaN = true;
   static const float_denorm_style has_denorm = denorm_absent;
   static const bool has_denorm_loss = false;
   static boost::math::concepts::std_real_concept infinity() BOOST_NOEXCEPT_OR_NOTHROW;
   static boost::math::concepts::std_real_concept quiet_NaN() BOOST_NOEXCEPT_OR_NOTHROW;
   static boost::math::concepts::std_real_concept signaling_NaN() BOOST_NOEXCEPT_OR_NOTHROW;
   static boost::math::concepts::std_real_concept denorm_min() BOOST_NOEXCEPT_OR_NOTHROW;
   static const bool is_iec559 = true;
   static const bool is_bounded = false;
   static const bool is_modulo = false;
   static const bool traps = false;
   static const bool tinyness_before = false;
   static const float_round_style round_style = round_toward_zero;
#ifndef BOOST_NO_CXX11_NUMERIC_LIMITS
   static const int max_digits10 = digits10 + 2;
#endif
};
}
#endif
#ifdef EMULATE64
namespace std{
template<>
struct numeric_limits<boost::math::concepts::std_real_concept>
{
   static const bool is_specialized = true;
   static boost::math::concepts::std_real_concept min NULL_MACRO() BOOST_NOEXCEPT_OR_NOTHROW;
   static boost::math::concepts::std_real_concept max NULL_MACRO() BOOST_NOEXCEPT_OR_NOTHROW;
   static const int digits = 53;
   static const int digits10 = 15;
   static const bool is_signed = true;
   static const bool is_integer = false;
   static const bool is_exact = false;
   static const int radix = 2;
   static boost::math::concepts::std_real_concept epsilon() BOOST_NOEXCEPT_OR_NOTHROW;
   static boost::math::concepts::std_real_concept round_error() BOOST_NOEXCEPT_OR_NOTHROW;
   static const int min_exponent = -1021;
   static const int min_exponent10 = -307;
   static const int max_exponent = 1024;
   static const int max_exponent10 = 308;
   static const bool has_infinity = true;
   static const bool has_quiet_NaN = true;
   static const bool has_signaling_NaN = true;
   static const float_denorm_style has_denorm = denorm_absent;
   static const bool has_denorm_loss = false;
   static boost::math::concepts::std_real_concept infinity() BOOST_NOEXCEPT_OR_NOTHROW;
   static boost::math::concepts::std_real_concept quiet_NaN() BOOST_NOEXCEPT_OR_NOTHROW;
   static boost::math::concepts::std_real_concept signaling_NaN() BOOST_NOEXCEPT_OR_NOTHROW;
   static boost::math::concepts::std_real_concept denorm_min() BOOST_NOEXCEPT_OR_NOTHROW;
   static const bool is_iec559 = true;
   static const bool is_bounded = false;
   static const bool is_modulo = false;
   static const bool traps = false;
   static const bool tinyness_before = false;
   static const float_round_style round_style = round_toward_zero;
};
}
#endif
#ifdef EMULATE80
namespace std{
template<>
struct numeric_limits<boost::math::concepts::std_real_concept>
{
   static const bool is_specialized = true;
   static boost::math::concepts::std_real_concept min NULL_MACRO() BOOST_NOEXCEPT_OR_NOTHROW;
   static boost::math::concepts::std_real_concept max NULL_MACRO() BOOST_NOEXCEPT_OR_NOTHROW;
   static const int digits = 64;
   static const int digits10 = 18;
   static const bool is_signed = true;
   static const bool is_integer = false;
   static const bool is_exact = false;
   static const int radix = 2;
   static boost::math::concepts::std_real_concept epsilon() BOOST_NOEXCEPT_OR_NOTHROW;
   static boost::math::concepts::std_real_concept round_error() BOOST_NOEXCEPT_OR_NOTHROW;
   static const int min_exponent = -16381;
   static const int min_exponent10 = -4931;
   static const int max_exponent = 16384;
   static const int max_exponent10 = 4932;
   static const bool has_infinity = true;
   static const bool has_quiet_NaN = true;
   static const bool has_signaling_NaN = true;
   static const float_denorm_style has_denorm = denorm_absent;
   static const bool has_denorm_loss = false;
   static boost::math::concepts::std_real_concept infinity() BOOST_NOEXCEPT_OR_NOTHROW;
   static boost::math::concepts::std_real_concept quiet_NaN() BOOST_NOEXCEPT_OR_NOTHROW;
   static boost::math::concepts::std_real_concept signaling_NaN() BOOST_NOEXCEPT_OR_NOTHROW;
   static boost::math::concepts::std_real_concept denorm_min() BOOST_NOEXCEPT_OR_NOTHROW;
   static const bool is_iec559 = true;
   static const bool is_bounded = false;
   static const bool is_modulo = false;
   static const bool traps = false;
   static const bool tinyness_before = false;
   static const float_round_style round_style = round_toward_zero;
};
}
#endif
#ifdef EMULATE128
namespace std{
template<>
struct numeric_limits<boost::math::concepts::std_real_concept>
{
   static const bool is_specialized = true;
   static boost::math::concepts::std_real_concept min NULL_MACRO() BOOST_NOEXCEPT_OR_NOTHROW;
   static boost::math::concepts::std_real_concept max NULL_MACRO() BOOST_NOEXCEPT_OR_NOTHROW;
   static const int digits = 113;
   static const int digits10 = 33;
   static const bool is_signed = true;
   static const bool is_integer = false;
   static const bool is_exact = false;
   static const int radix = 2;
   static boost::math::concepts::std_real_concept epsilon() BOOST_NOEXCEPT_OR_NOTHROW;
   static boost::math::concepts::std_real_concept round_error() BOOST_NOEXCEPT_OR_NOTHROW;
   static const int min_exponent = -16381;
   static const int min_exponent10 = -4931;
   static const int max_exponent = 16384;
   static const int max_exponent10 = 4932;
   static const bool has_infinity = true;
   static const bool has_quiet_NaN = true;
   static const bool has_signaling_NaN = true;
   static const float_denorm_style has_denorm = denorm_absent;
   static const bool has_denorm_loss = false;
   static boost::math::concepts::std_real_concept infinity() BOOST_NOEXCEPT_OR_NOTHROW;
   static boost::math::concepts::std_real_concept quiet_NaN() BOOST_NOEXCEPT_OR_NOTHROW;
   static boost::math::concepts::std_real_concept signaling_NaN() BOOST_NOEXCEPT_OR_NOTHROW;
   static boost::math::concepts::std_real_concept denorm_min() BOOST_NOEXCEPT_OR_NOTHROW;
   static const bool is_iec559 = true;
   static const bool is_bounded = false;
   static const bool is_modulo = false;
   static const bool traps = false;
   static const bool tinyness_before = false;
   static const float_round_style round_style = round_toward_zero;
};
}
#endif



int main()
{
   instantiate(boost::math::concepts::std_real_concept(0));
}



//  Copyright John Maddock 2020.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_MATH_ASSERT_UNDEFINED_POLICY false
#include <boost/math/tools/config.hpp>
//
// Poison the long double std math functions so we can find accidental usage of these
// when the user has requested that we do *not* use them.
//
namespace poison {
long double abs(long double, void* = 0);
long double fabs(long double, void* = 0);
long double sin(long double, void* = 0);
long double cos(long double, void* = 0);
long double tan(long double, void* = 0);
long double asin(long double, void* = 0);
long double acos(long double, void* = 0);
long double atan(long double, void* = 0);
long double exp(long double, void* = 0);
long double log(long double, void* = 0);
long double pow(long double, long double, void* = 0);
long double fmod(long double, long double, void* = 0);
long double modf(long double, long double*, void* = 0);
long double cosh(long double, void* = 0);
long double sinh(long double, void* = 0);
long double tanh(long double, void* = 0);
long double frexp(long double, void*);
long double ldexp(long double, short);
long double atan2(long double, long double, void* = 0);
long double ceil(long double, void* = 0);
long double floor(long double, void* = 0);
long double log10(long double, void* = 0);
long double sqrt(long double, void* = 0);
} // namespace poison

#undef BOOST_MATH_STD_USING_CORE
#undef BOOST_MATH_STD_USING

#define BOOST_MATH_STD_USING_CORE \
   using std::abs;                \
   using std::acos;               \
   using std::cos;                \
   using std::fmod;               \
   using std::modf;               \
   using std::tan;                \
   using std::asin;               \
   using std::cosh;               \
   using std::frexp;              \
   using std::pow;                \
   using std::tanh;               \
   using std::atan;               \
   using std::exp;                \
   using std::ldexp;              \
   using std::sin;                \
   using std::atan2;              \
   using std::fabs;               \
   using std::log;                \
   using std::sinh;               \
   using std::ceil;               \
   using std::floor;              \
   using std::log10;              \
   using std::sqrt;                \
   using poison::abs;                \
   using poison::acos;               \
   using poison::cos;                \
   using poison::fmod;               \
   using poison::modf;               \
   using poison::tan;                \
   using poison::asin;               \
   using poison::cosh;               \
   using poison::frexp;              \
   using poison::pow;                \
   using poison::tanh;               \
   using poison::atan;               \
   using poison::exp;                \
   using poison::ldexp;              \
   using poison::sin;                \
   using poison::atan2;              \
   using poison::fabs;               \
   using poison::log;                \
   using poison::sinh;               \
   using poison::ceil;               \
   using poison::floor;              \
   using poison::log10;              \
   using poison::sqrt;

#define BOOST_MATH_STD_USING BOOST_MATH_STD_USING_CORE



#define TEST_GROUP_8
#define TEST_GROUP_9
#include "compile_test/instantiate.hpp"

int main()
{
   //boost::math::foo(0.0L);
   instantiate(0.0);
}

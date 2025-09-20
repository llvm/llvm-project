//  (C) Copyright John Maddock 2005.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pch_light.hpp>

#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error
#define BOOST_TEST_MAIN 
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/special_functions/math_fwd.hpp>

#include <cmath>

#ifdef BOOST_NO_STDC_NAMESPACE
namespace std{ using ::sqrt; }
#endif

//
// test_boundaries:
// This is an accuracy test, sets the two arguments to hypot to just
// above or just below various boundary conditions, and checks the accuracy
// of the result.  The values computed at double precision will use a 
// different computation method to those computed at float precision:
// as long as these compute the same values then everything's OK.
//
// Tolerance is 2*epsilon, expressed here as a percentage:
//
static const float tolerance = 200 * (std::numeric_limits<float>::epsilon)();
const float boundaries[] = {
   0,
   1,
   2,
   (std::numeric_limits<float>::max)()/2,
   (std::numeric_limits<float>::min)(),
   std::numeric_limits<float>::epsilon(),
   std::sqrt((std::numeric_limits<float>::max)()) / 2,
   std::sqrt((std::numeric_limits<float>::min)()),
   std::sqrt((std::numeric_limits<float>::max)()) / 4,
   std::sqrt((std::numeric_limits<float>::min)()) * 2,
};

void do_test_boundaries(float x, float y)
{
   float expected = static_cast<float>((boost::math::hypot)(
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
      static_cast<long double>(x), 
      static_cast<long double>(y)));
#else
      static_cast<double>(x), 
      static_cast<double>(y)));
#endif
   float found = (boost::math::hypot)(x, y);
   BOOST_CHECK_CLOSE(expected, found, tolerance);
}

void test_boundaries(float x, float y)
{
   do_test_boundaries(x, y);
   do_test_boundaries(-x, y); 
   do_test_boundaries(-x, -y);
   do_test_boundaries(x, -y);
}

void test_boundaries(float x)
{
   for(unsigned i = 0; i < sizeof(boundaries)/sizeof(float); ++i)
   {
      test_boundaries(x, boundaries[i]);
      test_boundaries(x, boundaries[i] + std::numeric_limits<float>::epsilon()*boundaries[i]);
      test_boundaries(x, boundaries[i] - std::numeric_limits<float>::epsilon()*boundaries[i]);
   }
}

void test_boundaries()
{
   for(unsigned i = 0; i < sizeof(boundaries)/sizeof(float); ++i)
   {
      test_boundaries(boundaries[i]);
      test_boundaries(boundaries[i] + std::numeric_limits<float>::epsilon()*boundaries[i]);
      test_boundaries(boundaries[i] - std::numeric_limits<float>::epsilon()*boundaries[i]);
   }
}

void test_spots()
{
   static const float zero = 0;
   for(unsigned i = 0; i < sizeof(boundaries)/sizeof(float); ++i)
   {
      BOOST_CHECK_EQUAL(boost::math::hypot(boundaries[i], zero), std::fabs(boundaries[i]));
      BOOST_CHECK_EQUAL(boost::math::hypot(-boundaries[i], zero), std::fabs(-boundaries[i]));
      BOOST_CHECK_EQUAL(boost::math::hypot(boundaries[i], -zero), std::fabs(boundaries[i]));
      BOOST_CHECK_EQUAL(boost::math::hypot(-boundaries[i], -zero), std::fabs(-boundaries[i]));
      for(unsigned j = 0; j < sizeof(boundaries)/sizeof(float); ++j)
      {
         BOOST_CHECK_EQUAL(boost::math::hypot(boundaries[i], boundaries[j]), boost::math::hypot(boundaries[j], boundaries[i]));
         BOOST_CHECK_EQUAL(boost::math::hypot(boundaries[i], boundaries[j]), boost::math::hypot(boundaries[i], -boundaries[j]));
         BOOST_CHECK_EQUAL(boost::math::hypot(-boundaries[i], -boundaries[j]), boost::math::hypot(-boundaries[j], -boundaries[i]));
         BOOST_CHECK_EQUAL(boost::math::hypot(-boundaries[i], -boundaries[j]), boost::math::hypot(-boundaries[i], boundaries[j]));
      }
   }
   if((std::numeric_limits<float>::has_infinity) && (std::numeric_limits<float>::has_quiet_NaN))
   {
      static const float nan = std::numeric_limits<float>::quiet_NaN();
      static const float inf = std::numeric_limits<float>::infinity();
      BOOST_CHECK_EQUAL(boost::math::hypot(inf, nan), inf);
      BOOST_CHECK_EQUAL(boost::math::hypot(-inf, nan), inf);
      BOOST_CHECK_EQUAL(boost::math::hypot(nan, inf), inf);
      BOOST_CHECK_EQUAL(boost::math::hypot(nan, -inf), inf);
      for(unsigned j = 0; j < sizeof(boundaries)/sizeof(float); ++j)
      {
         BOOST_CHECK_EQUAL(boost::math::hypot(boundaries[j], inf), inf);
         BOOST_CHECK_EQUAL(boost::math::hypot(-boundaries[j], inf), inf);
         BOOST_CHECK_EQUAL(boost::math::hypot(inf, boundaries[j]), inf);
         BOOST_CHECK_EQUAL(boost::math::hypot(inf, -boundaries[j]), inf);
         BOOST_CHECK_EQUAL(boost::math::hypot(boundaries[j], -inf), inf);
         BOOST_CHECK_EQUAL(boost::math::hypot(-boundaries[j], -inf), inf);
         BOOST_CHECK_EQUAL(boost::math::hypot(-inf, boundaries[j]), inf);
         BOOST_CHECK_EQUAL(boost::math::hypot(-inf, -boundaries[j]), inf);
      }
   }
}

BOOST_AUTO_TEST_CASE( test_main )
{
   BOOST_MATH_CONTROL_FP;
   test_boundaries();
   test_spots();
}

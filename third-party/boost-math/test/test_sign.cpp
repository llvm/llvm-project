// Copyright John Maddock 2008
//  (C) Copyright Paul A. Bristow 2011 (added tests for changesign)
// Copyright Matt Borland 2024
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/concepts/real_concept.hpp> // for real_concept
#include <boost/math/special_functions/sign.hpp>

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // Boost.Test
#include <boost/test/results_collector.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

#include <iostream>
   using std::cout;
   using std::endl;
   using std::setprecision;

template <class RealType>
void test_spots(RealType /*T*/, const char* /*type_name*/)
{
   // Basic sanity checks.
   RealType a = 0;
   RealType b = 1;
   RealType c = -1;
   BOOST_CHECK_EQUAL((boost::math::signbit)(a), 0);
   BOOST_CHECK_EQUAL((boost::math::sign)(a), 0);
   BOOST_CHECK_EQUAL((boost::math::changesign)(b), RealType(-1));
   BOOST_CHECK_EQUAL((boost::math::changesign)(c), RealType(+1));
   BOOST_CHECK_EQUAL((boost::math::changesign)(a), RealType(0));

   // Compare to formula for changsign(x) = copysign(x, signbit(x) ? 1.0 : -1.0)
   BOOST_CHECK_EQUAL((boost::math::changesign)(b),
      (boost::math::copysign)(b, (boost::math::signbit)(b) ? RealType(1.) : RealType(-1.) ));


   BOOST_CHECK_EQUAL((boost::math::copysign)(b, a), RealType(1));
   BOOST_CHECK_EQUAL((boost::math::copysign)(c, a), RealType(1));
   a = 1;
   BOOST_CHECK_EQUAL((boost::math::signbit)(a), 0);
   BOOST_CHECK_EQUAL((boost::math::sign)(a), 1);
   BOOST_CHECK_EQUAL((boost::math::copysign)(b, a), RealType(1));
   BOOST_CHECK_EQUAL((boost::math::copysign)(c, a), RealType(1));
   a = -1;
   BOOST_CHECK((boost::math::signbit)(a) != 0);
   BOOST_CHECK_EQUAL((boost::math::sign)(a), -1);
   BOOST_CHECK_EQUAL((boost::math::copysign)(b, a), RealType(-1));
   BOOST_CHECK_EQUAL((boost::math::copysign)(c, a), RealType(-1));
   a = boost::math::tools::max_value<RealType>();
   BOOST_CHECK_EQUAL((boost::math::signbit)(a), 0);
   BOOST_CHECK_EQUAL((boost::math::sign)(a), 1);
   BOOST_CHECK_EQUAL((boost::math::copysign)(b, a), RealType(1));
   BOOST_CHECK_EQUAL((boost::math::copysign)(c, a), RealType(1));
   a = -boost::math::tools::max_value<RealType>();
   BOOST_CHECK((boost::math::signbit)(a) != 0);
   BOOST_CHECK_EQUAL((boost::math::sign)(a), -1);
   BOOST_CHECK_EQUAL((boost::math::copysign)(b, a), RealType(-1));
   BOOST_CHECK_EQUAL((boost::math::copysign)(c, a), RealType(-1));

   if(std::numeric_limits<RealType>::has_infinity)
   {
      a = std::numeric_limits<RealType>::infinity();
      BOOST_CHECK_EQUAL((boost::math::signbit)(a), 0);
      BOOST_CHECK_EQUAL((boost::math::sign)(a), 1);
      BOOST_CHECK_EQUAL((boost::math::copysign)(b, a), RealType(1));
      BOOST_CHECK_EQUAL((boost::math::copysign)(c, a), RealType(1));
      BOOST_CHECK_EQUAL((boost::math::changesign)(a), -a);

      a = -std::numeric_limits<RealType>::infinity();
      BOOST_CHECK((boost::math::signbit)(a) != 0);
      BOOST_CHECK_EQUAL((boost::math::sign)(a), -1);
      BOOST_CHECK_EQUAL((boost::math::copysign)(b, a), RealType(-1));
      BOOST_CHECK_EQUAL((boost::math::copysign)(c, a), RealType(-1));
      BOOST_CHECK_EQUAL((boost::math::changesign)(a), -a);
   }
#if !defined(__SUNPRO_CC) && !defined(__INTEL_COMPILER)
   if(std::numeric_limits<RealType>::has_quiet_NaN)
   {
      a = std::numeric_limits<RealType>::quiet_NaN();
      BOOST_CHECK_EQUAL((boost::math::signbit)(a), 0);
      BOOST_CHECK_EQUAL((boost::math::sign)(a), 1);
      BOOST_CHECK_EQUAL((boost::math::copysign)(b, a), RealType(1));
      BOOST_CHECK_EQUAL((boost::math::copysign)(c, a), RealType(1));
      // BOOST_CHECK_EQUAL((boost::math::changesign)(a), -a); // NaN comparison fails always!
      BOOST_CHECK((boost::math::signbit)((boost::math::changesign)(a)) != 0);

      a = -std::numeric_limits<RealType>::quiet_NaN();
      BOOST_CHECK((boost::math::signbit)(a) != 0);
      BOOST_CHECK_EQUAL((boost::math::sign)(a), -1);
      BOOST_CHECK_EQUAL((boost::math::copysign)(b, a), RealType(-1));
      BOOST_CHECK_EQUAL((boost::math::copysign)(c, a), RealType(-1));
      //BOOST_CHECK_EQUAL((boost::math::changesign)(a), -a); // NaN comparison fails always!
      BOOST_CHECK_EQUAL((boost::math::signbit)((boost::math::changesign)(a)), 0);

   }
#endif
   //
   // Try some extreme values:
   //
   a = boost::math::tools::min_value<RealType>();
   b = -a;
   c = -1;
   BOOST_CHECK((boost::math::signbit)(a) == 0);
   BOOST_CHECK_EQUAL((boost::math::sign)(a), 1);
   BOOST_CHECK_EQUAL((boost::math::copysign)(c, a), RealType(1));
   BOOST_CHECK((boost::math::signbit)(b) != 0);
   BOOST_CHECK_EQUAL((boost::math::sign)(b), -1);
   c = 1;
   BOOST_CHECK_EQUAL((boost::math::copysign)(c, b), RealType(-1));
   //
   // try denormalised values:
   //
   a /= 4;
   if(a != 0)
   {
      b = -a;
      c = -1;
      BOOST_CHECK((boost::math::signbit)(a) == 0);
      BOOST_CHECK_EQUAL((boost::math::sign)(a), 1);
      BOOST_CHECK_EQUAL((boost::math::copysign)(c, a), RealType(1));
      BOOST_CHECK((boost::math::signbit)(b) != 0);
      BOOST_CHECK_EQUAL((boost::math::sign)(b), -1);
      c = 1;
      BOOST_CHECK_EQUAL((boost::math::copysign)(c, b), RealType(-1));
   }
   a = boost::math::tools::max_value<RealType>() / 2;
   b = -a;
   c = -1;
   BOOST_CHECK((boost::math::signbit)(a) == 0);
   BOOST_CHECK_EQUAL((boost::math::sign)(a), 1);
   BOOST_CHECK_EQUAL((boost::math::copysign)(c, a), RealType(1));
   BOOST_CHECK((boost::math::signbit)(b) != 0);
   BOOST_CHECK_EQUAL((boost::math::sign)(b), -1);
   c = 1;
   BOOST_CHECK_EQUAL((boost::math::copysign)(c, b), RealType(-1));
}


BOOST_AUTO_TEST_CASE( test_main )
{
   // Basic sanity-check spot values.
   // (Parameter value, arbitrarily zero, only communicates the floating point type).
   test_spots(0.0F, "float"); // Test float. OK at decdigits = 0 tolerance = 0.0001 %
   test_spots(0.0, "double"); // Test double. OK at decdigits 7, tolerance = 1e07 %
   // long double support for the sign functions is considered "core" so we always test it
   // even when long double support is turned off via BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   test_spots(0.0L, "long double"); // Test long double.
#endif
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
   test_spots(boost::math::concepts::real_concept(0), "real_concept"); // Test real_concept.
#endif

   
} // BOOST_AUTO_TEST_CASE( test_main )


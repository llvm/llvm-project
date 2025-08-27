// Copyright John Maddock 2013.
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

//
// This file verifies that certain core functions are always
// available, even when long double support is patchy at best
// and BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS is defined.
//
#define BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // Boost.Test
#include <boost/math/special_functions/sign.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

BOOST_AUTO_TEST_CASE( test_main )
{
   BOOST_CHECK_EQUAL((boost::math::signbit)(1.0L), 0.0L);
   BOOST_CHECK((boost::math::signbit)(-1.0L) != 0);
   BOOST_CHECK_EQUAL((boost::math::sign)(1.0L), 1.0L);
   BOOST_CHECK_EQUAL((boost::math::sign)(-1.0L), -1.0L);
   BOOST_CHECK_EQUAL((boost::math::changesign)(1.0L), -1.0L);
   BOOST_CHECK_EQUAL((boost::math::changesign)(-1.0L), 1.0L);

   BOOST_CHECK_EQUAL((boost::math::fpclassify)(1.0L), (int)FP_NORMAL);
   BOOST_CHECK_EQUAL((boost::math::isnan)(1.0L), false);
   BOOST_CHECK_EQUAL((boost::math::isinf)(1.0L), false);
   BOOST_CHECK_EQUAL((boost::math::isnormal)(1.0L), true);
   BOOST_CHECK_EQUAL((boost::math::isfinite)(1.0L), true);
} // BOOST_AUTO_TEST_CASE( test_main )


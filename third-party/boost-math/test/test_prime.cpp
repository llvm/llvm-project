//  (C) Copyright John Maddock 2025.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pch_light.hpp>

#include <boost/math/concepts/real_concept.hpp>
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/special_functions/prime.hpp>
#include <boost/math/tools/test.hpp>

//
// DESCRIPTION:
// ~~~~~~~~~~~~
//
// This file tests the function prime.  
//


BOOST_AUTO_TEST_CASE( test_main )
{
   BOOST_CHECK_EQUAL(boost::math::prime(0), 2);
   BOOST_CHECK_EQUAL(boost::math::prime(49), 229);
   BOOST_CHECK_EQUAL(boost::math::prime(99), 541);
   BOOST_CHECK_EQUAL(boost::math::prime(499), 3571);
   BOOST_CHECK_EQUAL(boost::math::prime(4999), 48611);

   BOOST_CHECK_THROW(boost::math::prime(100000), std::domain_error);

#ifdef BOOST_MATH_HAVE_CONSTEXPR_TABLES
   static_assert(boost::math::prime(0) == 2, "ooops");
   static_assert(boost::math::prime(49) == 229, "ooops");
   static_assert(boost::math::prime(99) == 541, "ooops");
   static_assert(boost::math::prime(499) == 3571, "ooops");
   static_assert(boost::math::prime(4999) == 48611, "ooops");
#endif
}



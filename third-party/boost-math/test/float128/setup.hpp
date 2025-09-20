//  Copyright 2014 John Maddock. Distributed under the Boost
//  Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_

#ifndef BOOST_MP_MATH_SETUP_HPP
#define BOOST_MP_MATH_SETUP_HPP

#ifdef _MSC_VER
#  define _SCL_SECURE_NO_WARNINGS
#endif

#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error

#include <boost/cstdfloat.hpp>

#define ALL_TESTS    test(boost::floatmax_t(0), "boost::floatmax_t");
typedef boost::floatmax_t test_type_1;
   
static_assert(std::numeric_limits<boost::floatmax_t>::digits == 113, "These tests should only be run for 128-bit floating point types.");

#ifndef BOOST_MATH_TEST_TYPE
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#endif

#endif


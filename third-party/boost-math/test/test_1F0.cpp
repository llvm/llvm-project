//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "test_1F0.hpp"

#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>


BOOST_AUTO_TEST_CASE( test_main )
{
#if !defined(TEST) || (TEST == 1)
#ifndef BOOST_MATH_BUGGY_LARGE_FLOAT_CONSTANTS
   test_spots(0.0F);
#endif
   test_spots(0.0);
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   test_spots(0.0L);
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
   test_spots(boost::math::concepts::real_concept(0.1));
#endif
#endif
#endif
#if !defined(TEST) || (TEST == 2)
   #if defined(BOOST_MATH_RUN_MP_TESTS)
   test_spots(boost::multiprecision::cpp_bin_float_quad());
   #endif
#endif
#if (!defined(TEST) || (TEST == 3))
   #if defined(BOOST_MATH_RUN_MP_TESTS)
   test_spots(boost::multiprecision::cpp_dec_float_50());
   #endif
#endif
}




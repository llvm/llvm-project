//  (C) Copyright John Maddock 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//
// This test verifies that the limits we use in the numerical approximations to erf/erc
// are indeed correct.  The values tested must of course match the values used in erf.hpp.
// See https://github.com/boostorg/math/issues/710
//

#define BOOST_TEST_MODULE erf_limits

#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <boost/test/included/unit_test.hpp>

#include <cfloat>

BOOST_AUTO_TEST_CASE(limits_53_digits)
{
   double arg = 5.93f;

   double t = (double)boost::math::erf(boost::multiprecision::cpp_bin_float_50(arg));
   BOOST_CHECK_EQUAL(t, 1.0);

   arg = 5.9;
   BOOST_CHECK_LT(boost::math::erf(arg), 1.0);

   arg = 28;
   t = (double)boost::math::erfc(boost::multiprecision::cpp_bin_float_50(arg));
   BOOST_CHECK_EQUAL(t, 0.0);

   arg = 27.2;

   BOOST_CHECK_GT(boost::math::erfc(arg), 0.0);
}

BOOST_AUTO_TEST_CASE(limits_64_digits)
{
   float arg = 6.6f;

   boost::multiprecision::cpp_bin_float_double_extended t = (boost::multiprecision::cpp_bin_float_double_extended)boost::math::erf(boost::multiprecision::cpp_bin_float_50(arg));
   BOOST_CHECK_EQUAL(t, 1.0);

#if (LDBL_MANT_DIG == 64) && !defined(BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS)
   arg = 6.5;
   BOOST_CHECK_LT(boost::math::erf(static_cast<long double>(arg)), 1.0L);
#endif
   arg = 110;
   t = (boost::multiprecision::cpp_bin_float_double_extended)boost::math::erfc(boost::multiprecision::cpp_bin_float_50(arg));
   BOOST_CHECK_EQUAL(t, 0.0);

#if (LDBL_MANT_DIG == 64) && !defined(BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS)
   arg = 106.5;
   BOOST_CHECK_GT(boost::math::erfc(static_cast<long double>(arg)), 0.0L);
#endif
}

BOOST_AUTO_TEST_CASE(limits_113_digits)
{
   //
   // This limit is not actually used in the code, logged here for future reference...
   //
   float arg = 8.8f;

   boost::multiprecision::cpp_bin_float_quad t = (boost::multiprecision::cpp_bin_float_quad)boost::math::erf(boost::multiprecision::cpp_bin_float_50(arg));
   BOOST_CHECK_EQUAL(t, 1.0);

   arg = 110;
   t = (boost::multiprecision::cpp_bin_float_quad)boost::math::erfc(boost::multiprecision::cpp_bin_float_50(arg));
   BOOST_CHECK_EQUAL(t, 0.0);
}

//  (C) Copyright John Maddock 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_MATH_PROMOTE_DOUBLE_POLICY false

#include "math_unit_test.hpp"
#include <cfenv>
#include <iostream>
#include <boost/math/special_functions/gamma.hpp>


int main()
{
   CHECK_EQUAL(boost::math::tgamma(-200.5), 0.0); // triggers internal exception handling
   CHECK_EQUAL(boost::math::gamma_p(500.125, 1e-50), 0.0); // triggers internal exception handling

   // Lines that can only be hit when promotion to 80-bit reals is turned off
   CHECK_ULP_CLOSE(boost::math::tgamma(44.0, 0.000001), 6.04152630633738356373551320685139975072645120000000000000000e52, 10);
   CHECK_ULP_CLOSE(boost::math::gamma_p(1.0001, boost::math::gamma_p_inv(1.0001, 1e-200)), 1e-200, 10);
   return boost::math::test::report_errors();
}

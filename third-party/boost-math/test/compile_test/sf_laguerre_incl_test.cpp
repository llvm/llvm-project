//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/special_functions/laguerre.hpp>
// #includes all the files that it needs to.
//
#include <boost/math/special_functions/laguerre.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
   check_result<float>(boost::math::laguerre<float>(u, f));
   check_result<double>(boost::math::laguerre<double>(u, d));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::laguerre<long double>(u, l));
#endif

   typedef boost::math::policies::policy<> def_pol;
   def_pol p;

   check_result<float>(boost::math::laguerre<float, def_pol>(u, u, f, p));
   check_result<double>(boost::math::laguerre<double, def_pol>(u, u, d, p));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   check_result<long double>(boost::math::laguerre<long double, def_pol>(u, u, l, p));
#endif
}

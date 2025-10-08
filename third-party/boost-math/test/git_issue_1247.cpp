//  (C) Copyright John Maddock 2025.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// See also: https://godbolt.org/z/nhMsKb8Yr

#include <boost/math/special_functions/beta.hpp>
#include <boost/core/lightweight_test.hpp>

int main()
{
   float a = 1e-20;
   float b = 1e-21;
   float z = 0.5;
   float p = boost::math::ibeta(a, b, z, boost::math::policies::make_policy(boost::math::policies::promote_float<false>()));

   BOOST_TEST(p != 0);

   return boost::report_errors();
}

//  Copyright John Maddock 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_MATH_PROMOTE_DOUBLE_POLICY false

#include <iostream>
#include <boost/math/distributions/skew_normal.hpp>

template <class T>
void test()
{
   boost::math::skew_normal_distribution<T> skew(573.39724735636185, 77.0, 4.0);
   const T x = boost::math::quantile(skew, 0.00285612015554148);
   const T y = boost::math::quantile(skew, 0.00285612015554149);
   const T z = boost::math::quantile(skew, 0.00285612015554150);

   BOOST_MATH_ASSERT(boost::math::isnormal(x));
   BOOST_MATH_ASSERT(boost::math::isnormal(y));
   BOOST_MATH_ASSERT(boost::math::isnormal(z));

   BOOST_MATH_ASSERT(x <= y);
   BOOST_MATH_ASSERT(y <= z);
}


int main()
{
   test<float>();
   test<double>();
   test<long double>();

   return 0;
}

//  Copyright Nick Thompson 2017.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef BOOST_MATH_STANDALONE

#include <boost/math/concepts/std_real_concept.hpp>
#include <boost/math/quadrature/trapezoidal.hpp>

boost::math::concepts::std_real_concept func(boost::math::concepts::std_real_concept x)
{
   return x;
}

void compile_and_link_test()
{
   boost::math::concepts::std_real_concept a = 0;
   boost::math::concepts::std_real_concept b = 1;
   boost::math::quadrature::trapezoidal(func, a, b);
}

#endif

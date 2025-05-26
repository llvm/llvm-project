//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef BOOST_MATH_STANDALONE

#include <boost/math/concepts/std_real_concept.hpp>
#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>

void compile_and_link_test()
{
   boost::math::concepts::std_real_concept data[] = { 1, 2, 3 };
   boost::math::interpolators::cardinal_cubic_b_spline<boost::math::concepts::std_real_concept> s(data, 3, 2, 1), s2;
   s(1.0);
   s.prime(1.0);
}

#endif

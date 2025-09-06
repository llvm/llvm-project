//  Copyright John Maddock 2017.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef BOOST_MATH_STANDALONE

#include <boost/math/concepts/std_real_concept.hpp>
#include <boost/math/interpolators/barycentric_rational.hpp>

void compile_and_link_test()
{
   boost::math::concepts::std_real_concept x[] = { 1, 2, 3 };
   boost::math::concepts::std_real_concept y[] = { 13, 15, 17 };
   boost::math::interpolators::barycentric_rational<boost::math::concepts::std_real_concept> s(x, y, 3, 3);
   s(1.0);
}

#endif

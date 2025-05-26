/*
 * Copyright Nick Thompson, 2017
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#if !defined(BOOST_MATH_STANDALONE) && (!defined(_MSC_VER) || (_MSC_VER >= 1900))

#include <boost/math/concepts/std_real_concept.hpp>
#include <boost/math/quadrature/naive_monte_carlo.hpp>

using boost::math::concepts::std_real_concept;
using boost::math::quadrature::naive_monte_carlo;

void compile_and_link_test()
{
   auto g = [&](std::vector<std_real_concept> const &)
   {
     return 1.873;
   };
   std::vector<std::pair<std_real_concept, std_real_concept>> bounds{{0, 1}, {0, 1}, {0, 1}};
   naive_monte_carlo<std_real_concept, decltype(g)> mc(g, bounds, 1.0);

   auto task = mc.integrate();
   task.get();
}

#else
void compile_and_link_test()
{
}
#endif

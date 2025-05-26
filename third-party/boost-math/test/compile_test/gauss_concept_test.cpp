//  Copyright Nick Thompson 2017.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Test requires the use of lexical cast in tools/big_constant.hpp
#ifndef BOOST_MATH_STANDALONE

#include <boost/math/concepts/std_real_concept.hpp>
#include <boost/math/quadrature/gauss.hpp>

void compile_and_link_test()
{
    boost::math::concepts::std_real_concept a = 0;
    boost::math::concepts::std_real_concept b = 1;
    auto f = [](boost::math::concepts::std_real_concept x) { return x; };
    boost::math::quadrature::gauss<boost::math::concepts::std_real_concept, 7> integrator;
    integrator.integrate(f, a, b);
}

#endif

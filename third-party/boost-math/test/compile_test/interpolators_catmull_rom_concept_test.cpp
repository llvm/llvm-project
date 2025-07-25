//  Copyright Nick Thompson 2018.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <boost/math/tools/config.hpp>
#include <boost/math/concepts/std_real_concept.hpp>
#include <boost/math/interpolators/catmull_rom.hpp>

void compile_and_link_test()
{
    #ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
    std::vector<boost::math::concepts::std_real_concept> p0{0.1, 0.2, 0.3};
    std::vector<boost::math::concepts::std_real_concept> p1{0.2, 0.3, 0.4};
    std::vector<boost::math::concepts::std_real_concept> p2{0.3, 0.4, 0.5};
    std::vector<boost::math::concepts::std_real_concept> p3{0.4, 0.5, 0.6};
    std::vector<boost::math::concepts::std_real_concept> p4{0.5, 0.6, 0.7};
    std::vector<boost::math::concepts::std_real_concept> p5{0.6, 0.7, 0.8};
    std::vector<std::vector<boost::math::concepts::std_real_concept>> v{p0, p1, p2, p3, p4, p5};
    boost::math::catmull_rom<std::vector<boost::math::concepts::std_real_concept>> cat(std::move(v));
    cat(0.0);
    cat.prime(0.0);
    #endif
}

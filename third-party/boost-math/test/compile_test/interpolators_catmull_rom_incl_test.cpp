//  Copyright Nick Thompson 2018.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/interpolators/catmull_rom.hpp>
// #includes all the files that it needs to.
//
#include <boost/math/interpolators/catmull_rom.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
    std::vector<double> p0{0.1, 0.2, 0.3};
    std::vector<double> p1{0.2, 0.3, 0.4};
    std::vector<double> p2{0.3, 0.4, 0.5};
    std::vector<double> p3{0.4, 0.5, 0.6};
    std::vector<double> p4{0.5, 0.6, 0.7};
    std::vector<double> p5{0.6, 0.7, 0.8};
    std::vector<std::vector<double>> v{p0, p1, p2, p3, p4, p5};
    boost::math::catmull_rom<std::vector<double>> cat(std::move(v));
    check_result<std::vector<double>>(cat(0.0));
    check_result<std::vector<double>>(cat.prime(0.0));
}

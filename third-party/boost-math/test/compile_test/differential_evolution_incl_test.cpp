//  Copyright Nick Thompson 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header
// #includes all the files that it needs to.
//
#include <boost/math/optimization/differential_evolution.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
    auto f = [](std::vector<double> const & v) { return v[0]*v[0]; };
    boost::math::optimization::differential_evolution_parameters<std::vector<double>> params;
    std::mt19937_64 gen(12345);
    auto v = boost::math::optimization::differential_evolution(f, params, gen);
}

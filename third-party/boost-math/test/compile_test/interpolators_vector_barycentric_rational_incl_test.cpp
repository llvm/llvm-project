//  Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// A sanity check that this file
// #includes all the files that it needs to.
//
#include <boost/math/interpolators/vector_barycentric_rational.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"
//
// This test includes <vector> because many of the interpolators are not compatible with pointers/c-style arrays
//
#include <vector>

void compile_and_link_test()
{
   std::vector<double> time = { 1, 2, 3 };
   std::vector<std::vector<double>> space = {{1}, {2}, {3}};
   boost::math::interpolators::vector_barycentric_rational<decltype(time), decltype(space)> s(std::move(time), std::move(space));

   check_result<std::vector<double>>(s(1.0));
}

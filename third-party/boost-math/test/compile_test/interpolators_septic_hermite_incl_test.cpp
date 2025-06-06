//  Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// A sanity check that this file
// #includes all the files that it needs to.
//
#include <boost/math/interpolators/septic_hermite.hpp>
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
   std::vector<double> data = { 1, 2, 3 };
   std::vector<double> data_1st_deriv = {1, 2, 3};
   std::vector<double> data_2nd_deriv = {1, 2, 3};
   std::vector<double> data_3rd_deriv = {1, 2, 3};
   boost::math::interpolators::cardinal_septic_hermite<std::vector<double>> s(std::move(data), std::move(data_1st_deriv), 
                                                                              std::move(data_2nd_deriv), std::move(data_3rd_deriv), 1.0, 1.0);
   check_result<double>(s(1.0));
}

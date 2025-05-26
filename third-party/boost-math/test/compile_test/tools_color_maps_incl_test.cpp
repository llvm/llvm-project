//  (C) Copyright Matt Borland 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/tools/color_maps.hpp>
#include "test_compile_result.hpp"

void compile_and_link_test()
{
    check_result<float>(boost::math::tools::black_body<float>(0.5f)[0]);
    check_result<float>(boost::math::tools::extended_kindlmann<float>(0.5f)[0]);
    check_result<double>(boost::math::tools::inferno<double>(0.5)[0]);
    check_result<double>(boost::math::tools::kindlmann<double>(0.5)[0]);
    check_result<float>(boost::math::tools::plasma<float>(0.5f)[0]);
    check_result<float>(boost::math::tools::smooth_cool_warm<float>(0.5f)[0]);
    check_result<float>(boost::math::tools::viridis<float>(0.5f)[0]);
}

//  Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header
// #includes all the files that it needs to.
//
#include <boost/math/quadrature/wavelet_transforms.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
    auto f = [](double x) { return x; };
    auto psi = boost::math::daubechies_wavelet<double, 1>();
    auto Wf = boost::math::quadrature::daubechies_wavelet_transform(f, psi);
    check_result<double>(Wf(0.0, 0.0));
}

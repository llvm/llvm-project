//  Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header
// #includes all the files that it needs to.
//
#include <boost/math/quadrature/ooura_fourier_integrals.hpp>
#ifndef _MSC_VER
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
    auto f = [](double x) { return x; };
    boost::math::quadrature::ooura_fourier_sin<double> sin_integrator;
    boost::math::quadrature::ooura_fourier_cos<double> cos_integrator;
    check_result<std::pair<double, double>>(sin_integrator.integrate(f, 1.0));
    check_result<std::pair<double, double>>(cos_integrator.integrate(f, 1.0));
}
#endif

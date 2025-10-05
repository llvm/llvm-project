//  (C) Copyright Nick Thompson 2020.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Deliberately contains some unicode characters:
// 
// boost-no-inspect
//
#include <iostream>
#include <boost/math/constants/constants.hpp>
#include <boost/math/tools/simple_continued_fraction.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>

using boost::math::constants::root_two;
using boost::math::constants::phi;
using boost::math::constants::pi;
using boost::math::constants::e;
using boost::math::constants::zeta_three;
using boost::math::tools::simple_continued_fraction;

int main()
{
    using Real = boost::multiprecision::cpp_bin_float_100;
    auto phi_cfrac = simple_continued_fraction(phi<Real>());
    std::cout << "φ ≈ " << phi_cfrac << "\n\n";

    auto pi_cfrac = simple_continued_fraction(pi<Real>());
    std::cout << "π ≈ " << pi_cfrac << "\n";
    std::cout << "Known: [3; 7, 15, 1, 292, 1, 1, 1, 2, 1, 3, 1, 14, 2, 1, 1, 2, 2, 2, 2, 1, 84, 2, 1, 1, 15, 3, 13, 1, 4, 2, 6, 6, 99, 1, 2, 2, 6, 3, 5, 1, 1, 6, 8, 1, 7, 1, 2, 3, 7, 1, 2, 1, 1, 12, 1, 1, 1, 3, 1, 1, 8, 1, 1, 2, 1, 6, 1, 1, 5, 2, 2, 3, 1, 2, 4, 4, 16, 1, 161, 45, 1, 22, 1, 2, 2, 1, 4, 1, 2, 24, 1, 2, 1, 3, 1, 2, 1, ...]\n\n";

    auto rt_cfrac = simple_continued_fraction(root_two<Real>());
    std::cout << "√2 ≈ " << rt_cfrac << "\n\n";

    auto e_cfrac = simple_continued_fraction(e<Real>());
    std::cout << "e ≈ " << e_cfrac << "\n";

    // Correctness can be checked in Mathematica via: ContinuedFraction[Zeta[3], 500]
    auto z_cfrac = simple_continued_fraction(zeta_three<Real>());
    std::cout << "ζ(3) ≈ " << z_cfrac << "\n";
}

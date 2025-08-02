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
#include <boost/math/tools/centered_continued_fraction.hpp>
#include <boost/multiprecision/mpfr.hpp>

using boost::math::constants::root_two;
using boost::math::constants::phi;
using boost::math::constants::pi;
using boost::math::constants::e;
using boost::math::constants::zeta_three;
using boost::math::tools::centered_continued_fraction;
using boost::multiprecision::mpfr_float;

int main()
{
    using Real = mpfr_float;
    int p = 10000;
    mpfr_float::default_precision(p);
    auto phi_cfrac = centered_continued_fraction(phi<Real>());
    std::cout << "φ ≈ " << phi_cfrac << "\n";
    std::cout << "Khinchin mean: " << std::setprecision(10) << phi_cfrac.khinchin_geometric_mean() << "\n\n\n";

    auto pi_cfrac = centered_continued_fraction(pi<Real>());
    std::cout << "π ≈ " << pi_cfrac << "\n";
    std::cout << "Khinchin mean: " << std::setprecision(10) << pi_cfrac.khinchin_geometric_mean() << "\n\n\n";

    auto rt_cfrac = centered_continued_fraction(root_two<Real>());
    std::cout << "√2 ≈ " << rt_cfrac << "\n";
    std::cout << "Khinchin mean: " << std::setprecision(10) <<  rt_cfrac.khinchin_geometric_mean() << "\n\n\n";

    auto e_cfrac = centered_continued_fraction(e<Real>());
    std::cout << "e ≈ " << e_cfrac << "\n";
    std::cout << "Khinchin mean: " << std::setprecision(10) <<  e_cfrac.khinchin_geometric_mean() << "\n\n\n";

    auto z_cfrac = centered_continued_fraction(zeta_three<Real>());
    std::cout << "ζ(3) ≈ " << z_cfrac << "\n";
    std::cout << "Khinchin mean: " << std::setprecision(10) <<  z_cfrac.khinchin_geometric_mean() << "\n\n\n";


    // http://jeremiebourdon.free.fr/data/Khintchine.pdf
    std::cout << "The expected Khinchin mean for a random centered continued fraction is 5.45451724454\n";
}

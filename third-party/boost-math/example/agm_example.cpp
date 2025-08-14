//  (C) Copyright Nick Thompson 2020.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#include <cmath>
#include <iostream>
#include <iomanip>
#include <boost/math/tools/agm.hpp>
#include <boost/math/constants/constants.hpp>

#ifndef BOOST_MATH_STANDALONE
#include <boost/multiprecision/cpp_bin_float.hpp>
#endif

// This example computes the lemniscate constant to high precision using the agm:
using boost::math::tools::agm;
using boost::math::constants::pi;

int main() {
    using std::sqrt;
    
    #ifndef BOOST_MATH_STANDALONE
    using Real = boost::multiprecision::cpp_bin_float_100;
    #else
    using Real = long double;
    #endif

    Real G = agm(sqrt(Real(2)), Real(1));
    std::cout << std::setprecision(std::numeric_limits<Real>::max_digits10);
    std::cout << " Gauss's lemniscate constant = " << pi<Real>()/G << "\n";
    std::cout << "Expected lemniscate constant = " << "2.62205755429211981046483958989111941368275495143162316281682170380079058707041425023029553296142909344613\n";
}

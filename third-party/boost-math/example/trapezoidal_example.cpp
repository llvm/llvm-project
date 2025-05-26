/*
 * Copyright Nick Thompson, 2017
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 *
 * This example shows to to numerically integrate a periodic function using the adaptive_trapezoidal routine provided by boost.
 */

#include <iostream>
#include <cmath>
#include <limits>
#include <boost/math/quadrature/trapezoidal.hpp>

int main()
{
    using boost::math::constants::two_pi;
    using boost::math::constants::third;
    using boost::math::quadrature::trapezoidal;
    // This function has an analytic form for its integral over a period: 2pi/3.
    auto f = [](double x) { return 1/(5 - 4*cos(x)); };

    double Q = trapezoidal(f, (double) 0, two_pi<double>());

    std::cout << std::setprecision(std::numeric_limits<double>::digits10);
    std::cout << "The adaptive trapezoidal rule gives the integral of our function as " << Q << "\n";
    std::cout << "The exact result is                                                 " << two_pi<double>()*third<double>() << "\n";

}

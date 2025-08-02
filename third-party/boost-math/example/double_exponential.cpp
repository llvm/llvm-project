// Copyright Nick Thompson, 2017
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <cmath>
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/quadrature/sinh_sinh.hpp>
#include <boost/math/quadrature/exp_sinh.hpp>

using boost::math::quadrature::tanh_sinh;
using boost::math::quadrature::sinh_sinh;
using boost::math::quadrature::exp_sinh;
using boost::math::constants::pi;
using boost::math::constants::half_pi;
using boost::math::constants::half;
using boost::math::constants::third;
using boost::math::constants::root_pi;
using std::log;
using std::cos;
using std::cosh;
using std::exp;
using std::sqrt;

int main()
{
    std::cout << std::setprecision(std::numeric_limits<double>::digits10);
    double tol = sqrt(std::numeric_limits<double>::epsilon());
    // For an integral over a finite domain, use tanh_sinh:
    tanh_sinh<double> tanh_integrator(tol, 10);
    auto f1 = [](double x) { return log(x)*log(1-x); };
    double Q = tanh_integrator.integrate(f1, (double) 0, (double) 1);
    double Q_expected = 2 - pi<double>()*pi<double>()*half<double>()*third<double>();

    std::cout << "tanh_sinh quadrature of log(x)log(1-x) gives " << Q << std::endl;
    std::cout << "The exact integral is                        " << Q_expected << std::endl;

    // For an integral over the entire real line, use sinh-sinh quadrature:
    sinh_sinh<double> sinh_integrator(10);
    auto f2 = [](double t) { return cos(t)/cosh(t);};
    Q = sinh_integrator.integrate(f2);
    Q_expected = pi<double>()/cosh(half_pi<double>());
    std::cout << "sinh_sinh quadrature of cos(x)/cosh(x) gives " << Q << std::endl;
    std::cout << "The exact integral is                        " << Q_expected << std::endl;

    // For half-infinite intervals, use exp-sinh.
    // Endpoint singularities are handled well:
    exp_sinh<double> exp_integrator(10);
    auto f3 = [](double t) { return exp(-t)/sqrt(t); };
    Q = exp_integrator.integrate(f3, 0, std::numeric_limits<double>::infinity());
    Q_expected = root_pi<double>();
    std::cout << "exp_sinh quadrature of exp(-t)/sqrt(t) gives " << Q << std::endl;
    std::cout << "The exact integral is                        " << Q_expected << std::endl;



}

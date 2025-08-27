
// Copyright Nick Thompson, 2017

// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or
// copy at http://www.boost.org/LICENSE_1_0.txt).

#include <iostream>
#include <limits>
#include <map>

//[barycentric_rational_example2

/*`This further example shows how to use the iterator based constructor, and then uses the
function object in our root finding algorithms to locate the points where the potential
achieves a specific value.
*/

#include <boost/math/interpolators/barycentric_rational.hpp>
#include <boost/range/adaptors.hpp>
#include <boost/math/tools/roots.hpp>

int main()
{
    // The lithium potential is given in Kohn's paper, Table I.
    // (We could equally easily use an unordered_map, a list of tuples or pairs, or a 2-dimensional array).
    std::map<double, double> r;

    r[0.02] = 5.727;
    r[0.04] = 5.544;
    r[0.06] = 5.450;
    r[0.08] = 5.351;
    r[0.10] = 5.253;
    r[0.12] = 5.157;
    r[0.14] = 5.058;
    r[0.16] = 4.960;
    r[0.18] = 4.862;
    r[0.20] = 4.762;
    r[0.24] = 4.563;
    r[0.28] = 4.360;
    r[0.32] = 4.1584;
    r[0.36] = 3.9463;
    r[0.40] = 3.7360;
    r[0.44] = 3.5429;
    r[0.48] = 3.3797;
    r[0.52] = 3.2417;
    r[0.56] = 3.1209;
    r[0.60] = 3.0138;
    r[0.68] = 2.8342;
    r[0.76] = 2.6881;
    r[0.84] = 2.5662;
    r[0.92] = 2.4242;
    r[1.00] = 2.3766;
    r[1.08] = 2.3058;
    r[1.16] = 2.2458;
    r[1.24] = 2.2035;
    r[1.32] = 2.1661;
    r[1.40] = 2.1350;
    r[1.48] = 2.1090;
    r[1.64] = 2.0697;
    r[1.80] = 2.0466;
    r[1.96] = 2.0325;
    r[2.12] = 2.0288;
    r[2.28] = 2.0292;
    r[2.44] = 2.0228;
    r[2.60] = 2.0124;
    r[2.76] = 2.0065;
    r[2.92] = 2.0031;
    r[3.08] = 2.0015;
    r[3.24] = 2.0008;
    r[3.40] = 2.0004;
    r[3.56] = 2.0002;
    r[3.72] = 2.0001;

    // Let's discover the abscissa that will generate a potential of exactly 3.0,
    // start by creating 2 ranges for the x and y values:
    auto x_range = boost::adaptors::keys(r);
    auto y_range = boost::adaptors::values(r);
    boost::math::interpolators::barycentric_rational<double> b(x_range.begin(), x_range.end(), y_range.begin());
    //
    // We'll use a lambda expression to provide the functor to our root finder, since we want
    // the abscissa value that yields 3, not zero.  We pass the functor b by value to the
    // lambda expression since barycentric_rational is trivial to copy.
    // Here we're using simple bisection to find the root:
    std::uintmax_t iterations = (std::numeric_limits<std::uintmax_t>::max)();
    double abscissa_3 = boost::math::tools::bisect([=](double x) { return b(x) - 3; }, 0.44, 1.24, boost::math::tools::eps_tolerance<double>(), iterations).first;
    std::cout << "Abscissa value that yields a potential of 3 = " << abscissa_3 << std::endl;
    std::cout << "Root was found in " << iterations << " iterations." << std::endl;
    //
    // However, we have a more efficient root finding algorithm than simple bisection:
    iterations = (std::numeric_limits<std::uintmax_t>::max)();
    abscissa_3 = boost::math::tools::bracket_and_solve_root([=](double x) { return b(x) - 3; }, 0.6, 1.2, false, boost::math::tools::eps_tolerance<double>(), iterations).first;
    std::cout << "Abscissa value that yields a potential of 3 = " << abscissa_3 << std::endl;
    std::cout << "Root was found in " << iterations << " iterations." << std::endl;
}
//] [/barycentric_rational_example2]


//[barycentric_rational_example2_out
/*` Program output is:
[pre
Abscissa value that yields a potential of 3 = 0.604728
Root was found in 54 iterations.
Abscissa value that yields a potential of 3 = 0.604728
Root was found in 10 iterations.
]
*/
//]

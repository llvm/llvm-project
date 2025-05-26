// Copyright Nick Thompson, 2017

// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or
// copy at http://www.boost.org/LICENSE_1_0.txt).

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <boost/math/interpolators/catmull_rom.hpp>
#include <boost/math/constants/constants.hpp>

using std::sin;
using std::cos;
using boost::math::catmull_rom;

int main()
{
    std::cout << "This shows how to use Boost's Catmull-Rom spline to create an Archimedean spiral.\n";

    // The Archimedean spiral is given by r = a*theta. We have set a = 1.
    std::vector<std::array<double, 2>> spiral_points(500);
    double theta_max = boost::math::constants::pi<double>();
    for (size_t i = 0; i < spiral_points.size(); ++i)
    {
        double theta = ((double) i/ (double) spiral_points.size())*theta_max;
        spiral_points[i] = {theta*cos(theta), theta*sin(theta)};
    }

    auto archimedean = catmull_rom<std::array<double,2>>(std::move(spiral_points));
    double max_s = archimedean.max_parameter();
    std::cout << "Max s = " << max_s << std::endl;
    for (double s = 0; s < max_s; s += 0.01)
    {
        auto p = archimedean(s);
        double x = p[0];
        double y = p[1];
        double r = sqrt(x*x + y*y);
        double theta = atan2(y/r, x/r);
        std::cout << "r = " << r << ", theta = " << theta << ", r - theta = " << r - theta << std::endl;
    }

    return 0;
}

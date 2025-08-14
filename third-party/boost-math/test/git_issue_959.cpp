// Copyright Matt Borland, 2023
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <array>
#include <vector>
#include <boost/math/interpolators/bezier_polynomial.hpp>

using tControlPoints = std::vector<std::array<double, 3>>;

void interpolateWithPoints(tControlPoints cp)
{
    const auto cpSize = cp.size();
    auto bp = boost::math::interpolators::bezier_polynomial(std::move(cp));

    // Interpolate at t = 0.5:
    std::array<double, 3> point = bp(0.5);
    std::cout << cpSize << " points, t = 0.5:\n";

    for (const auto& c : point) 
    {
        std::cout << "  " << c << "\n";
    }
}


int main(void)
{
    auto cp3 = tControlPoints{{0,0,0}, {1,0,0}, {0,1,0}};
    interpolateWithPoints(cp3);

    auto cp4 = tControlPoints{{0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}};
    interpolateWithPoints(cp4);

    auto cp3b = tControlPoints{{0,0,0}, {1,0,0}, {0,1,0}};
    interpolateWithPoints(cp3b);

    return 0;
}

// Copyright Nicholas Thompson 2017.
// Copyright Paul A. Bristow 2017.
// Copyright John Maddock 2017.

// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or
// copy at http://www.boost.org/LICENSE_1_0.txt).

#include <iostream>
#include <limits>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <iterator>
#include <cmath>
#include <random>
#include <cstdint>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/math/tools/roots.hpp>

//[cubic_b_spline_example

/*`This example demonstrates how to use the cubic b spline interpolator for regularly spaced data.
*/
#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>

int main()
{
    // We begin with an array of samples:
    std::vector<double> v(500);
    // And decide on a stepsize:
    double step = 0.01;

    // Initialize the vector with a function we'd like to interpolate:
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = sin(i*step);
    }
    // We could define an arbitrary start time, but for now we'll just use 0:
    boost::math::interpolators::cardinal_cubic_b_spline<double> spline(v.data(), v.size(), 0 /* start time */, step);

    // Now we can evaluate the spline wherever we please.
    std::mt19937 gen;
    boost::random::uniform_real_distribution<double> abscissa(0, v.size()*step);
    for (size_t i = 0; i < 10; ++i)
    {
        double x = abscissa(gen);
        std::cout << "sin(" << x << ") = " << sin(x) << ", spline interpolation gives " << spline(x) << std::endl;
        std::cout << "cos(" << x << ") = " << cos(x) << ", spline derivative interpolation gives " << spline.prime(x) << std::endl;
    }

    // The next example is less trivial:
    // We will try to figure out when the population of the United States crossed 100 million.
    // Since the census is taken every 10 years, the data is equally spaced, so we can use the cubic b spline.
    // Data taken from https://en.wikipedia.org/wiki/United_States_Census
    // We'll start at the year 1860:
    double t0 = 1860;
    double time_step = 10;
    std::vector<double> population{31443321,  /* 1860 */
                                   39818449,  /* 1870 */
                                   50189209,  /* 1880 */
                                   62947714,  /* 1890 */
                                   76212168,  /* 1900 */
                                   92228496,  /* 1910 */
                                   106021537, /* 1920 */
                                   122775046, /* 1930 */
                                   132164569, /* 1940 */
                                   150697361, /* 1950 */
                                   179323175};/* 1960 */

    // An eyeball estimate indicates that the population crossed 100 million around 1915.
    // Let's see what interpolation says:
    boost::math::interpolators::cardinal_cubic_b_spline<double> p(population.data(), population.size(), t0, time_step);

    // Now create a function which has a zero at p = 100,000,000:
    auto f = [=](double t){ return p(t) - 100000000; };

    // Boost includes a bisection algorithm, which is robust, though not as fast as some others
    // we provide, but let's try that first.  We need a termination condition for it, which
    // takes the two endpoints of the range and returns either true (stop) or false (keep going),
    // we could use a predefined one such as boost::math::tools::eps_tolerance<double>, but that
    // won't stop until we have full double precision which is overkill, since we just need the
    // endpoint to yield the same month.  While we're at it, we'll keep track of the number of
    // iterations required too, though this is strictly optional:

    auto termination = [](double left, double right)
    {
       double left_month = std::round((left - std::floor(left)) * 12 + 1);
       double right_month = std::round((right - std::floor(right)) * 12 + 1);
       return (left_month == right_month) && (std::floor(left) == std::floor(right));
    };
    std::uintmax_t iterations = 1000;
    auto result =  boost::math::tools::bisect(f, 1910.0, 1920.0, termination, iterations);
    auto time = result.first;  // termination condition ensures that both endpoints yield the same result
    auto month = std::round((time - std::floor(time))*12  + 1);
    auto year = std::floor(time);
    std::cout << "The population of the United States surpassed 100 million on the ";
    std::cout << month << "th month of " << year << std::endl;
    std::cout << "Found in " << iterations << " iterations" << std::endl;

    // Since the cubic B spline offers the first derivative, we could equally have used Newton iterations,
    // this takes "number of bits correct" as a termination condition - 20 should be plenty for what we need,
    // and once again, we track how many iterations are taken:

    auto f_n = [=](double t) { return std::make_pair(p(t) - 100000000, p.prime(t)); };
    iterations = 1000;
    time = boost::math::tools::newton_raphson_iterate(f_n, 1910.0, 1900.0, 2000.0, 20, iterations);
    month = std::round((time - std::floor(time))*12  + 1);
    year = std::floor(time);
    std::cout << "The population of the United States surpassed 100 million on the ";
    std::cout << month << "th month of " << year << std::endl;
    std::cout << "Found in " << iterations << " iterations" << std::endl;

}

//] [/cubic_b_spline_example]

//[cubic_b_spline_example_out
/*` Program output is:
[pre
sin(4.07362) = -0.802829, spline interpolation gives - 0.802829
cos(4.07362) = -0.596209, spline derivative interpolation gives - 0.596209
sin(0.677385) = 0.626758, spline interpolation gives 0.626758
cos(0.677385) = 0.779214, spline derivative interpolation gives 0.779214
sin(4.52896) = -0.983224, spline interpolation gives - 0.983224
cos(4.52896) = -0.182402, spline derivative interpolation gives - 0.182402
sin(4.17504) = -0.85907, spline interpolation gives - 0.85907
cos(4.17504) = -0.511858, spline derivative interpolation gives - 0.511858
sin(0.634934) = 0.593124, spline interpolation gives 0.593124
cos(0.634934) = 0.805111, spline derivative interpolation gives 0.805111
sin(4.84434) = -0.991307, spline interpolation gives - 0.991307
cos(4.84434) = 0.131567, spline derivative interpolation gives 0.131567
sin(4.56688) = -0.989432, spline interpolation gives - 0.989432
cos(4.56688) = -0.144997, spline derivative interpolation gives - 0.144997
sin(1.10517) = 0.893541, spline interpolation gives 0.893541
cos(1.10517) = 0.448982, spline derivative interpolation gives 0.448982
sin(3.1618) = -0.0202022, spline interpolation gives - 0.0202022
cos(3.1618) = -0.999796, spline derivative interpolation gives - 0.999796
sin(1.54084) = 0.999551, spline interpolation gives 0.999551
cos(1.54084) = 0.0299566, spline derivative interpolation gives 0.0299566
The population of the United States surpassed 100 million on the 11th month of 1915
Found in 12 iterations
The population of the United States surpassed 100 million on the 11th month of 1915
Found in 3 iterations
]
*/
//]

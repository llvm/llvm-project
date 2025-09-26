
// Copyright Nick Thompson, 2017

// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or
// copy at http://www.boost.org/LICENSE_1_0.txt).

#include <iostream>
#include <limits>
#include <vector>

//[barycentric_rational_example

/*`
This example shows how to use barycentric rational interpolation, using Walter Kohn's classic paper
"Solution of the Schrodinger Equation in Periodic Lattices with an Application to Metallic Lithium"
In this paper, Kohn needs to repeatedly solve an ODE (the radial Schrodinger equation) given a potential
which is only known at non-equally samples data.

If he'd only had the barycentric rational interpolant of Boost.Math!

References: Kohn, W., and N. Rostoker. "Solution of the Schrodinger equation in periodic lattices with an application to metallic lithium." Physical Review 94.5 (1954): 1111.
*/

#include <boost/math/interpolators/barycentric_rational.hpp>

int main()
{
    // The lithium potential is given in Kohn's paper, Table I:
    std::vector<double> r(45);
    std::vector<double> mrV(45);

    // We'll skip the code for filling the above vectors with data for now...
    //<-

    r[0] = 0.02; mrV[0] = 5.727;
    r[1] = 0.04, mrV[1] = 5.544;
    r[2] = 0.06, mrV[2] = 5.450;
    r[3] = 0.08, mrV[3] = 5.351;
    r[4] = 0.10, mrV[4] = 5.253;
    r[5] = 0.12, mrV[5] = 5.157;
    r[6] = 0.14, mrV[6] = 5.058;
    r[7] = 0.16, mrV[7] = 4.960;
    r[8] = 0.18, mrV[8] = 4.862;
    r[9] = 0.20, mrV[9] = 4.762;
    r[10] = 0.24, mrV[10] = 4.563;
    r[11] = 0.28, mrV[11] = 4.360;
    r[12] = 0.32, mrV[12] = 4.1584;
    r[13] = 0.36, mrV[13] = 3.9463;
    r[14] = 0.40, mrV[14] = 3.7360;
    r[15] = 0.44, mrV[15] = 3.5429;
    r[16] = 0.48, mrV[16] = 3.3797;
    r[17] = 0.52, mrV[17] = 3.2417;
    r[18] = 0.56, mrV[18] = 3.1209;
    r[19] = 0.60, mrV[19] = 3.0138;
    r[20] = 0.68, mrV[20] = 2.8342;
    r[21] = 0.76, mrV[21] = 2.6881;
    r[22] = 0.84, mrV[22] = 2.5662;
    r[23] = 0.92, mrV[23] = 2.4242;
    r[24] = 1.00, mrV[24] = 2.3766;
    r[25] = 1.08, mrV[25] = 2.3058;
    r[26] = 1.16, mrV[26] = 2.2458;
    r[27] = 1.24, mrV[27] = 2.2035;
    r[28] = 1.32, mrV[28] = 2.1661;
    r[29] = 1.40, mrV[29] = 2.1350;
    r[30] = 1.48, mrV[30] = 2.1090;
    r[31] = 1.64, mrV[31] = 2.0697;
    r[32] = 1.80, mrV[32] = 2.0466;
    r[33] = 1.96, mrV[33] = 2.0325;
    r[34] = 2.12, mrV[34] = 2.0288;
    r[35] = 2.28, mrV[35] = 2.0292;
    r[36] = 2.44, mrV[36] = 2.0228;
    r[37] = 2.60, mrV[37] = 2.0124;
    r[38] = 2.76, mrV[38] = 2.0065;
    r[39] = 2.92, mrV[39] = 2.0031;
    r[40] = 3.08, mrV[40] = 2.0015;
    r[41] = 3.24, mrV[41] = 2.0008;
    r[42] = 3.40, mrV[42] = 2.0004;
    r[43] = 3.56, mrV[43] = 2.0002;
    r[44] = 3.72, mrV[44] = 2.0001;
    //->

    // Now we want to interpolate this potential at any r:
    boost::math::interpolators::barycentric_rational<double> b(r.data(), mrV.data(), r.size());

    for (size_t i = 1; i < 8; ++i)
    {
        double r = i*0.5;
        std::cout <<  "(r, V) = (" << r << ", " << -b(r)/r << ")\n";
    }
}
//] [/barycentric_rational_example]

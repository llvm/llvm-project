// Copyright Paul A. Bristow, 2019
// Copyright Nick Thompson, 2019

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef BOOST_NO_CXX11_LAMBDAS
#  error "This example requires a C++11 compiler that supports lambdas.  Try C++11 or later."
#endif

//#define BOOST_MATH_INSTRUMENT_OOURA // or -DBOOST_MATH_INSTRUMENT_OOURA etc for diagnostics.

#include <boost/math/quadrature/ooura_fourier_integrals.hpp>
#include <boost/math/constants/constants.hpp>  // For pi (including for multiprecision types, if used.)

#include <cmath>
#include <iostream>
#include <limits>
#include <iostream>

int main()
{
  try
  {
  std::cout.precision(std::numeric_limits<double>::max_digits10); // Show all potentially significant digits.

  using boost::math::quadrature::ooura_fourier_sin;
  using boost::math::constants::half_pi;

//[ooura_fourier_integrals_example_1
  ooura_fourier_sin<double>integrator = ooura_fourier_sin<double>();
  // Use the default tolerance root_epsilon and eight levels for type double.

  auto f = [](double x)
  { // Simple reciprocal function for sinc.
    return 1 / x;
  };

  double omega = 1;
  std::pair<double, double> result = integrator.integrate(f, omega);
  std::cout << "Integral = " << result.first << ", relative error estimate " << result.second << std::endl;

//] [/ooura_fourier_integrals_example_1]

//[ooura_fourier_integrals_example_2

  constexpr double expected = half_pi<double>();
  std::cout << "pi/2 =     " << expected << ", difference " << result.first - expected << std::endl;
//] [/ooura_fourier_integrals_example_2]
  }
  catch (std::exception const & ex)
  {
    // Lacking try&catch blocks, the program will abort after any throw, whereas the
    // message below from the thrown exception will give some helpful clues as to the cause of the problem.
    std::cout << "\n""Message from thrown exception was:\n   " << ex.what() << std::endl;
  }
} // int main()

/*

//[ooura_fourier_integrals_example_output_1

integral = 1.5707963267948966, relative error estimate 1.2655356398390254e-11
pi/2 =     1.5707963267948966, difference 0

//] [/ooura_fourier_integrals_example_output_1]


//[ooura_fourier_integrals_example_diagnostic_output_1

ooura_fourier_sin with relative error goal 1.4901161193847656e-08 & 8 levels.
h = 1.000000000000000, I_h = 1.571890732004545 = 0x1.92676e56d853500p+0, absolute error estimate = nan
h = 0.500000000000000, I_h = 1.570793292491940 = 0x1.921f825c076f600p+0, absolute error estimate = 1.097439512605325e-03
h = 0.250000000000000, I_h = 1.570796326814776 = 0x1.921fb54458acf00p+0, absolute error estimate = 3.034322835882008e-06
h = 0.125000000000000, I_h = 1.570796326794897 = 0x1.921fb54442d1800p+0, absolute error estimate = 1.987898734512328e-11
Integral = 1.570796326794897e+00, relative error estimate 1.265535639839025e-11
pi/2 =     1.570796326794897e+00, difference 0.000000000000000e+00

//] [/ooura_fourier_integrals_example_diagnostic_output_1]

*/

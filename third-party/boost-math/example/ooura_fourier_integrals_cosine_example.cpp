// Copyright Paul A. Bristow, 2019
// Copyright Nick Thompson, 2019

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

//#define BOOST_MATH_INSTRUMENT_OOURA // or -DBOOST_MATH_INSTRUMENT_OOURA etc for diagnostic output.

#include <boost/math/quadrature/ooura_fourier_integrals.hpp> // For ooura_fourier_cos
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

  using boost::math::quadrature::ooura_fourier_cos;
  using boost::math::constants::half_pi;
  using boost::math::constants::e;

 //[ooura_fourier_integrals_cosine_example_1
  auto integrator = ooura_fourier_cos<double>();
  // Use the default tolerance root_epsilon and eight levels for type double.

  auto f = [](double x)
  { // More complex example function.
    return 1 / (x * x + 1);
  };

  double omega = 1;

  auto [result, relative_error] = integrator.integrate(f, omega);
  std::cout << "Integral = " << result << ", relative error estimate " << relative_error << std::endl;

  //] [/ooura_fourier_integrals_cosine_example_1]

  //[ooura_fourier_integrals_cosine_example_2

  constexpr double expected = half_pi<double>() / e<double>();
  std::cout << "pi/(2e) =  " << expected << ", difference " << result - expected << std::endl;
  //] [/ooura_fourier_integrals_cosine_example_2]
  }
  catch (std::exception const & ex)
  {
    // Lacking try&catch blocks, the program will abort after any throw, whereas the
    // message below from the thrown exception will give some helpful clues as to the cause of the problem.
    std::cout << "\n""Message from thrown exception was:\n   " << ex.what() << std::endl;
  }

} // int main()

/*

//[ooura_fourier_integrals_example_cosine_output_1
``
Integral = 0.57786367489546109, relative error estimate 6.4177395404415149e-09
pi/(2e) =  0.57786367489546087, difference 2.2204460492503131e-16
``
//] [/ooura_fourier_integrals_example_cosine_output_1]


//[ooura_fourier_integrals_example_cosine_diagnostic_output_1
``
ooura_fourier_cos with relative error goal 1.4901161193847656e-08 & 8 levels.
epsilon for type = 2.2204460492503131e-16
h = 1.000000000000000, I_h = 0.588268622591776 = 0x1.2d318b7e96dbe00p-1, absolute error estimate = nan
h = 0.500000000000000, I_h = 0.577871642184837 = 0x1.27decab8f07b200p-1, absolute error estimate = 1.039698040693926e-02
h = 0.250000000000000, I_h = 0.577863671186883 = 0x1.27ddbf42969be00p-1, absolute error estimate = 7.970997954576120e-06
h = 0.125000000000000, I_h = 0.577863674895461 = 0x1.27ddbf6271dc000p-1, absolute error estimate = 3.708578555361441e-09
Integral = 5.778636748954611e-01, relative error estimate 6.417739540441515e-09
pi/(2e)  = 5.778636748954609e-01, difference 2.220446049250313e-16
``
//] [/ooura_fourier_integrals_example_cosine_diagnostic_output_1]

*/

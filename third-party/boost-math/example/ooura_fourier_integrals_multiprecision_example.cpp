// Copyright Paul A. Bristow, 2019
// Copyright Nick Thompson, 2019

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(__cpp_structured_bindings) || (__cpp_structured_bindings < 201606L)
#  error "This example requires a C++17 compiler that supports 'structured bindings'. Try /std:c++17 or -std=c++17 or later."
#endif

//#define BOOST_MATH_INSTRUMENT_OOURA // or -DBOOST_MATH_INSTRUMENT_OOURA etc for diagnostic output.

#include <boost/math/quadrature/ooura_fourier_integrals.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp> // for cpp_bin_float_quad, cpp_bin_float_50...
#include <boost/math/constants/constants.hpp>  // For pi (including for multiprecision types, if used.)

#include <cmath>
#include <iostream>
#include <limits>
#include <iostream>
#include <exception>

int main()
{
  try
  {
    typedef boost::multiprecision::cpp_bin_float_quad Real;

    std::cout.precision(std::numeric_limits<Real>::max_digits10); // Show all potentially significant digits.

    using boost::math::quadrature::ooura_fourier_cos;
    using boost::math::constants::half_pi;
    using boost::math::constants::e;

   //[ooura_fourier_integrals_multiprecision_example_1

    // Use the default parameters for tolerance root_epsilon and eight levels for a type of 8 bytes.
    //auto integrator = ooura_fourier_cos<Real>();
    // Decide on a (tight) tolerance.
    const Real tol = 2 * std::numeric_limits<Real>::epsilon();
    auto integrator = ooura_fourier_cos<Real>(tol, 8); // Loops or gets worse for more than 8.

    auto f = [](Real x)
    { // More complex example function.
      return 1 / (x * x + 1);
    };

    double omega = 1;
    auto [result, relative_error] = integrator.integrate(f, omega);

    //] [/ooura_fourier_integrals_multiprecision_example_1]

    //[ooura_fourier_integrals_multiprecision_example_2
    std::cout << "Integral = " << result << ", relative error estimate " << relative_error << std::endl;

    const Real expected = half_pi<Real>() / e<Real>(); // Expect integral = 1/(2e)
    std::cout << "pi/(2e)  = " << expected << ", difference " << result - expected << std::endl;
    //] [/ooura_fourier_integrals_multiprecision_example_2]
  }
  catch (std::exception const & ex)
  {
    // Lacking try&catch blocks, the program will abort after any throw, whereas the
    // message below from the thrown exception will give some helpful clues as to the cause of the problem.
    std::cout << "\n""Message from thrown exception was:\n   " << ex.what() << std::endl;
  }
} // int main()

/*

//[ooura_fourier_integrals_example_multiprecision_output_1
``
Integral = 0.5778636748954608589550465916563501587, relative error estimate 4.609814684522163895264277312610830278e-17
pi/(2e) = 0.5778636748954608659545328919193707407, difference -6.999486300263020581921171645255733758e-18
``
//] [/ooura_fourier_integrals_example_multiprecision_output_1]


//[ooura_fourier_integrals_example_multiprecision_diagnostic_output_1
``
ooura_fourier_cos with relative error goal 3.851859888774471706111955885169854637e-34 & 15 levels.
epsilon for type = 1.925929944387235853055977942584927319e-34
h = 1.000000000000000000000000000000000, I_h = 0.588268622591776615359568690603776 = 0.5882686225917766153595686906037760, absolute error estimate = nan
h = 0.500000000000000000000000000000000, I_h = 0.577871642184837461311756940493259 = 0.5778716421848374613117569404932595, absolute error estimate = 1.039698040693915404781175011051656e-02
h = 0.250000000000000000000000000000000, I_h = 0.577863671186882539559996800783122 = 0.5778636711868825395599968007831220, absolute error estimate = 7.970997954921751760139710137450075e-06
h = 0.125000000000000000000000000000000, I_h = 0.577863674895460885593491133506723 = 0.5778636748954608855934911335067232, absolute error estimate = 3.708578346033494332723601147051768e-09
h = 0.062500000000000000000000000000000, I_h = 0.577863674895460858955046591656350 = 0.5778636748954608589550465916563502, absolute error estimate = 2.663844454185037302771663314961535e-17
h = 0.031250000000000000000000000000000, I_h = 0.577863674895460858955046591656348 = 0.5778636748954608589550465916563484, absolute error estimate = 1.733336949948512267750380148326435e-33
h = 0.015625000000000000000000000000000, I_h = 0.577863674895460858955046591656348 = 0.5778636748954608589550465916563479, absolute error estimate = 4.814824860968089632639944856462318e-34
h = 0.007812500000000000000000000000000, I_h = 0.577863674895460858955046591656347 = 0.5778636748954608589550465916563473, absolute error estimate = 6.740754805355325485695922799047246e-34
h = 0.003906250000000000000000000000000, I_h = 0.577863674895460858955046591656347 = 0.5778636748954608589550465916563475, absolute error estimate = 1.925929944387235853055977942584927e-34
Integral = 5.778636748954608589550465916563475e-01, relative error estimate 3.332844800697411177051445985473052e-34
pi/(2e)  = 5.778636748954608589550465916563481e-01, difference -6.740754805355325485695922799047246e-34
``
//] [/ooura_fourier_integrals_example_multiprecision_diagnostic_output_1]

*/

// Copyright Paul A. Bristow, 2019

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

/*! \title  Simple example of computation of the Jacobi Zeta function using Boost.Math,
and also using corresponding WolframAlpha commands.
*/

#ifdef BOOST_NO_CXX11_NUMERIC_LIMITS
#  error "This example requires a C++ compiler that supports C++11 numeric_limits. Try C++11 or later."
#endif

#include <boost/math/special_functions/jacobi_zeta.hpp> // For jacobi_zeta function.
#include <boost/multiprecision/cpp_bin_float.hpp> // For cpp_bin_float_50.

#include <iostream>
#include <limits>
#include <iostream>
#include <exception>

int main()
{
  try
  {
    std::cout.precision(std::numeric_limits<double>::max_digits10); // Show all potentially significant digits.
    std::cout.setf(std::ios_base::showpoint); // Include any significant trailing zeros.

    using boost::math::jacobi_zeta; // jacobi_zeta(T1 k, T2 phi) |k| <=1, k = sqrt(m)
    using boost::multiprecision::cpp_bin_float_50;

    // Wolfram Mathworld function JacobiZeta[phi, m] where m = k^2
    // JacobiZeta[phi,m] gives the Jacobi zeta function Z(phi | m)

    // If phi = 2, and elliptic modulus k = 0.9 so m = 0.9 * 0.9 = 0.81

    // https://reference.wolfram.com/language/ref/JacobiZeta.html // Function information.
    // A simple computation using phi = 2. and m = 0.9 * 0.9
    // JacobiZeta[2, 0.9 * 0.9]
    // https://www.wolframalpha.com/input/?i=JacobiZeta%5B2,+0.9+*+0.9%5D
    // -0.248584...
    // To get the expected 17 decimal digits precision for a 64-bit double type,
    // we need to ask thus:
    // N[JacobiZeta[2, 0.9 * 0.9],17]
    // https://www.wolframalpha.com/input/?i=N%5BJacobiZeta%5B2,+0.9+*+0.9%5D,17%5D

    double k = 0.9;
    double m = k * k;
    double phi = 2.;

    std::cout << "m = k^2 = " << m << std::endl;  // m = k^2 =  0.81000000000000005
    std::cout << "jacobi_zeta(" << k << ", " << phi << " )  = " << jacobi_zeta(k, phi) << std::endl;
    // jacobi_zeta(0.90000000000000002, 2.0000000000000000 )  =
    //  -0.24858442708494899  Boost.Math
    //  -0.24858442708494893  Wolfram
    // that agree within the expected precision of 17 decimal digits for 64-bit type double.

    // We can also easily get a higher precision too:
    // For example, to get 50 decimal digit precision using WolframAlpha:
    // N[JacobiZeta[2, 0.9 * 0.9],50]
    // https://www.wolframalpha.com/input/?i=N%5BJacobiZeta%5B2,+0.9+*+0.9%5D,50%5D
    // -0.24858442708494893408462856109734087389683955309853

    // Using Boost.Multiprecision we can do them same almost as easily.

    // To check that we are not losing precision, we show all the significant digits of the arguments ad result:
    std::cout.precision(std::numeric_limits<cpp_bin_float_50>::digits10); // Show all significant digits.

    // We can force the computation to use 50 decimal digit precision thus:
    cpp_bin_float_50 k50("0.9");
    cpp_bin_float_50 phi50("2.");

    std::cout << "jacobi_zeta(" << k50 << ", " << phi50 << " )  = " << jacobi_zeta(k50, phi50) << std::endl;
    // jacobi_zeta(0.90000000000000000000000000000000000000000000000000,
    //   2.0000000000000000000000000000000000000000000000000 )
    //   = -0.24858442708494893408462856109734087389683955309853

    // and a comparison with Wolfram shows agreement to the expected precision.
    // -0.24858442708494893408462856109734087389683955309853  Boost.Math
    // -0.24858442708494893408462856109734087389683955309853  Wolfram

    // Taking care not to fall into the awaiting pit, we ensure that ALL arguments passed are of the
    // appropriate 50-digit precision and do NOT suffer from precision reduction to that of type double,
    // We do NOT write:
    std::cout << "jacobi_zeta<cpp_bin_float_50>(0.9, 2.)  = " << jacobi_zeta<cpp_bin_float_50>(0.9, 2) << std::endl;
    // jacobi_zeta(0.90000000000000000000000000000000000000000000000000,
    //   2.0000000000000000000000000000000000000000000000000 )
    //   = -0.24858442708494895921459900494815797085727097762164  << Wrong at about 17th digit!
    //     -0.24858442708494893408462856109734087389683955309853  Wolfram
  }
  catch (std::exception const& ex)
  {
    // Lacking try&catch blocks, the program will abort after any throw, whereas the
    // message below from the thrown exception will give some helpful clues as to the cause of the problem.
    std::cout << "\n""Message from thrown exception was:\n   " << ex.what() << std::endl;
    // An example of message:
    // std::cout << " = " << jacobi_zeta(2, 0.5) << std::endl;
    // Message from thrown exception was:
    // Error in function boost::math::ellint_k<long double>(long double) : Got k = 2, function requires |k| <= 1
    // Shows that first parameter is k and is out of range, as the definition in docs jacobi_zeta(T1 k, T2 phi);
  }
} // int main()

//  Copyright John Maddock 2014
//  Copyright Christopher Kormanyos 2014.
//  Copyright Paul A. Bristow 2016.

//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Contains Quickbook snippets as C++ comments - do not remove.

/*
`This example shows use of a specified-width floating-point typedef
to evaluate a moderately complex math function using
[@http://en.wikipedia.org/wiki/Double-precision_floating-point_format IEEE754 64-bit double-precision].
about 15 decimal digits `std::numeric_limits<boost::float64_t>::digits10`,
(but never exceeding 17 decimal digits `std::numeric_limits<boost::float64_t>::max_digits10`).

The Jahnke-Emden lambda function is described at

Weisstein, Eric W. "Lambda Function." From MathWorld--A Wolfram Web Resource.
http://mathworld.wolfram.com/LambdaFunction.html

E. Jahnke and F. Emden, "Tables of Functions with Formulae and Curves,"
Dover, New York, 4th ed., (1945), pages 180-188.

*/

//[cstdfloat_example_1
#include <boost/cstdfloat.hpp> // For float_64_t, float128_t. Must be first include!
#include <cmath>  // for pow function.
#include <boost/math/special_functions.hpp> // For gamma function.
//] [/cstdfloat_example_1]

#include <iostream>

/*!
Function max_digits10
Returns maximum number of possibly significant decimal digits for a floating-point type FPT,
even for older compilers/standard libraries that
lack support for std::numeric_limits<FPT>::max_digits10,
when the Kahan formula 2 + binary_digits * 0.3010 is used instead.
Also provides the correct result for Visual Studio 2010
(where the value 8 provided for float is wrong).
*/
namespace boost
{
template <typename FPT>
const int max_digits10()
{
// Since max_digits10 is not defined (or wrong) on older systems, define a local max_digits10.
  // Usage:   int m = max_digits10<boost::float64_t>();
  const int m =
#if (defined BOOST_NO_CXX11_NUMERIC_LIMITS) || (_MSC_VER == 1600) // is wrongly 8 not 9 for VS2010.
  2 + std::numeric_limits<FPT>::digits * 3010/10000;
#else
  std::numeric_limits<FPT>::max_digits10;
#endif
  return m;
}
} // namespace boost

//`Define the Jahnke-Emden_lambda function.
//[cstdfloat_example_2
boost::float64_t jahnke_emden_lambda(boost::float64_t v, boost::float64_t x)
{
  const boost::float64_t gamma_v_plus_one = boost::math::tgamma(v + 1);
  const boost::float64_t x_half_pow_v     = std::pow(x /2, v);

  return gamma_v_plus_one * boost::math::cyl_bessel_j(x, v) / x_half_pow_v;
}
//] [/cstdfloat_example_2]

int main()
{
  std::cout.setf(std::ios::showpoint); // show all significant trailing zeros.

    long double p = 1.L;
  //std::cout.precision(std::numeric_limits<long double>::digits10);

  std::cout << "pi = "  << p << std::endl;

//[cstdfloat_example_3
//`Ensure that all possibly significant digits (17) including trailing zeros are shown.

  std::cout.precision(std::numeric_limits<boost::float64_t>::max_digits10);
  std::cout.setf(std::ios::showpoint); // Show trailing zeros.

  try
  { // Always use try'n'catch blocks to ensure any error messages are displayed.

  // Evaluate and display an evaluation of the Jahnke-Emden lambda function:
  boost::float64_t v = 1.;
  boost::float64_t x = 1.;
  std::cout << jahnke_emden_lambda(v, x) << std::endl; // 0.88010117148986700
//] [/cstdfloat_example_3]

  // We can show some evaluations with various precisions:
  { // float64_t
    for (int i = 0; i < 10; i++)
    {
      std::cout << std::setprecision(2) << boost::float64_t(i) << ' '
        << std::setprecision(std::numeric_limits<boost::float64_t>::max_digits10)
        << jahnke_emden_lambda(boost::float64_t(i), v) << std::endl; //
    }
  }
  { // floatmax_t = the maximum available on this platform.
    for (int i = 0; i < 10; i++)
    {
      std::cout << std::setprecision(2) << boost::floatmax_t(i) << ' '
        << std::setprecision(std::numeric_limits<boost::floatmax_t>::max_digits10)
        << jahnke_emden_lambda(boost::floatmax_t(i), v) << std::endl; //
    }
  }
  // Show the precision of long double (this might be 64, 80 or 128 bits).
  std::cout << "Floating-point type long double is available with:" << std::endl;
  std::cout << "  std::numeric_limits<long double>::digits10 == "
    << std::numeric_limits<long double>::digits10 << std::endl; // 18
  std::cout << "  std::numeric_limits<long double>::max_digits10 == "
    << std::numeric_limits<long double>::max_digits10 << std::endl; // 21
  long double p = boost::math::constants::pi<double>();
  std::cout.precision(std::numeric_limits<long double>::digits10);
  std::cout << "pi = "  << p << std::endl;

//[cstdfloat_constant_2
//`These allow floating-point [*constants of at least the specified width] to be declared:

  // Declare Archimedes' constant using float32_t with approximately 7 decimal digits of precision.
  static const boost::float32_t pi = BOOST_FLOAT32_C(3.1415926536);

  // Declare the Euler-gamma constant with approximately 15 decimal digits of precision.
  static const boost::float64_t euler =
     BOOST_FLOAT64_C(0.57721566490153286060651209008240243104216);

  // Declare the Golden Ratio constant with the maximum decimal digits of precision that the platform supports.
  static const boost::floatmax_t golden_ratio =
     BOOST_FLOATMAX_C(1.61803398874989484820458683436563811772);
//] [/cstdfloat_constant_2]

// http://www.boost.org/doc/libs/1_55_0/libs/multiprecision/doc/html/boost_multiprecision/tut/floats/float128.html
//[cstdfloat_constant_1
// Display the constant pi to the maximum available precision.
  boost::floatmax_t pi_max = boost::math::constants::pi<boost::floatmax_t>();
  std::cout.precision(std::numeric_limits<boost::floatmax_t>::digits10);
  std::cout << "Most precise pi = "  << pi_max << std::endl;
// If floatmax_t is float_128_t, then
// Most precise pi = 3.141592653589793238462643383279503
//] [/cstdfloat_constant_1]

// Test all the floating-point precisions in turn, and if they are available
// then display how many decimal digits of precision.
#ifdef BOOST_FLOAT16_C
  std::cout << "Floating-point type boost::float16_t is available." << std::endl;
#else
  std::cout << "Floating-point type boost::float16_t is NOT available." << std::endl;
#endif

#ifdef BOOST_FLOAT32_C
  std::cout << "Floating-point type boost::float32_t is available." << std::endl;
  std::cout << "  std::numeric_limits<boost::float32_t>::digits10 == "
    << std::numeric_limits<boost::float32_t>::digits10 << std::endl;
  std::cout << "  std::numeric_limits<boost::float32_t>::max_digits10 == "
    << std::numeric_limits<boost::float32_t>::max_digits10 << std::endl;
#else
  std::cout << "Floating-point type boost::float32_t is NOT available." << std::endl;
#endif

#ifdef BOOST_FLOAT64_C
  std::cout << "Floating-point type boost::float64_t is available." << std::endl;
    std::cout << "  std::numeric_limits<boost::float64_t>::digits10 == "
    << std::numeric_limits<boost::float64_t>::digits10 << std::endl;
  std::cout << "  std::numeric_limits<boost::float64_t>::max_digits10 == "
    << std::numeric_limits<boost::float64_t>::max_digits10 << std::endl;
#else
  std::cout << "Floating-point type boost::float64_t is NOT available." << std::endl;
#endif

#ifdef BOOST_FLOAT80_C
  std::cout << "Floating-point type boost::float80_t is available." << std::endl;
  std::cout << "  std::numeric_limits<boost::float80_t>::digits10 == "
    << std::numeric_limits<boost::float80_t>::digits10 << std::endl;
  std::cout << "  std::numeric_limits<boost::float80_t>::max_digits10 == "
    << std::numeric_limits<boost::float80_t>::max_digits10 << std::endl;
#else
  std::cout << "Floating-point type boost::float80_t is NOT available." << std::endl;
#endif

#ifdef BOOST_FLOAT128_C
  std::cout << "Floating-point type boost::float128_t is available." << std::endl;
    std::cout << "  std::numeric_limits<boost::float128_t>::digits10 == "
    << std::numeric_limits<boost::float128_t>::digits10 << std::endl;
  std::cout << "  std::numeric_limits<boost::float128_t>::max_digits10 == "
    << std::numeric_limits<boost::float128_t>::max_digits10 << std::endl;
#else
  std::cout << "Floating-point type boost::float128_t is NOT available." << std::endl;
#endif

// Show some constants at a precision depending on the available type(s).
#ifdef BOOST_FLOAT16_C
  std::cout.precision(boost::max_digits10<boost::float16_t>()); // Show all significant decimal digits,
  std::cout.setf(std::ios::showpoint); // including all significant trailing zeros.

  std::cout << "BOOST_FLOAT16_C(123.456789012345678901234567890) = "
    << BOOST_FLOAT16_C(123.456789012345678901234567890) << std::endl;
  // BOOST_FLOAT16_C(123.456789012345678901234567890) = 123.45678901234568
#endif

//[floatmax_widths_1
#ifdef BOOST_FLOAT32_C
  std::cout.precision(boost::max_digits10<boost::float32_t>()); // Show all significant decimal digits,
  std::cout.setf(std::ios::showpoint); // including all significant trailing zeros.
  std::cout << "BOOST_FLOAT32_C(123.4567890123456789012345678901234567890) = "
    << BOOST_FLOAT32_C(123.4567890123456789012345678901234567890) << std::endl;
  //   BOOST_FLOAT32_C(123.4567890123456789012345678901234567890) = 123.456787
#endif
//] [/floatmax_widths_1]

#ifdef BOOST_FLOAT64_C
  std::cout.precision(boost::max_digits10<boost::float64_t>()); // Show all significant decimal digits,
  std::cout.setf(std::ios::showpoint); // including all significant trailing zeros.
  std::cout << "BOOST_FLOAT64_C(123.4567890123456789012345678901234567890) = "
    << BOOST_FLOAT64_C(123.4567890123456789012345678901234567890) << std::endl;
  // BOOST_FLOAT64_C(123.4567890123456789012345678901234567890) = 123.45678901234568
#endif

#ifdef BOOST_FLOAT80_C
  std::cout.precision(boost::max_digits10<boost::float80_t>()); // Show all significant decimal digits,
  std::cout.setf(std::ios::showpoint); // including all significant trailing zeros.
  std::cout << "BOOST_FLOAT80_C(123.4567890123456789012345678901234567890) = "
    << BOOST_FLOAT80_C(123.4567890123456789012345678901234567890) << std::endl;
  // BOOST_FLOAT80_C(123.4567890123456789012345678901234567890) = 123.456789012345678903
#endif

#ifdef BOOST_FLOAT128_C
  std::cout.precision(boost::max_digits10<boost::float128_t>()); // Show all significant decimal digits,
  std::cout.setf(std::ios::showpoint); // including all significant trailing zeros.
  std::cout << "BOOST_FLOAT128_C(123.4567890123456789012345678901234567890) = "
    << BOOST_FLOAT128_C(123.4567890123456789012345678901234567890) << std::endl;
  // BOOST_FLOAT128_C(123.4567890123456789012345678901234567890) = 123.456789012345678901234567890123453
#endif

/*
//[floatmax_widths_2
BOOST_FLOAT32_C(123.4567890123456789012345678901234567890) = 123.456787
BOOST_FLOAT64_C(123.4567890123456789012345678901234567890) = 123.45678901234568
BOOST_FLOAT80_C(123.4567890123456789012345678901234567890) = 123.456789012345678903
BOOST_FLOAT128_C(123.4567890123456789012345678901234567890) = 123.456789012345678901234567890123453
//] [/floatmax_widths_2]
*/

// Display the precisions available for floatmax_t

#ifdef BOOST_FLOATMAX_C
  BOOST_MATH_ASSERT(std::numeric_limits<boost::floatmax_t>::is_specialized == true);
  BOOST_MATH_ASSERT(std::numeric_limits<boost::floatmax_t>::is_iec559 == true);
  BOOST_MATH_ASSERT(BOOST_FLOATMAX_C(0.) == 0);

  std::cout << "floatmax_t " << std::numeric_limits<boost::floatmax_t>::digits << " bits\n" // 113
    << std::numeric_limits<boost::floatmax_t>::digits10 << " decimal digits\n" // 34
    << std::numeric_limits<boost::floatmax_t>::max_digits10 << " max_digits\n" // 36
    << std::numeric_limits<boost::floatmax_t>::radix << " radix\n"
    << std::endl;

  int significand_bits = std::numeric_limits<boost::floatmax_t>::digits;
  int exponent_max = std::numeric_limits<boost::floatmax_t>::max_exponent;
  int exponent_min = std::numeric_limits<boost::floatmax_t>::min_exponent;
  int exponent_bits = 1 + static_cast<int>(std::log2(std::numeric_limits<boost::floatmax_t>::max_exponent));
  int sign_bits = std::numeric_limits<boost::floatmax_t>::is_signed;

  std::cout << "significand_bits (including one implicit bit)" << significand_bits
    << ", exponent_bits " << exponent_bits
    << ", sign_bits " << sign_bits << std::endl;

  // One can compute the total number of bits in the floatmax_t,
  // but probably not at compile time.

  std::cout << "bits = " << significand_bits + exponent_bits + sign_bits -1 << std::endl;
  // -1 to take account of the implicit bit that is not part of the physical layout.

  // One can compare typedefs (but, of course, only those that are defined for the platform in use.)
  std::cout.setf(std::ios::boolalpha);
  std::cout << "double, double: " << std::is_same<double, double>::value << std::endl;
  bool b = boost::is_same<boost::floatmax_t, boost::float64_t>::value;
  std::cout << "boost::is_same<boost::floatmax_t, boost::float64_t>::value; " << b << std::endl;
  std::cout << "floatmax_t, float64_t: "
    << std::is_same<boost::floatmax_t, boost::float64_t>::value << std::endl;

/*`So the simplest way of obtaining the total number of bits in the floatmax_t
is to infer it from the std::numeric_limits<>::digits value.
This is possible because the type, must be a IEEE754 layout. */
//[floatmax_1
  const int fpbits =
    (std::numeric_limits<boost::floatmax_t>::digits == 113) ? 128 :
    (std::numeric_limits<boost::floatmax_t>::digits == 64) ? 80 :
    (std::numeric_limits<boost::floatmax_t>::digits == 53) ? 64 :
    (std::numeric_limits<boost::floatmax_t>::digits == 24) ? 32 :
    (std::numeric_limits<boost::floatmax_t>::digits == 11) ? 16 :
    0; // Unknown - not IEEE754 format.
   std::cout << fpbits << " bits." << std::endl;
//] [/floatmax_1]
#endif

  }
  catch (std::exception ex)
  { // Display details about why any exceptions are thrown.
    std::cout << "Thrown exception " << ex.what() << std::endl;
  }

} // int main()

/*
[cstdfloat_output

GCC 4.8.1 with quadmath

 pi = 1.00000
 0.88010117148986700
 0.0 0.0000000000000000
 1.0 0.88010117148986700
 2.0 4.6137984620549872
 3.0 16.274830009244951
 4.0 -25.360637961042869
 5.0 -1257.9038883512264
 6.0 -12749.592182518225
 7.0 -3020.9830849309437
 8.0 2421897.6013183584
 9.0 45577595.449204877
 0.0 0.00000000000000000000000000000000000
 1.0 0.880101171489866995756301548681221902
 2.0 4.61379846205498722611082484945654869
 3.0 16.2748300092449511566883302293717861
 4.0 -25.3606379610428689375112298876047134
 5.0 -1257.90388835122644195507746189832687
 6.0 -12749.5921825182249449426308274269104
 7.0 -3020.98308493094373261556029319763184
 8.0 2421897.60131835844367742538452148438
 9.0 45577595.4492048770189285278320312500
 Floating-point type long double is available with:
 std::numeric_limits<long double>::digits10 == 18
 std::numeric_limits<long double>::max_digits10 == 21
 pi = 3.14159265358979312
 Most precise pi = 3.141592653589793238462643383279503
 Floating-point type boost::float16_t is NOT available.
 Floating-point type boost::float32_t is available.
 std::numeric_limits<boost::float32_t>::digits10 == 6
 std::numeric_limits<boost::float32_t>::max_digits10 == 9
 Floating-point type boost::float64_t is available.
 std::numeric_limits<boost::float64_t>::digits10 == 15
 std::numeric_limits<boost::float64_t>::max_digits10 == 17
 Floating-point type boost::float80_t is available.
 std::numeric_limits<boost::float80_t>::digits10 == 18
 std::numeric_limits<boost::float80_t>::max_digits10 == 21
 Floating-point type boost::float128_t is available.
 std::numeric_limits<boost::float128_t>::digits10 == 34
 std::numeric_limits<boost::float128_t>::max_digits10 == 36
 BOOST_FLOAT32_C(123.4567890123456789012345678901234567890) = 123.456787
 BOOST_FLOAT64_C(123.4567890123456789012345678901234567890) = 123.45678901234568
 BOOST_FLOAT80_C(123.4567890123456789012345678901234567890) = 123.456789012345678903
 BOOST_FLOAT128_C(123.4567890123456789012345678901234567890) = 123.456789012345678901234567890123453
 floatmax_t 113 bits
 34 decimal digits
 36 max_digits
 2 radix

 significand_bits (including one implicit bit)113, exponent_bits 15, sign_bits 1
 bits = 128
 double, double: true
 boost::is_same<boost::floatmax_t, boost::float64_t>::value; false
 floatmax_t, float64_t: false
 128 bits.

 RUN SUCCESSFUL (total time: 53ms)

GCC 6.1.1

pi = 1.00000
0.88010117148986700
0.0 0.0000000000000000
1.0 0.88010117148986700
2.0 4.6137984620549872
3.0 16.274830009244951
4.0 -25.360637961042869
5.0 -1257.9038883512264
6.0 -12749.592182518225
7.0 -3020.9830849309437
8.0 2421897.6013183584
9.0 45577595.449204877
0.0 0.00000000000000000000000000000000000
1.0 0.880101171489866995756301548681221902
2.0 4.61379846205498722611082484945654869
3.0 16.2748300092449511566883302293717861
4.0 -25.3606379610428689375112298876047134
5.0 -1257.90388835122644195507746189832687
6.0 -12749.5921825182249449426308274269104
7.0 -3020.98308493094373261556029319763184
8.0 2421897.60131835844367742538452148438
9.0 45577595.4492048770189285278320312500
Floating-point type long double is available with:
  std::numeric_limits<long double>::digits10 == 18
  std::numeric_limits<long double>::max_digits10 == 21
pi = 3.14159265358979312
Most precise pi = 3.14159265358979323846264338327950
Floating-point type boost::float16_t is NOT available.
Floating-point type boost::float32_t is available.
  std::numeric_limits<boost::float32_t>::digits10 == 6
  std::numeric_limits<boost::float32_t>::max_digits10 == 9
Floating-point type boost::float64_t is available.
  std::numeric_limits<boost::float64_t>::digits10 == 15
  std::numeric_limits<boost::float64_t>::max_digits10 == 17
Floating-point type boost::float80_t is available.
  std::numeric_limits<boost::float80_t>::digits10 == 18
  std::numeric_limits<boost::float80_t>::max_digits10 == 21
Floating-point type boost::float128_t is available.
  std::numeric_limits<boost::float128_t>::digits10 == 33
  std::numeric_limits<boost::float128_t>::max_digits10 == 36
BOOST_FLOAT32_C(123.4567890123456789012345678901234567890) = 123.456787
BOOST_FLOAT64_C(123.4567890123456789012345678901234567890) = 123.45678901234568
BOOST_FLOAT80_C(123.4567890123456789012345678901234567890) = 123.456789012345678903
BOOST_FLOAT128_C(123.4567890123456789012345678901234567890) = 123.456789012345678901234567890123453
floatmax_t 113 bits
33 decimal digits
36 max_digits
2 radix

significand_bits (including one implicit bit)113, exponent_bits 15, sign_bits 1
bits = 128
double, double: true
boost::is_same<boost::floatmax_t, boost::float64_t>::value; false
floatmax_t, float64_t: false
128 bits.


 MSVC 2013  64-bit

 1>  pi = 1.00000
 1>  0.88010117148986700
 1>  0.00 0.00000000000000000
 1>  1.0 0.88010117148986700
 1>  2.0 4.6137984620549854
 1>  3.0 16.274830009244948
 1>  4.0 -25.360637961042869
 1>  5.0 -1257.9038883512258
 1>  6.0 -12749.592182518225
 1>  7.0 -3020.9830849309396
 1>  8.0 2421897.6013183575
 1>  9.0 45577595.449204892
 1>  0.00 0.00000000000000000
 1>  1.0 0.88010117148986700
 1>  2.0 4.6137984620549854
 1>  3.0 16.274830009244948
 1>  4.0 -25.360637961042869
 1>  5.0 -1257.9038883512258
 1>  6.0 -12749.592182518225
 1>  7.0 -3020.9830849309396
 1>  8.0 2421897.6013183575
 1>  9.0 45577595.449204892
 1>  Floating-point type long double is available with:
 1>    std::numeric_limits<long double>::digits10 == 15
 1>    std::numeric_limits<long double>::max_digits10 == 17
 1>  pi = 3.14159265358979
 1>  Most precise pi = 3.14159265358979
 1>  Floating-point type boost::float16_t is NOT available.
 1>  Floating-point type boost::float32_t is available.
 1>    std::numeric_limits<boost::float32_t>::digits10 == 6
 1>    std::numeric_limits<boost::float32_t>::max_digits10 == 9
 1>  Floating-point type boost::float64_t is available.
 1>    std::numeric_limits<boost::float64_t>::digits10 == 15
 1>    std::numeric_limits<boost::float64_t>::max_digits10 == 17
 1>  Floating-point type boost::float80_t is NOT available.
 1>  Floating-point type boost::float128_t is NOT available.
 1>  BOOST_FLOAT32_C(123.4567890123456789012345678901234567890) = 123.456787
 1>  BOOST_FLOAT64_C(123.4567890123456789012345678901234567890) = 123.45678901234568
 1>  floatmax_t 53 bits
 1>  15 decimal digits
 1>  17 max_digits
 1>  2 radix
 1>
 1>  significand_bits (including one implicit bit)53, exponent_bits 11, sign_bits 1
 1>  bits = 64
 1>  double, double: true
 1>  boost::is_same<boost::floatmax_t, boost::float64_t>::value; true
 1>  floatmax_t, float64_t: true
 1>  64 bits.
] [/cstdfloat_output]


*/


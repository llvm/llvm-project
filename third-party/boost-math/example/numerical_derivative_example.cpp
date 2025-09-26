// Copyright Christopher Kormanyos 2013.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or
// copy at http://www.boost.org/LICENSE_1_0.txt).

#ifdef _MSC_VER
#  pragma warning (disable : 4996) // assignment operator could not be generated.
#endif

# include <iostream>
# include <iomanip>
# include <limits>
# include <type_traits>
# include <cmath>

#include <boost/math/tools/assert.hpp>
#include <boost/math/special_functions/next.hpp> // for float_distance

//[numeric_derivative_example
/*`The following example shows how multiprecision calculations can be used to
obtain full precision in a numerical derivative calculation that suffers from precision loss.

Consider some well-known central difference rules for numerically
computing the 1st derivative of a function [f'(x)] with [/x] real.

Need a reference here?  Introduction to Partial Differential Equations, Peter J. Olver
 December 16, 2012 

Here, the implementation uses a C++ template that can be instantiated with various
floating-point types such as `float`, `double`, `long double`, or even
a user-defined floating-point type like __multiprecision.

We will now use the derivative template with the built-in type `double` in
order to numerically compute the derivative of a function, and then repeat
with a 5 decimal digit higher precision user-defined floating-point type. 

Consider the function  shown below.
!!
(3)
We will now take the derivative of this function with respect to x evaluated
at x = 3= 2. In other words,

(4)

The expected result is

 0:74535 59924 99929 89880 . (5)
The program below uses the derivative template in order to perform
the numerical calculation of this derivative. The program also compares the
numerically-obtained result with the expected result and reports the absolute
relative error scaled to a deviation that can easily be related to the number of
bits of lost precision.

*/

/*` [note Requires the C++11 feature of
[@http://en.wikipedia.org/wiki/Anonymous_function#C.2B.2B anonymous functions]
for the derivative function calls like `[]( const double & x_) -> double`.
*/



template <typename value_type,  typename function_type>
value_type derivative (const value_type x, const value_type dx, function_type function)
{
  /*! \brief Compute the derivative of function using a 3-point central difference rule of O(dx^6).
    \tparam value_type, floating-point type, for example: `double` or `cpp_dec_float_50`
    \tparam function_type  
    
    \param x Value at which to evaluate derivative.
    \param dx Incremental step-size.
    \param function Function whose derivative is to computed.
  
    \return derivative at x.
  */

  static_assert(false == std::numeric_limits<value_type>::is_integer, "value_type must be a floating-point type!");

  const value_type dx2(dx * 2U);
  const value_type dx3(dx * 3U);
  // Difference terms.
  const value_type m1 ((function (x + dx) - function(x - dx)) / 2U);
  const value_type m2 ((function (x + dx2) - function(x - dx2)) / 4U);
  const value_type m3 ((function (x + dx3) - function(x - dx3)) / 6U);
  const value_type fifteen_m1 (m1 * 15U);
  const value_type six_m2 (m2 * 6U);
  const value_type ten_dx (dx * 10U);
  return ((fifteen_m1 - six_m2) + m3) / ten_dx;  // Derivative.
} // 

#include <boost/multiprecision/cpp_dec_float.hpp>
  using boost::multiprecision::number;
  using boost::multiprecision::cpp_dec_float;

// Re-compute using 5 extra decimal digits precision (22) than double (17).
#define MP_DIGITS10 unsigned (std::numeric_limits<double>::max_digits10 + 5)

typedef cpp_dec_float<MP_DIGITS10> mp_backend;
typedef number<mp_backend> mp_type;


int main()
{
  {
    const double d =
      derivative
      ( 1.5, // x = 3.2
        std::ldexp (1., -9), // step size 2^-9 = see below for choice.
        [](const double & x)->double // Function f(x).
        {
          return std::sqrt((x * x) - 1.) - std::acos(1. / x);
        }
      );
  
    // The 'exactly right' result is [sqrt]5 / 3 = 0.74535599249992989880.
    const double rel_error = (d - 0.74535599249992989880) / 0.74535599249992989880;
    const double bit_error = std::abs(rel_error) / std::numeric_limits<double>::epsilon();
    std::cout.precision (std::numeric_limits<double>::digits10); // Show all guaranteed decimal digits.
    std::cout << std::showpoint ; // Ensure that any trailing zeros are shown too.

    std::cout << " derivative : " << d << std::endl;
    std::cout << " expected   : " << 0.74535599249992989880 << std::endl;
    // Can compute an 'exact' value using multiprecision type.
    std::cout << " expected   : " << sqrt(static_cast<mp_type>(5))/3U << std::endl;
    std::cout << " bit_error : " << static_cast<unsigned long>(bit_error)  << std::endl;

    std::cout.precision(6);
    std::cout << "float_distance = " << boost::math::float_distance(0.74535599249992989880, d) << std::endl;

  }

  { // Compute using multiprecision type with an extra 5 decimal digits of precision.
    const mp_type mp =
      derivative(mp_type(mp_type(3) / 2U), // x = 3/2
        mp_type(mp_type(1) / 10000000U), // Step size 10^7.
        [](const mp_type & x)->mp_type
        {
          return sqrt((x * x) - 1.) - acos (1. / x); // Function
        }
    );

    const double d = mp.convert_to<double>(); // Convert to closest double.
    const double rel_error = (d - 0.74535599249992989880) / 0.74535599249992989880;
    const double bit_error = std::abs (rel_error) / std::numeric_limits<double>::epsilon();
    std::cout.precision (std::numeric_limits <double>::digits10); // All guaranteed decimal digits.
    std::cout << std::showpoint ; // Ensure that any trailing zeros are shown too.
    std::cout << " derivative : " << d << std::endl;
    // Can compute an 'exact' value using multiprecision type.
    std::cout << " expected   : " << sqrt(static_cast<mp_type>(5))/3U << std::endl;
    std::cout << " expected   : " << 0.74535599249992989880
    << std::endl;
    std::cout << " bit_error : "  << static_cast<unsigned long>(bit_error)  << std::endl;

    std::cout.precision(6);
    std::cout << "float_distance = " << boost::math::float_distance(0.74535599249992989880, d) << std::endl;

    
  }


} // int main()

/*`
The result of this program on a system with an eight-byte, 64-bit IEEE-754
conforming floating-point representation for `double` is:

 derivative : 0.745355992499951

 derivative : 0.745355992499943
 expected   : 0.74535599249993
 bit_error : 78

    derivative : 0.745355992499930
   expected   : 0.745355992499930
   bit_error : 0

The resulting bit error is 0. This means that the result of the derivative
calculation is bit-identical with the double representation of the expected result,
and this is the best result possible for the built-in type.

The derivative in this example has a known closed form. There are, however,
countless situations in numerical analysis (and not only for numerical deriva-
tives) for which the calculation at hand does not have a known closed-form
solution or for which the closed-form solution is highly inconvenient to use. In
such cases, this technique may be useful.

This example has shown how multiprecision can be used to add extra digits
to an ill-conditioned calculation that suffers from precision loss. When the result
of the multiprecision calculation is converted to a built-in type such as double,
the entire precision of the result in double is preserved.

 */

/*

  Description: Autorun "J:\Cpp\big_number\Debug\numerical_derivative_example.exe"
   derivative : 0.745355992499943
   expected   : 0.745355992499930
   expected   : 0.745355992499930
   bit_error : 78
  float_distance = 117.000
   derivative : 0.745355992499930
   expected   : 0.745355992499930
   expected   : 0.745355992499930
   bit_error : 0
  float_distance = 0.000000

 */


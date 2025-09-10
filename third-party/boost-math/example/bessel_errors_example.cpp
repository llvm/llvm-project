// Copyright Christopher Kormanyos 2013.
// Copyright Paul A. Bristow 2013.
// Copyright John Maddock 2013.

// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or
// copy at http://www.boost.org/LICENSE_1_0.txt).

#ifdef _MSC_VER
#  pragma warning (disable : 4512) // assignment operator could not be generated.
#  pragma warning (disable : 4996) // assignment operator could not be generated.
#endif

#include <iostream>
#include <limits>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <exception>

// Weisstein, Eric W. "Bessel Function Zeros." From MathWorld--A Wolfram Web Resource.
// http://mathworld.wolfram.com/BesselFunctionZeros.html
// Test values can be calculated using [@wolframalpha.com WolframAlpha]
// See also http://dlmf.nist.gov/10.21

//[bessel_errors_example_1

/*`[h5 Error messages from 'bad' input]

Another example demonstrates calculating zeros of the Bessel functions
showing the error messages from 'bad' input is handled by throwing exceptions.

To use the functions for finding zeros of the functions we need:
*/
  #include <boost/math/special_functions/bessel.hpp>
  #include <boost/math/special_functions/airy.hpp>

//] [/bessel_errors_example_1]

int main()
{
//[bessel_errors_example_2

/*`[tip It is always wise to place all code using Boost.Math inside try'n'catch blocks;
this will ensure that helpful error messages can be shown when exceptional conditions arise.]

Examples below show messages from several 'bad' arguments that throw a `domain_error` exception.
*/
  try
  { // Try a zero order v.
    float dodgy_root = boost::math::cyl_bessel_j_zero(0.F, 0);
    std::cout << "boost::math::cyl_bessel_j_zero(0.F, 0) " << dodgy_root << std::endl;
    // Thrown exception Error in function boost::math::cyl_bessel_j_zero<double>(double, int):
    // Requested the 0'th zero of J0, but the rank must be > 0 !
  }
  catch (std::exception& ex)
  {
    std::cout << "Thrown exception " << ex.what() << std::endl;
  }

/*`[note The type shown in the error message is the type [*after promotion],
using __precision_policy and __promotion_policy, from `float` to `double` in this case.]

In this example the promotion goes:

# Arguments are `float` and `int`.
# Treat `int` "as if" it were a `double`, so arguments are `float` and `double`.
# Common type is `double` - so that's the precision we want (and the type that will be returned).
# Evaluate internally as `double` for full `float` precision.

See full code for other examples that promote from `double` to `long double`.

Other examples of 'bad' inputs like infinity and NaN are below.
Some compiler warnings indicate that 'bad' values are detected at compile time.
*/

  try
  { // order v = inf
     std::cout << "boost::math::cyl_bessel_j_zero(inf, 1) " << std::endl;
     double inf = std::numeric_limits<double>::infinity();
     double inf_root = boost::math::cyl_bessel_j_zero(inf, 1);
     std::cout << "boost::math::cyl_bessel_j_zero(inf, 1) " << inf_root << std::endl;
     // Throw exception Error in function boost::math::cyl_bessel_j_zero<long double>(long double, unsigned):
     // Order argument is 1.#INF, but must be finite >= 0 !
  }
  catch (std::exception& ex)
  {
    std::cout << "Thrown exception " << ex.what() << std::endl;
  }

  try
  { // order v = NaN, rank m = 1
     std::cout << "boost::math::cyl_bessel_j_zero(nan, 1) " << std::endl;
     double nan = std::numeric_limits<double>::quiet_NaN();
     double nan_root = boost::math::cyl_bessel_j_zero(nan, 1);
     std::cout << "boost::math::cyl_bessel_j_zero(nan, 1) " << nan_root << std::endl;
     // Throw exception Error in function boost::math::cyl_bessel_j_zero<long double>(long double, unsigned):
     // Order argument is 1.#QNAN, but must be finite >= 0 !
  }
  catch (std::exception& ex)
  {
    std::cout << "Thrown exception " << ex.what() << std::endl;
  }

/*`The output from other examples are shown appended to the full code listing.
*/
//] [/bessel_errors_example_2]
  try
  {   // Try a zero rank m.
    std::cout << "boost::math::cyl_neumann_zero(0.0, 0) " << std::endl;
    double dodgy_root = boost::math::cyl_bessel_j_zero(0.0, 0);
    //  warning C4146: unary minus operator applied to unsigned type, result still unsigned.
    std::cout << "boost::math::cyl_neumann_zero(0.0, -1) " << dodgy_root << std::endl;
    //  boost::math::cyl_neumann_zero(0.0, -1) 6.74652e+009
    // This *should* fail because m is unreasonably large.

  }
  catch (std::exception& ex)
  {
    std::cout << "Thrown exception " << ex.what() << std::endl;
  }

  try
  { // m = inf
   std::cout << "boost::math::cyl_bessel_j_zero(0.0, inf) " << std::endl;
   double inf = std::numeric_limits<double>::infinity();
     double inf_root = boost::math::cyl_bessel_j_zero(0.0, inf);
     // warning C4244: 'argument' : conversion from 'double' to 'int', possible loss of data.
     std::cout << "boost::math::cyl_bessel_j_zero(0.0, inf) " << inf_root << std::endl;
     // Throw exception Error in function boost::math::cyl_bessel_j_zero<long double>(long double, int):
     // Requested the 0'th zero, but must be > 0 !

  }
  catch (std::exception& ex)
  {
    std::cout << "Thrown exception " << ex.what() << std::endl;
  }

  try
  { // m = NaN
     double nan = std::numeric_limits<double>::quiet_NaN();
     double nan_root = boost::math::airy_ai_zero<double>(nan);
     // warning C4244: 'argument' : conversion from 'double' to 'int', possible loss of data.
     std::cout << "boost::math::airy_ai_zero<double>(nan) " << nan_root << std::endl;
     // Thrown exception Error in function boost::math::airy_ai_zero<double>(double,double):
     // The requested rank of the zero is 0, but must be 1 or more !
  }
  catch (std::exception& ex)
  {
    std::cout << "Thrown exception " << ex.what() << std::endl;
  }
 } // int main()

/*
Output:

  Description: Autorun "J:\Cpp\big_number\Debug\bessel_errors_example.exe"
  Thrown exception Error in function boost::math::cyl_bessel_j_zero<double>(double, int): Requested the 0'th zero of J0, but the rank must be > 0 !
  boost::math::cyl_bessel_j_zero(inf, 1) 
  Thrown exception Error in function boost::math::cyl_bessel_j_zero<long double>(long double, int): Order argument is 1.#INF, but must be finite >= 0 !
  boost::math::cyl_bessel_j_zero(nan, 1) 
  Thrown exception Error in function boost::math::cyl_bessel_j_zero<long double>(long double, int): Order argument is 1.#QNAN, but must be finite >= 0 !
  boost::math::cyl_neumann_zero(0.0, 0) 
  Thrown exception Error in function boost::math::cyl_bessel_j_zero<long double>(long double, int): Requested the 0'th zero of J0, but the rank must be > 0 !
  boost::math::cyl_bessel_j_zero(0.0, inf) 
  Thrown exception Error in function boost::math::cyl_bessel_j_zero<long double>(long double, int): Requested the -2147483648'th zero, but the rank must be positive !
  Thrown exception Error in function boost::math::airy_ai_zero<double>(double,double): The requested rank of the zero is 0, but must be 1 or more !

 
*/


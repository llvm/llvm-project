// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Copyright Paul A. Bristow 2019.
// Copyright Christopher Kormanyos 2012.
// Copyright John Maddock 2012.

// This file is written to be included from a Quickbook .qbk document.
// It can be compiled by the C++ compiler, and run. Any output can
// also be added here as comment or included or pasted in elsewhere.
// Caution: this file contains Quickbook markup as well as code
// and comments: don't change any of the special comment markups!

#ifdef _MSC_VER
#pragma warning(disable : 4512) // assignment operator could not be generated.
#pragma warning(disable : 4996)
#endif

//[big_seventh_example_1

/*`[h5 Using Boost.Multiprecision `cpp_float` types for numerical calculations with higher precision than built-in `long double`.]

The Boost.Multiprecision library can be used for computations requiring precision
exceeding that of standard built-in types such as `float`, `double`
and `long double`. For extended-precision calculations, Boost.Multiprecision
supplies several template data types called `cpp_bin_float_`.

The number of decimal digits of precision is fixed at compile-time via template parameter.

To use these floating-point types and 
[@https://www.boost.org/doc/libs/release/libs/math/doc/html/constants.html Boost.Math collection of high-precision constants],
we need some includes:
*/

#include <boost/math/constants/constants.hpp>

#include <boost/multiprecision/cpp_bin_float.hpp>
// that includes some predefined typedefs that can be used thus:
// using boost::multiprecision::cpp_bin_float_quad;
// using boost::multiprecision::cpp_bin_float_50;
// using boost::multiprecision::cpp_bin_float_100;

#include <iostream>
#include <limits>
#include <type_traits>

/*` So now we can demonstrate with some trivial calculations:
*/

//] //[big_seventh_example_1]

void no_et()
{
  using namespace boost::multiprecision;

  std::cout.setf(std::ios_base::boolalpha);
  
   typedef number<backends::cpp_bin_float<113, backends::digit_base_2, void, std::int16_t, -16382, 16383>, et_on> cpp_bin_float_quad_et_on;
   typedef number<backends::cpp_bin_float<113, backends::digit_base_2, void, std::int16_t, -16382, 16383>, et_off> cpp_bin_float_quad_et_off;

   typedef number<backends::cpp_bin_float<113, backends::digit_base_2, void, std::int16_t, -16382, 16383>, et_off> cpp_bin_float_oct;


  cpp_bin_float_quad  x("42.");
  std::cout << "cpp_bin_float_quad x =  " << x << std::endl;

  cpp_bin_float_quad_et_on q("42.");

  std::cout << "std::is_same<cpp_bin_float_quad, cpp_bin_float_quad_et_off>::value is " << std::is_same<cpp_bin_float_quad, cpp_bin_float_quad_et_off>::value << std::endl; 
  std::cout << "std::is_same<cpp_bin_float_quad, cpp_bin_float_quad_et_on>::value is " << std::is_same<cpp_bin_float_quad, cpp_bin_float_quad_et_on>::value << std::endl; 

  std::cout << "cpp_bin_float_quad_et_on   q =  " << q << std::endl;
  cpp_bin_float_50   y("42.");  // typedef number<backends::cpp_bin_float<50> >  cpp_bin_float_50;

  std::cout << "cpp_bin_float_50   y =  " << y << std::endl;

  typedef number<backends::cpp_bin_float<50>, et_off > cpp_bin_float_50_no_et;
  typedef number<backends::cpp_bin_float<50>, et_on > cpp_bin_float_50_et;

  cpp_bin_float_50_no_et z("42.");  // typedef number<backends::cpp_bin_float<50> >  cpp_bin_float_50;

  std::cout << "cpp_bin_float_50_no_et   z =  " << z << std::endl;

  std::cout << " std::is_same<cpp_bin_float_50, cpp_bin_float_50_no_et>::value is " << std::is_same<cpp_bin_float_50, cpp_bin_float_50_no_et>::value << std::endl; 
  std::cout << " std::is_same<cpp_bin_float_50_et, cpp_bin_float_50_no_et>::value is " << std::is_same<cpp_bin_float_50_et, cpp_bin_float_50_no_et>::value << std::endl; 

} // void no_et()

int main()
{

  no_et();

  return 0;


   //[big_seventh_example_2
   /*`Using `typedef cpp_bin_float_50` hides the complexity of multiprecision,
allows us to define variables with 50 decimal digit precision just like built-in `double`.
*/
   using boost::multiprecision::cpp_bin_float_50;

   cpp_bin_float_50 seventh = cpp_bin_float_50(1) / 7; // 1 / 7

   /*`By default, output would only show the standard 6 decimal digits,
 so set precision to show all 50 significant digits, including any trailing zeros.
*/
   std::cout.precision(std::numeric_limits<cpp_bin_float_50>::digits10);
   std::cout << std::showpoint << std::endl; // Append any trailing zeros.
   std::cout << seventh << std::endl;
   /*`which outputs:

  0.14285714285714285714285714285714285714285714285714

We can also use __math_constants like [pi],
guaranteed to be initialized with the very last bit of precision (__ULP) for the floating-point type.
*/
   std::cout << "pi = " << boost::math::constants::pi<cpp_bin_float_50>() << std::endl;
   cpp_bin_float_50 circumference = boost::math::constants::pi<cpp_bin_float_50>() * 2 * seventh;
   std::cout << "c =  " << circumference << std::endl;

   /*`which outputs

  pi = 3.1415926535897932384626433832795028841971693993751

  c =  0.89759790102565521098932668093700082405633411410717
*/
   //]  [/big_seventh_example_2]

   //[big_seventh_example_3
   /*`So using `cpp_bin_float_50` looks like a simple 'drop-in' for the __fundamental_type like 'double',
but beware of loss of precision from construction or conversion from `double` or other lower precision types.
This is a mistake that is very easy to make, 
and very difficult to detect because the loss of precision is only visible after the 17th decimal digit.

We can show this by constructing from `double`, (avoiding the schoolboy-error `double d7 = 1 / 7;` giving zero!)
*/

   double d7 = 1. / 7; //
   std::cout << "d7 = " << d7 << std::endl;

   cpp_bin_float_50 seventh_0 = cpp_bin_float_50(1 / 7); // Avoid the schoolboy-error 1 / 7 == 0!)
   std::cout << "seventh_0 = " << seventh_0 << std::endl;
   // seventh_double0 = 0.0000000000000000000000000000000000000000000000000

   cpp_bin_float_50 seventh_double = cpp_bin_float_50(1. / 7);      // Construct from double!
   std::cout << "seventh_double = " << seventh_double << std::endl; // Boost.Multiprecision post-school error!
   // seventh_double = 0.14285714285714284921269268124888185411691665649414

   /*`Did you spot the mistake?  After the 17th decimal digit, result is random!

14285714285714 should be recurring.
*/

   cpp_bin_float_50 seventh_big(1); // 1
   seventh_big /= 7;
   std::cout << "seventh_big = " << seventh_big << std::endl; //
   // seventh_big     = 0.14285714285714285714285714285714285714285714285714
   /*`Note the recurring 14285714285714 pattern as expected.

As one would expect, the variable can be `const` (but sadly [*not yet `constexpr`]).
*/

   const cpp_bin_float_50 seventh_const(cpp_bin_float_50(1) / 7);
   std::cout << "seventh_const = " << seventh_const << std::endl; 
  // seventh_const = 0.14285714285714285714285714285714285714285714285714

/*`The full output is:
*/

//]  [/big_seventh_example_3

//[big_seventh_example_constexpr

// Sadly we cannot (yet) write:
// constexpr cpp_bin_float_50 any_constexpr(0);

// constexpr cpp_bin_float_50 seventh_constexpr (cpp_bin_float_50(1) / 7);
// std::cout << "seventh_constexpr = " << seventh_constexpr << std::endl; //
// constexpr cpp_bin_float_50 seventh_constexpr(seventh_const);

//]  [/big_seventh_example_constexpr

   return 0;
} // int main()

/*
//[big_seventh_example_output

0.14285714285714285714285714285714285714285714285714
pi = 3.1415926535897932384626433832795028841971693993751
c =  0.89759790102565521098932668093700082405633411410717
d7 = 0.14285714285714284921269268124888185411691665649414
seventh_0 = 0.0000000000000000000000000000000000000000000000000
seventh_double = 0.14285714285714284921269268124888185411691665649414
seventh_big = 0.14285714285714285714285714285714285714285714285714
seventh_const = 0.14285714285714285714285714285714285714285714285714

//] //[big_seventh_example_output]

*/

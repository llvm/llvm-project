// Copyright Paul A. Bristow 2015.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Note that this file contains Quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!

// Example of root finding using Boost.Multiprecision.

#ifndef BOOST_MATH_STANDALONE

#include <boost/math/tools/roots.hpp>
//using boost::math::policies::policy;
//using boost::math::tools::newton_raphson_iterate;
//using boost::math::tools::halley_iterate;
//using boost::math::tools::eps_tolerance; // Binary functor for specified number of bits.
//using boost::math::tools::bracket_and_solve_root;
//using boost::math::tools::toms748_solve;

#include <boost/math/special_functions/next.hpp> // For float_distance.
#include <boost/math/special_functions/pow.hpp>
#include <boost/math/constants/constants.hpp>

//[root_finding_multiprecision_include_1
#include <boost/multiprecision/cpp_bin_float.hpp> // For cpp_bin_float_50.
#include <boost/multiprecision/cpp_dec_float.hpp> // For cpp_dec_float_50.
#ifndef _MSC_VER  // float128 is not yet supported by Microsoft compiler at 2013.
#  include <boost/multiprecision/float128.hpp> // Requires libquadmath.
#endif
//] [/root_finding_multiprecision_include_1]

#include <iostream>
// using std::cout; using std::endl;
#include <iomanip>
// using std::setw; using std::setprecision;
#include <limits>
// using std::numeric_limits;
#include <tuple>
#include <utility> // pair, make_pair

// #define BUILTIN_POW_GUESS // define to use std::pow function to obtain a guess.

template <class T>
T cbrt_2deriv(T x)
{ // return cube root of x using 1st and 2nd derivatives and Halley.
  using namespace std;  // Help ADL of std functions.
  using namespace boost::math::tools; // For halley_iterate.

  // If T is not a binary floating-point type, for example, cpp_dec_float_50
  // then frexp may not be defined,
  // so it may be necessary to compute the guess using a built-in type,
  // probably quickest using double, but perhaps with float or long double.
  // Note that the range of exponent may be restricted by a built-in-type for guess.

  typedef long double guess_type;

#ifdef BUILTIN_POW_GUESS
  guess_type pow_guess = std::pow(static_cast<guess_type>(x), static_cast<guess_type>(1) / 3);
  T guess = pow_guess;
  T min = pow_guess /2;
  T max = pow_guess * 2;
#else
  int exponent;
  frexp(static_cast<guess_type>(x), &exponent); // Get exponent of z (ignore mantissa).
  T guess = ldexp(static_cast<guess_type>(1.), exponent / 3); // Rough guess is to divide the exponent by three.
  T min = ldexp(static_cast<guess_type>(1.) / 2, exponent / 3); // Minimum possible value is half our guess.
  T max = ldexp(static_cast<guess_type>(2.), exponent / 3); // Maximum possible value is twice our guess.
#endif

  int digits = std::numeric_limits<T>::digits / 2; // Half maximum possible binary digits accuracy for type T.
  const std::uintmax_t maxit = 20;
  std::uintmax_t it = maxit;
  T result = halley_iterate(cbrt_functor_2deriv<T>(x), guess, min, max, digits, it);
  // Can show how many iterations (updated by halley_iterate).
  // std::cout << "Iterations " << it << " (from max of "<< maxit << ")." << std::endl;
  return result;
} // cbrt_2deriv(x)


template <class T>
struct cbrt_functor_2deriv
{ // Functor returning both 1st and 2nd derivatives.
  cbrt_functor_2deriv(T const& to_find_root_of) : a(to_find_root_of)
  { // Constructor stores value to find root of, for example:
  }

  // using boost::math::tuple; // to return three values.
  std::tuple<T, T, T> operator()(T const& x)
  { 
    // Return both f(x) and f'(x) and f''(x).
    T fx = x*x*x - a;                     // Difference (estimate x^3 - value).
    // std::cout << "x = " << x << "\nfx = " << fx << std::endl;
    T dx = 3 * x*x;                       // 1st derivative = 3x^2.
    T d2x = 6 * x;                        // 2nd derivative = 6x.
    return std::make_tuple(fx, dx, d2x);  // 'return' fx, dx and d2x.
  }
private:
  T a;                                    // to be 'cube_rooted'.
}; // struct cbrt_functor_2deriv

template <int n, class T>
struct nth_functor_2deriv
{ // Functor returning both 1st and 2nd derivatives.

  nth_functor_2deriv(T const& to_find_root_of) : value(to_find_root_of)
  { /* Constructor stores value to find root of, for example: */ }

  // using std::tuple; // to return three values.
  std::tuple<T, T, T> operator()(T const& x)
  { 
    // Return both f(x) and f'(x) and f''(x).
    using boost::math::pow;
    T fx = pow<n>(x) - value;              // Difference (estimate x^3 - value).
    T dx = n * pow<n - 1>(x);              // 1st derivative = 5x^4.
    T d2x = n * (n - 1) * pow<n - 2 >(x);  // 2nd derivative = 20 x^3
    return std::make_tuple(fx, dx, d2x);   // 'return' fx, dx and d2x.
  }
private:
  T value;                                 // to be 'nth_rooted'.
}; // struct nth_functor_2deriv


template <int n, class T>
T nth_2deriv(T x)
{ 
  // return nth root of x using 1st and 2nd derivatives and Halley.
  using namespace std;  // Help ADL of std functions.
  using namespace boost::math; // For halley_iterate.

  int exponent;
  frexp(x, &exponent);                                 // Get exponent of z (ignore mantissa).
  T guess = ldexp(static_cast<T>(1.), exponent / n);   // Rough guess is to divide the exponent by three.
  T min = ldexp(static_cast<T>(0.5), exponent / n);    // Minimum possible value is half our guess.
  T max = ldexp(static_cast<T>(2.), exponent / n);     // Maximum possible value is twice our guess.

  int digits = std::numeric_limits<T>::digits / 2;     // Half maximum possible binary digits accuracy for type T.
  const std::uintmax_t maxit = 50;
  std::uintmax_t it = maxit;
  T result = halley_iterate(nth_functor_2deriv<n, T>(x), guess, min, max, digits, it);
  // Can show how many iterations (updated by halley_iterate).
  std::cout << it << " iterations (from max of " << maxit << ")" << std::endl;

  return result;
} // nth_2deriv(x)

//[root_finding_multiprecision_show_1

template <typename T>
T show_cube_root(T value)
{ // Demonstrate by printing the root using all definitely significant digits.
  std::cout.precision(std::numeric_limits<T>::digits10);
  T r = cbrt_2deriv(value);
  std::cout << "value = " << value << ", cube root =" << r << std::endl;
  return r;
}

//] [/root_finding_multiprecision_show_1]

int main()
{
  std::cout << "Multiprecision Root finding Example." << std::endl;
  // Show all possibly significant decimal digits.
  std::cout.precision(std::numeric_limits<double>::digits10);
  // or use   cout.precision(max_digits10 = 2 + std::numeric_limits<double>::digits * 3010/10000);
  //[root_finding_multiprecision_example_1
  using boost::multiprecision::cpp_dec_float_50; // decimal.
  using boost::multiprecision::cpp_bin_float_50; // binary.
#ifndef _MSC_VER  // Not supported by Microsoft compiler.
  using boost::multiprecision::float128;
#endif
  //] [/root_finding_multiprecision_example_1

  try
  { // Always use try'n'catch blocks with Boost.Math to get any error messages.
    // Increase the precision to 50 decimal digits using Boost.Multiprecision
//[root_finding_multiprecision_example_2

      std::cout.precision(std::numeric_limits<cpp_dec_float_50>::digits10);

      cpp_dec_float_50 two = 2; // 
      cpp_dec_float_50  r = cbrt_2deriv(two);
      std::cout << "cbrt(" << two << ") = " << r << std::endl;

      r = cbrt_2deriv(2.); // Passing a double, so ADL will compute a double precision result.
      std::cout << "cbrt(" << two << ") = " << r << std::endl;
      // cbrt(2) = 1.2599210498948731906665443602832965552806854248047 'wrong' from digits 17 onwards!
      r = cbrt_2deriv(static_cast<cpp_dec_float_50>(2.)); // Passing a cpp_dec_float_50, 
      // so will compute a cpp_dec_float_50 precision result.
      std::cout << "cbrt(" << two << ") = " << r << std::endl;
      r = cbrt_2deriv<cpp_dec_float_50>(2.); // Explicitly a cpp_dec_float_50, so will compute a cpp_dec_float_50 precision result.
      std::cout << "cbrt(" << two << ") = " << r << std::endl;
      // cpp_dec_float_50 1.2599210498948731647672106072782283505702514647015
//] [/root_finding_multiprecision_example_2
     //  N[2^(1/3), 50]  1.2599210498948731647672106072782283505702514647015

      //show_cube_root(2); // Integer parameter - Errors!
      //show_cube_root(2.F); // Float parameter - Warnings!
//[root_finding_multiprecision_example_3
      show_cube_root(2.);
      show_cube_root(2.L);
      show_cube_root(two);

//] [/root_finding_multiprecision_example_3

  }
  catch (const std::exception& e)
  { // Always useful to include try&catch blocks because default policies 
    // are to throw exceptions on arguments that cause errors like underflow & overflow. 
    // Lacking try&catch blocks, the program will abort without a message below,
    // which may give some helpful clues as to the cause of the exception.
    std::cout <<
      "\n""Message from thrown exception was:\n   " << e.what() << std::endl;
  }
  return 0;
} // int main()


/*

Description: Autorun "J:\Cpp\MathToolkit\test\Math_test\Release\root_finding_multiprecision.exe"
Multiprecision Root finding Example.
cbrt(2) = 1.2599210498948731647672106072782283505702514647015
cbrt(2) = 1.2599210498948731906665443602832965552806854248047
cbrt(2) = 1.2599210498948731647672106072782283505702514647015
cbrt(2) = 1.2599210498948731647672106072782283505702514647015
value = 2, cube root =1.25992104989487
value = 2, cube root =1.25992104989487
value = 2, cube root =1.2599210498948731647672106072782283505702514647015


*/

#endif // BOOST_MATH_STANDALONE

// Copyright Paul A. Bristow 2014, 2015.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Note that this file contains Quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!

// Example of finding nth root using 1st and 2nd derivatives of x^n.

#include <boost/math/tools/roots.hpp>
//using boost::math::policies::policy;
//using boost::math::tools::newton_raphson_iterate;
//using boost::math::tools::halley_iterate;
//using boost::math::tools::eps_tolerance; // Binary functor for specified number of bits.
//using boost::math::tools::bracket_and_solve_root;
//using boost::math::tools::toms748_solve;

#include <boost/math/special_functions/next.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/special_functions/pow.hpp>
#include <boost/math/constants/constants.hpp>

#include <boost/multiprecision/cpp_dec_float.hpp> // For cpp_dec_float_50.
#include <boost/multiprecision/cpp_bin_float.hpp> // using boost::multiprecision::cpp_bin_float_50;
#ifndef _MSC_VER  // float128 is not yet supported by Microsoft compiler at 2013.
# include <boost/multiprecision/float128.hpp>
#endif

#include <iostream>
// using std::cout; using std::endl;
#include <iomanip>
// using std::setw; using std::setprecision;
#include <limits>
using std::numeric_limits;
#include <tuple>
#include <utility> // pair, make_pair


//[root_finding_nth_functor_2deriv
template <int N, class T = double>
struct nth_functor_2deriv
{ // Functor returning both 1st and 2nd derivatives.
  static_assert(boost::is_integral<T>::value == false, "Only floating-point type types can be used!");
  static_assert((N > 0) == true, "root N must be > 0!");

  nth_functor_2deriv(T const& to_find_root_of) : a(to_find_root_of)
  { /* Constructor stores value a to find root of, for example: */ }

  // using boost::math::tuple; // to return three values.
  std::tuple<T, T, T> operator()(T const& x)
  { 
    // Return f(x), f'(x) and f''(x).
    using boost::math::pow;
    T fx = pow<N>(x) - a;                  // Difference (estimate x^n - a).
    T dx = N * pow<N - 1>(x);              // 1st derivative f'(x).
    T d2x = N * (N - 1) * pow<N - 2 >(x);  // 2nd derivative f''(x).

    return std::make_tuple(fx, dx, d2x);   // 'return' fx, dx and d2x.
  }
private:
  T a;                                     // to be 'nth_rooted'.
};

//] [/root_finding_nth_functor_2deriv]

/*
To show the progress, one might use this before the return statement above?
#ifdef BOOST_MATH_ROOT_DIAGNOSTIC
std::cout << " x = " << x << ", fx = " << fx << ", dx = " << dx << ", dx2 = " << d2x << std::endl;
#endif
*/

// If T is a floating-point type, might be quicker to compute the guess using a built-in type,
// probably quickest using double, but perhaps with float or long double, T.

// If T is a type for which frexp and ldexp are not defined,
// then it is necessary to compute the guess using a built-in type,
// probably quickest (but limited range) using double,
// but perhaps with float or long double, or a multiprecision T for the full range of T.
// typedef double guess_type; is used to specify the this.

//[root_finding_nth_function_2deriv

template <int N, class T = double>
T nth_2deriv(T x)
{ // return nth root of x using 1st and 2nd derivatives and Halley.

  using namespace std;  // Help ADL of std functions.
  using namespace boost::math::tools; // For halley_iterate.

  static_assert(boost::is_integral<T>::value == false, "Only floating-point type types can be used!");
  static_assert((N > 0) == true, "root N must be > 0!");
  static_assert((N > 1000) == false, "root N is too big!");

  typedef double guess_type; // double may restrict (exponent) range for a multiprecision T?

  int exponent;
  frexp(static_cast<guess_type>(x), &exponent);                 // Get exponent of z (ignore mantissa).
  T guess = ldexp(static_cast<guess_type>(1.), exponent / N);   // Rough guess is to divide the exponent by n.
  T min = ldexp(static_cast<guess_type>(1.) / 2, exponent / N); // Minimum possible value is half our guess.
  T max = ldexp(static_cast<guess_type>(2.), exponent / N);     // Maximum possible value is twice our guess.

  int digits = std::numeric_limits<T>::digits * 0.4;            // Accuracy triples with each step, so stop when
                                                                // slightly more than one third of the digits are correct.
  const std::uintmax_t maxit = 20;
  std::uintmax_t it = maxit;
  T result = halley_iterate(nth_functor_2deriv<N, T>(x), guess, min, max, digits, it);
  return result;
}

//] [/root_finding_nth_function_2deriv]


template <int N, typename T = double>
T show_nth_root(T value)
{ // Demonstrate by printing the nth root using all possibly significant digits.
  //std::cout.precision(std::numeric_limits<T>::max_digits10);
  // or use   cout.precision(max_digits10 = 2 + std::numeric_limits<double>::digits * 3010/10000);
  // Or guaranteed significant digits:
   std::cout.precision(std::numeric_limits<T>::digits10);

  T r = nth_2deriv<N>(value);
  std::cout << "Type " << typeid(T).name() << " value = " << value << ", " << N << "th root = " << r << std::endl;
  return r;
} // print_nth_root


int main()
{
  std::cout << "nth Root finding Example." << std::endl;
  using boost::multiprecision::cpp_dec_float_50; // decimal.
  using boost::multiprecision::cpp_bin_float_50; // binary.
#ifndef _MSC_VER  // Not supported by Microsoft compiler.
  using boost::multiprecision::float128; // Requires libquadmath
#endif
  try
  { // Always use try'n'catch blocks with Boost.Math to get any error messages.

//[root_finding_n_example_1
    double r1 = nth_2deriv<5, double>(2); // Integral value converted to double.

    // double r2 = nth_2deriv<5>(2); // Only floating-point type types can be used!

//] [/root_finding_n_example_1

    //show_nth_root<5, float>(2); // Integral value converted to float.
    //show_nth_root<5, float>(2.F); // 'initializing' : conversion from 'double' to 'float', possible loss of data

//[root_finding_n_example_2


    show_nth_root<5, double>(2.);
    show_nth_root<5, long double>(2.);
#ifndef _MSC_VER  // float128 is not supported by Microsoft compiler 2013.
    show_nth_root<5, float128>(2);
#endif
    show_nth_root<5, cpp_dec_float_50>(2); // dec
    show_nth_root<5, cpp_bin_float_50>(2); // bin
//] [/root_finding_n_example_2

    // show_nth_root<1000000>(2.); // Type double value = 2, 555th root = 1.00124969405651
    // Type double value = 2, 1000th root = 1.00069338746258
    // Type double value = 2, 1000000th root = 1.00000069314783
  }
  catch (const std::exception& e)
  { // Always useful to include try & catch blocks because default policies
    // are to throw exceptions on arguments that cause errors like underflow, overflow.
    // Lacking try & catch blocks, the program will abort without a message below,
    // which may give some helpful clues as to the cause of the exception.
    std::cout <<
      "\n""Message from thrown exception was:\n   " << e.what() << std::endl;
  }
  return 0;
} // int main()


/*
//[root_finding_example_output_1
 Using MSVC 2013

nth Root finding Example.
Type double value = 2, 5th root = 1.14869835499704
Type long double value = 2, 5th root = 1.14869835499704
Type class boost::multiprecision::number<class boost::multiprecision::backends::cpp_dec_float<50,int,void>,1> value = 2,
  5th root = 1.1486983549970350067986269467779275894438508890978
Type class boost::multiprecision::number<class boost::multiprecision::backends::cpp_bin_float<50,10,void,int,0,0>,0> value = 2,
  5th root = 1.1486983549970350067986269467779275894438508890978

//] [/root_finding_example_output_1]

//[root_finding_example_output_2

 Using GCC 4.91  (includes float_128 type)

 nth Root finding Example.
Type d value = 2, 5th root = 1.14869835499704
Type e value = 2, 5th root = 1.14869835499703501
Type N5boost14multiprecision6numberINS0_8backends16float128_backendELNS0_26expression_template_optionE0EEE value = 2, 5th root = 1.148698354997035006798626946777928
Type N5boost14multiprecision6numberINS0_8backends13cpp_dec_floatILj50EivEELNS0_26expression_template_optionE1EEE value = 2, 5th root = 1.1486983549970350067986269467779275894438508890978
Type N5boost14multiprecision6numberINS0_8backends13cpp_bin_floatILj50ELNS2_15digit_base_typeE10EviLi0ELi0EEELNS0_26expression_template_optionE0EEE value = 2, 5th root = 1.1486983549970350067986269467779275894438508890978

RUN SUCCESSFUL (total time: 63ms)

//] [/root_finding_example_output_2]
*/

/*
Throw out of range using GCC release mode :-(

 */

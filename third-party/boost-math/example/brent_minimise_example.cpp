//! \file
//! \brief Brent_minimise_example.cpp

// Copyright Paul A. Bristow 2015, 2018.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Note that this file contains Quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!

// For some diagnostic information:
//#define BOOST_MATH_INSTRUMENT
// If quadmath float128 is available:
//#define BOOST_HAVE_QUADMATH

// Example of finding minimum of a function with Brent's method.
//[brent_minimise_include_1
#include <boost/math/tools/minima.hpp>
//] [/brent_minimise_include_1]

#include <boost/math/special_functions/next.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/special_functions/pow.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/test/tools/floating_point_comparison.hpp> // For is_close_at)tolerance and is_small

//[brent_minimise_mp_include_0
#include <boost/multiprecision/cpp_dec_float.hpp> // For decimal boost::multiprecision::cpp_dec_float_50.
#include <boost/multiprecision/cpp_bin_float.hpp> // For binary boost::multiprecision::cpp_bin_float_50;
//] [/brent_minimise_mp_include_0]

//#ifndef _MSC_VER  // float128 is not yet supported by Microsoft compiler at 2018.
#ifdef BOOST_HAVE_QUADMATH  // Define only if GCC or Intel, and have quadmath.lib or .dll library available.
#  include <boost/multiprecision/float128.hpp>
#endif

#include <iostream>
// using std::cout; using std::endl;
#include <iomanip>
// using std::setw; using std::setprecision;
#include <limits>
using std::numeric_limits;
#include <tuple>
#include <utility> // pair, make_pair
#include <type_traits>
#include <typeinfo>

 //typedef boost::multiprecision::number<boost::multiprecision::cpp_dec_float<50>,
 //   boost::multiprecision::et_off>
 //   cpp_dec_float_50_et_off;
 //
 // typedef boost::multiprecision::number<boost::multiprecision::cpp_bin_float<50>,
 //   boost::multiprecision::et_off>
 //   cpp_bin_float_50_et_off;

// http://en.wikipedia.org/wiki/Brent%27s_method Brent's method

// An example of a function for which we want to find a minimum.
double f(double x)
{
  return (x + 3) * (x - 1) * (x - 1);
}

//[brent_minimise_double_functor
struct funcdouble
{
  double operator()(double const& x)
  {
    return (x + 3) * (x - 1) * (x - 1); // (x + 3)(x - 1)^2
  }
};
//] [/brent_minimise_double_functor]

//[brent_minimise_T_functor
struct func
{
  template <class T>
  T operator()(T const& x)
  {
    return (x + 3) * (x - 1) * (x - 1); // (x + 3)(x - 1)^2
  }
};
//] [/brent_minimise_T_functor]

//! Test if two values are close within a given tolerance.
template<typename FPT>
inline bool
is_close_to(FPT left, FPT right, FPT tolerance)
{
  return boost::math::fpc::close_at_tolerance<FPT>(tolerance) (left, right);
}

//[brent_minimise_close

//! Compare if value got is close to expected,
//! checking first if expected is very small
//! (to avoid divide by tiny or zero during comparison)
//! before comparing expect with value got.

template <class T>
bool is_close(T expect, T got, T tolerance)
{
  using boost::math::fpc::close_at_tolerance;
  using boost::math::fpc::is_small;
  using boost::math::fpc::FPC_STRONG;

  if (is_small<T>(expect, tolerance))
  {
    return is_small<T>(got, tolerance);
  }

  return close_at_tolerance<T>(tolerance, FPC_STRONG) (expect, got);
} // bool is_close(T expect, T got, T tolerance)

//] [/brent_minimise_close]

//[brent_minimise_T_show

//! Example template function to find and show minima.
//! \tparam T floating-point or fixed_point type.
template <class T>
void show_minima()
{
  using boost::math::tools::brent_find_minima;
  using std::sqrt;
  try
  { // Always use try'n'catch blocks with Boost.Math to ensure you get any error messages.

    int bits = std::numeric_limits<T>::digits/2; // Maximum is digits/2;
    std::streamsize prec = static_cast<int>(2 + sqrt((double)bits));  // Number of significant decimal digits.
    std::streamsize precision = std::cout.precision(prec); // Save and set.

    std::cout << "\n\nFor type: " << typeid(T).name()
      << ",\n  epsilon = " << std::numeric_limits<T>::epsilon()
      // << ", precision of " << bits << " bits"
      << ",\n  the maximum theoretical precision from Brent's minimization is "
      << sqrt(std::numeric_limits<T>::epsilon())
      << "\n  Displaying to std::numeric_limits<T>::digits10 " << prec << ", significant decimal digits."
      << std::endl;

    const std::uintmax_t maxit = 20;
    std::uintmax_t it = maxit;
    // Construct using string, not double, avoids loss of precision.
    //T bracket_min = static_cast<T>("-4");
    //T bracket_max = static_cast<T>("1.3333333333333333333333333333333333333333333333333");

    // Construction from double may cause loss of precision for multiprecision types like cpp_bin_float,
    // but brackets values are good enough for using Brent minimization.
    T bracket_min = static_cast<T>(-4);
    T bracket_max = static_cast<T>(1.3333333333333333333333333333333333333333333333333);

    std::pair<T, T> r = brent_find_minima<func, T>(func(), bracket_min, bracket_max, bits, it);

    std::cout << "  x at minimum = " << r.first << ", f(" << r.first << ") = " << r.second;
    if (it < maxit)
    {
      std::cout << ",\n  met " << bits << " bits precision" << ", after " << it << " iterations." << std::endl;
    }
    else
    {
      std::cout << ",\n  did NOT meet " << bits << " bits precision" << " after " << it << " iterations!" << std::endl;
    }
    // Check that result is that expected (compared to theoretical uncertainty).
    T uncertainty = sqrt(std::numeric_limits<T>::epsilon());
    std::cout << std::boolalpha << "x == 1 (compared to uncertainty " << uncertainty << ") is "
      << is_close(static_cast<T>(1), r.first, uncertainty) << std::endl;
    std::cout << std::boolalpha << "f(x) == (0 compared to uncertainty " << uncertainty << ") is "
      << is_close(static_cast<T>(0), r.second, uncertainty) << std::endl;
    // Problems with this using multiprecision with expression template on?
    std::cout.precision(precision);  // Restore.
  }
  catch (const std::exception& e)
  { // Always useful to include try & catch blocks because default policies
    // are to throw exceptions on arguments that cause errors like underflow, overflow.
    // Lacking try & catch blocks, the program will abort without a message below,
    // which may give some helpful clues as to the cause of the exception.
    std::cout <<
      "\n""Message from thrown exception was:\n   " << e.what() << std::endl;
  }
} // void show_minima()

//] [/brent_minimise_T_show]

int main()
{
  using boost::math::tools::brent_find_minima;
  using std::sqrt;
  std::cout << "Brent's minimisation examples." << std::endl;
  std::cout << std::boolalpha << std::endl;
  std::cout << std::showpoint << std::endl; // Show trailing zeros.

  // Tip - using
  // std::cout.precision(std::numeric_limits<T>::digits10);
  // during debugging is wise because it warns
  // if construction of multiprecision involves conversion from double
  // by finding random or zero digits after 17th decimal digit.

  // Specific type double - unlimited iterations (unwise?).
  {
    std::cout << "\nType double - unlimited iterations (unwise?)" << std::endl;
  //[brent_minimise_double_1
    const int double_bits = std::numeric_limits<double>::digits;
    std::pair<double, double> r = brent_find_minima(funcdouble(), -4., 4. / 3, double_bits);

    std::streamsize precision_1 = std::cout.precision(std::numeric_limits<double>::digits10);
    // Show all double precision decimal digits and trailing zeros.
    std::cout << "x at minimum = " << r.first
      << ", f(" << r.first << ") = " << r.second << std::endl;
    //] [/brent_minimise_double_1]
    std::cout << "x at minimum = " << (r.first - 1.) / r.first << std::endl;
    // x at minimum = 1.00000000112345, f(1.00000000112345) = 5.04852568272458e-018
    double uncertainty = sqrt(std::numeric_limits<double>::epsilon());
    std::cout << "Uncertainty sqrt(epsilon) =  " << uncertainty << std::endl;
    // sqrt(epsilon) =  1.49011611938477e-008
    // (epsilon is always > 0, so no need to take abs value).

    std::cout.precision(precision_1); // Restore.
  //[brent_minimise_double_1a

  using boost::math::fpc::close_at_tolerance;
  using boost::math::fpc::is_small;

  std::cout << "x = " << r.first << ", f(x) = " << r.second << std::endl;
  std::cout << std::boolalpha << "x == 1 (compared to uncertainty "
    << uncertainty << ") is " << is_close(1., r.first, uncertainty) << std::endl; // true
  std::cout << std::boolalpha << "f(x) == 0 (compared to uncertainty "
    << uncertainty << ") is " << is_close(0., r.second, uncertainty) << std::endl; // true
//] [/brent_minimise_double_1a]

  }
  std::cout << "\nType double with limited iterations." << std::endl;
  {
    const int bits = std::numeric_limits<double>::digits;
    // Specific type double - limit maxit to 20 iterations.
    std::cout << "Precision bits = " << bits << std::endl;
  //[brent_minimise_double_2
    const std::uintmax_t maxit = 20;
    std::uintmax_t it = maxit;
    std::pair<double, double> r = brent_find_minima(funcdouble(), -4., 4. / 3, bits, it);
    std::cout << "x at minimum = " << r.first << ", f(" << r.first << ") = " << r.second
      << " after " << it << " iterations. " << std::endl;
    //] [/brent_minimise_double_2]
      // x at minimum = 1.00000000112345, f(1.00000000112345) = 5.04852568272458e-018
//[brent_minimise_double_3
  std::streamsize prec = static_cast<int>(2 + sqrt((double)bits));  // Number of significant decimal digits.
  std::streamsize precision_3 = std::cout.precision(prec); // Save and set new precision.
  std::cout << "Showing " << bits << " bits "
    "precision with " << prec
    << " decimal digits from tolerance " << sqrt(std::numeric_limits<double>::epsilon())
    << std::endl;

  std::cout << "x at minimum = " << r.first
    << ", f(" << r.first << ") = " << r.second
    << " after " << it << " iterations. " << std::endl;
  std::cout.precision(precision_3); // Restore.
//] [/brent_minimise_double_3]
  // Showing 53 bits precision with 9 decimal digits from tolerance 1.49011611938477e-008
  //  x at minimum = 1, f(1) = 5.04852568e-018
  }

  std::cout << "\nType double with limited iterations and half double bits." << std::endl;
  {

//[brent_minimise_double_4
  const int bits_div_2 = std::numeric_limits<double>::digits / 2; // Half digits precision (effective maximum).
  double epsilon_2 = boost::math::pow<-(std::numeric_limits<double>::digits/2 - 1), double>(2);
  std::streamsize prec = static_cast<int>(2 + sqrt((double)bits_div_2));  // Number of significant decimal digits.

  std::cout << "Showing " << bits_div_2 << " bits precision with " << prec
    << " decimal digits from tolerance " << sqrt(epsilon_2)
    << std::endl;
  std::streamsize precision_4 = std::cout.precision(prec); // Save.
  const std::uintmax_t maxit = 20;
  std::uintmax_t it_4 = maxit;
  std::pair<double, double> r = brent_find_minima(funcdouble(), -4., 4. / 3, bits_div_2, it_4);
  std::cout << "x at minimum = " << r.first << ", f(" << r.first << ") = " << r.second << std::endl;
  std::cout << it_4 << " iterations. " << std::endl;
  std::cout.precision(precision_4); // Restore.

//] [/brent_minimise_double_4]
  }
  // x at minimum = 1, f(1) = 5.04852568e-018

  {
    std::cout << "\nType double with limited iterations and quarter double bits." << std::endl;
  //[brent_minimise_double_5
    const int bits_div_4 = std::numeric_limits<double>::digits / 4; // Quarter precision.
    double epsilon_4 = boost::math::pow<-(std::numeric_limits<double>::digits / 4 - 1), double>(2);
    std::streamsize prec = static_cast<int>(2 + sqrt((double)bits_div_4));  // Number of significant decimal digits.
    std::cout << "Showing " << bits_div_4 << " bits precision with " << prec
      << " decimal digits from tolerance " << sqrt(epsilon_4)
      << std::endl;
    std::streamsize precision_5 = std::cout.precision(prec); // Save & set.
    const std::uintmax_t maxit = 20;

    std::uintmax_t it_5 = maxit;
    std::pair<double, double> r = brent_find_minima(funcdouble(), -4., 4. / 3, bits_div_4, it_5);
    std::cout << "x at minimum = " << r.first << ", f(" << r.first << ") = " << r.second
    << ", after " << it_5 << " iterations. " << std::endl;
    std::cout.precision(precision_5); // Restore.

  //] [/brent_minimise_double_5]
  }

  // Showing 13 bits precision with 9 decimal digits from tolerance 0.015625
  // x at minimum = 0.9999776, f(0.9999776) = 2.0069572e-009
  //  7 iterations.

  {
    std::cout << "\nType long double with limited iterations and all long double bits." << std::endl;
//[brent_minimise_template_1
    std::streamsize precision_t1 = std::cout.precision(std::numeric_limits<long double>::digits10); // Save & set.
    long double bracket_min = -4.;
    long double bracket_max = 4. / 3;
    const int bits = std::numeric_limits<long double>::digits;
    const std::uintmax_t maxit = 20;
    std::uintmax_t it = maxit;

    std::pair<long double, long double> r = brent_find_minima(func(), bracket_min, bracket_max, bits, it);
    std::cout << "x at minimum = " << r.first << ", f(" << r.first << ") = " << r.second
      << ", after " << it << " iterations. " << std::endl;
    std::cout.precision(precision_t1);  // Restore.
//] [/brent_minimise_template_1]
  }

  // Show use of built-in type Template versions.
  // (Will not work if construct bracket min and max from string).

//[brent_minimise_template_fd
  show_minima<float>();
  show_minima<double>();
  show_minima<long double>();

 //] [/brent_minimise_template_fd]

//[brent_minimise_mp_include_1
#ifdef BOOST_HAVE_QUADMATH  // Defined only if GCC or Intel and have quadmath.lib or .dll library available.
  using boost::multiprecision::float128;
#endif
//] [/brent_minimise_mp_include_1]

//[brent_minimise_template_quad
#ifdef BOOST_HAVE_QUADMATH  // Defined only if GCC or Intel and have quadmath.lib or .dll library available.
  show_minima<float128>(); // Needs quadmath_snprintf, sqrtQ, fabsq that are in in quadmath library.
#endif
//] [/brent_minimise_template_quad

  // User-defined floating-point template.

//[brent_minimise_mp_typedefs
  using boost::multiprecision::cpp_bin_float_50; // binary multiprecision typedef.
  using boost::multiprecision::cpp_dec_float_50; // decimal multiprecision typedef.

  // One might also need typedefs like these to switch expression templates off and on (default is on).
  typedef boost::multiprecision::number<boost::multiprecision::cpp_bin_float<50>,
    boost::multiprecision::et_on>
    cpp_bin_float_50_et_on;  // et_on is default so is same as cpp_bin_float_50.

  typedef boost::multiprecision::number<boost::multiprecision::cpp_bin_float<50>,
    boost::multiprecision::et_off>
    cpp_bin_float_50_et_off;

  typedef boost::multiprecision::number<boost::multiprecision::cpp_dec_float<50>,
    boost::multiprecision::et_on> // et_on is default so is same as cpp_dec_float_50.
    cpp_dec_float_50_et_on;

  typedef boost::multiprecision::number<boost::multiprecision::cpp_dec_float<50>,
    boost::multiprecision::et_off>
    cpp_dec_float_50_et_off;
//] [/brent_minimise_mp_typedefs]

  { // binary ET on by default.
//[brent_minimise_mp_1
    std::cout.precision(std::numeric_limits<cpp_bin_float_50>::digits10);
    int bits = std::numeric_limits<cpp_bin_float_50>::digits / 2 - 2;
    cpp_bin_float_50 bracket_min = static_cast<cpp_bin_float_50>("-4");
    cpp_bin_float_50 bracket_max = static_cast<cpp_bin_float_50>("1.3333333333333333333333333333333333333333333333333");

    std::cout << "Bracketing " << bracket_min << " to " << bracket_max << std::endl;
    const std::uintmax_t maxit = 20;
    std::uintmax_t it = maxit; // Will be updated with actual iteration count.
    std::pair<cpp_bin_float_50, cpp_bin_float_50> r
      = brent_find_minima(func(), bracket_min, bracket_max, bits, it);

    std::cout << "x at minimum = " << r.first << ",\n f(" << r.first << ") = " << r.second
    // x at minimum = 1, f(1) = 5.04853e-018
      << ", after " << it << " iterations. " << std::endl;

    is_close_to(static_cast<cpp_bin_float_50>("1"), r.first, sqrt(std::numeric_limits<cpp_bin_float_50>::epsilon()));
    is_close_to(static_cast<cpp_bin_float_50>("0"), r.second, sqrt(std::numeric_limits<cpp_bin_float_50>::epsilon()));

//] [/brent_minimise_mp_1]

/*
//[brent_minimise_mp_output_1
For type  class boost::multiprecision::number<class boost::multiprecision::backends::cpp_bin_float<50,10,void,int,0,0>,1>,
epsilon = 5.3455294202e-51,
the maximum theoretical precision from Brent minimization is 7.311312755e-26
Displaying to std::numeric_limits<T>::digits10 11 significant decimal digits.
x at minimum = 1, f(1) = 5.6273022713e-58,
met 84 bits precision, after 14 iterations.
x == 1 (compared to uncertainty 7.311312755e-26) is true
f(x) == (0 compared to uncertainty 7.311312755e-26) is true
-4 1.3333333333333333333333333333333333333333333333333
x at minimum = 0.99999999999999999999999999998813903221565569205253,
f(0.99999999999999999999999999998813903221565569205253) =
  5.6273022712501408640665300316078046703496236636624e-58
14 iterations
//] [/brent_minimise_mp_output_1]
*/
//[brent_minimise_mp_2
    show_minima<cpp_bin_float_50_et_on>(); //
//] [/brent_minimise_mp_2]

/*
//[brent_minimise_mp_output_2
    For type  class boost::multiprecision::number<class boost::multiprecision::backends::cpp_bin_float<50, 10, void, int, 0, 0>, 1>,

//] [/brent_minimise_mp_output_1]
*/
  }

  { // binary ET on explicit
    std::cout.precision(std::numeric_limits<cpp_bin_float_50_et_on>::digits10);

    int bits = std::numeric_limits<cpp_bin_float_50_et_on>::digits / 2 - 2;

    cpp_bin_float_50_et_on bracket_min = static_cast<cpp_bin_float_50_et_on>("-4");
    cpp_bin_float_50_et_on bracket_max = static_cast<cpp_bin_float_50_et_on>("1.3333333333333333333333333333333333333333333333333");

    std::cout << bracket_min << " " << bracket_max << std::endl;
    const std::uintmax_t maxit = 20;
    std::uintmax_t it = maxit;
    std::pair<cpp_bin_float_50_et_on, cpp_bin_float_50_et_on> r = brent_find_minima(func(), bracket_min, bracket_max, bits, it);

    std::cout << "x at minimum = " << r.first << ", f(" << r.first << ") = " << r.second << std::endl;
    // x at minimum = 1, f(1) = 5.04853e-018
    std::cout << it << " iterations. " << std::endl;

    show_minima<cpp_bin_float_50_et_on>(); //

  }
  return 0;

  // Some examples of switching expression templates on and off follow.

  { // binary ET off
    std::cout.precision(std::numeric_limits<cpp_bin_float_50_et_off>::digits10);

    int bits = std::numeric_limits<cpp_bin_float_50_et_off>::digits / 2 - 2;
    cpp_bin_float_50_et_off bracket_min = static_cast<cpp_bin_float_50_et_off>("-4");
    cpp_bin_float_50_et_off bracket_max = static_cast<cpp_bin_float_50_et_off>("1.3333333333333333333333333333333333333333333333333");

    std::cout << bracket_min << " " << bracket_max << std::endl;
    const std::uintmax_t maxit = 20;
    std::uintmax_t it = maxit;
    std::pair<cpp_bin_float_50_et_off, cpp_bin_float_50_et_off> r = brent_find_minima(func(), bracket_min, bracket_max, bits, it);

    std::cout << "x at minimum = " << r.first << ", f(" << r.first << ") = " << r.second << std::endl;
    // x at minimum = 1, f(1) = 5.04853e-018
    std::cout << it << " iterations. " << std::endl;

    show_minima<cpp_bin_float_50_et_off>(); //
  }

  { // decimal ET on by default
    std::cout.precision(std::numeric_limits<cpp_dec_float_50>::digits10);

    int bits = std::numeric_limits<cpp_dec_float_50>::digits / 2 - 2;

    cpp_dec_float_50 bracket_min = static_cast<cpp_dec_float_50>("-4");
    cpp_dec_float_50 bracket_max = static_cast<cpp_dec_float_50>("1.3333333333333333333333333333333333333333333333333");

    std::cout << bracket_min << " " << bracket_max << std::endl;
    const std::uintmax_t maxit = 20;
    std::uintmax_t it = maxit;
    std::pair<cpp_dec_float_50, cpp_dec_float_50> r = brent_find_minima(func(), bracket_min, bracket_max, bits, it);

    std::cout << "x at minimum = " << r.first << ", f(" << r.first << ") = " << r.second << std::endl;
    // x at minimum = 1, f(1) = 5.04853e-018
    std::cout << it << " iterations. " << std::endl;

    show_minima<cpp_dec_float_50>();
  }

  { // decimal ET on
    std::cout.precision(std::numeric_limits<cpp_dec_float_50_et_on>::digits10);

    int bits = std::numeric_limits<cpp_dec_float_50_et_on>::digits / 2 - 2;

    cpp_dec_float_50_et_on bracket_min = static_cast<cpp_dec_float_50_et_on>("-4");
    cpp_dec_float_50_et_on bracket_max = static_cast<cpp_dec_float_50_et_on>("1.3333333333333333333333333333333333333333333333333");
    std::cout << bracket_min << " " << bracket_max << std::endl;
    const std::uintmax_t maxit = 20;
    std::uintmax_t it = maxit;
    std::pair<cpp_dec_float_50_et_on, cpp_dec_float_50_et_on> r = brent_find_minima(func(), bracket_min, bracket_max, bits, it);

    std::cout << "x at minimum = " << r.first << ", f(" << r.first << ") = " << r.second << std::endl;
    // x at minimum = 1, f(1) = 5.04853e-018
    std::cout << it << " iterations. " << std::endl;

    show_minima<cpp_dec_float_50_et_on>();

  }

  { // decimal ET off
    std::cout.precision(std::numeric_limits<cpp_dec_float_50_et_off>::digits10);

    int bits = std::numeric_limits<cpp_dec_float_50_et_off>::digits / 2 - 2;

    cpp_dec_float_50_et_off bracket_min = static_cast<cpp_dec_float_50_et_off>("-4");
    cpp_dec_float_50_et_off bracket_max = static_cast<cpp_dec_float_50_et_off>("1.3333333333333333333333333333333333333333333333333");

    std::cout << bracket_min << " " << bracket_max << std::endl;
    const std::uintmax_t maxit = 20;
    std::uintmax_t it = maxit;
    std::pair<cpp_dec_float_50_et_off, cpp_dec_float_50_et_off> r = brent_find_minima(func(), bracket_min, bracket_max, bits, it);

    std::cout << "x at minimum = " << r.first << ", f(" << r.first << ") = " << r.second << std::endl;
    // x at minimum = 1, f(1) = 5.04853e-018
    std::cout << it << " iterations. " << std::endl;

    show_minima<cpp_dec_float_50_et_off>();
  }

  return 0;
} // int main()


/*

Typical output MSVC 15.7.3

brent_minimise_example.cpp
Generating code
7 of 2746 functions ( 0.3%) were compiled, the rest were copied from previous compilation.
0 functions were new in current compilation
1 functions had inline decision re-evaluated but remain unchanged
Finished generating code
brent_minimise_example.vcxproj -> J:\Cpp\MathToolkit\test\Math_test\Release\brent_minimise_example.exe
Autorun "J:\Cpp\MathToolkit\test\Math_test\Release\brent_minimise_example.exe"
Brent's minimisation examples.



Type double - unlimited iterations (unwise?)
x at minimum = 1.00000000112345, f(1.00000000112345) = 5.04852568272458e-18
x at minimum = 1.12344622367552e-09
Uncertainty sqrt(epsilon) =  1.49011611938477e-08
x = 1.00000, f(x) = 5.04853e-18
x == 1 (compared to uncertainty 1.49012e-08) is true
f(x) == 0 (compared to uncertainty 1.49012e-08) is true

Type double with limited iterations.
Precision bits = 53
x at minimum = 1.00000, f(1.00000) = 5.04853e-18 after 10 iterations.
Showing 53 bits precision with 9 decimal digits from tolerance 1.49011612e-08
x at minimum = 1.00000000, f(1.00000000) = 5.04852568e-18 after 10 iterations.

Type double with limited iterations and half double bits.
Showing 26 bits precision with 7 decimal digits from tolerance 0.000172633
x at minimum = 1.000000, f(1.000000) = 5.048526e-18
10 iterations.

Type double with limited iterations and quarter double bits.
Showing 13 bits precision with 5 decimal digits from tolerance 0.0156250
x at minimum = 0.99998, f(0.99998) = 2.0070e-09, after 7 iterations.

Type long double with limited iterations and all long double bits.
x at minimum = 1.00000000112345, f(1.00000000112345) = 5.04852568272458e-18, after 10 iterations.


For type: float,
epsilon = 1.1921e-07,
the maximum theoretical precision from Brent's minimization is 0.00034527
Displaying to std::numeric_limits<T>::digits10 5, significant decimal digits.
x at minimum = 1.0002, f(1.0002) = 1.9017e-07,
met 12 bits precision, after 7 iterations.
x == 1 (compared to uncertainty 0.00034527) is true
f(x) == (0 compared to uncertainty 0.00034527) is true


For type: double,
epsilon = 2.220446e-16,
the maximum theoretical precision from Brent's minimization is 1.490116e-08
Displaying to std::numeric_limits<T>::digits10 7, significant decimal digits.
x at minimum = 1.000000, f(1.000000) = 5.048526e-18,
met 26 bits precision, after 10 iterations.
x == 1 (compared to uncertainty 1.490116e-08) is true
f(x) == (0 compared to uncertainty 1.490116e-08) is true


For type: long double,
epsilon = 2.220446e-16,
the maximum theoretical precision from Brent's minimization is 1.490116e-08
Displaying to std::numeric_limits<T>::digits10 7, significant decimal digits.
x at minimum = 1.000000, f(1.000000) = 5.048526e-18,
met 26 bits precision, after 10 iterations.
x == 1 (compared to uncertainty 1.490116e-08) is true
f(x) == (0 compared to uncertainty 1.490116e-08) is true
Bracketing -4.0000000000000000000000000000000000000000000000000 to 1.3333333333333333333333333333333333333333333333333
x at minimum = 0.99999999999999999999999999998813903221565569205253,
f(0.99999999999999999999999999998813903221565569205253) = 5.6273022712501408640665300316078046703496236636624e-58, after 14 iterations.


For type: class boost::multiprecision::number<class boost::multiprecision::backends::cpp_bin_float<50,10,void,int,0,0>,1>,
epsilon = 5.3455294202e-51,
the maximum theoretical precision from Brent's minimization is 7.3113127550e-26
Displaying to std::numeric_limits<T>::digits10 11, significant decimal digits.
x at minimum = 1.0000000000, f(1.0000000000) = 5.6273022713e-58,
met 84 bits precision, after 14 iterations.
x == 1 (compared to uncertainty 7.3113127550e-26) is true
f(x) == (0 compared to uncertainty 7.3113127550e-26) is true
-4.0000000000000000000000000000000000000000000000000 1.3333333333333333333333333333333333333333333333333
x at minimum = 0.99999999999999999999999999998813903221565569205253, f(0.99999999999999999999999999998813903221565569205253) = 5.6273022712501408640665300316078046703496236636624e-58
14 iterations.


For type: class boost::multiprecision::number<class boost::multiprecision::backends::cpp_bin_float<50,10,void,int,0,0>,1>,
epsilon = 5.3455294202e-51,
the maximum theoretical precision from Brent's minimization is 7.3113127550e-26
Displaying to std::numeric_limits<T>::digits10 11, significant decimal digits.
x at minimum = 1.0000000000, f(1.0000000000) = 5.6273022713e-58,
met 84 bits precision, after 14 iterations.
x == 1 (compared to uncertainty 7.3113127550e-26) is true
f(x) == (0 compared to uncertainty 7.3113127550e-26) is true


============================================================================================================

 // GCC 7.2.0 with quadmath

Brent's minimisation examples.

Type double - unlimited iterations (unwise?)
x at minimum = 1.00000000112345, f(1.00000000112345) = 5.04852568272458e-018
x at minimum = 1.12344622367552e-009
Uncertainty sqrt(epsilon) =  1.49011611938477e-008
x = 1.00000, f(x) = 5.04853e-018
x == 1 (compared to uncertainty 1.49012e-008) is true
f(x) == 0 (compared to uncertainty 1.49012e-008) is true

Type double with limited iterations.
Precision bits = 53
x at minimum = 1.00000, f(1.00000) = 5.04853e-018 after 10 iterations.
Showing 53 bits precision with 9 decimal digits from tolerance 1.49011612e-008
x at minimum = 1.00000000, f(1.00000000) = 5.04852568e-018 after 10 iterations.

Type double with limited iterations and half double bits.
Showing 26 bits precision with 7 decimal digits from tolerance 0.000172633
x at minimum = 1.000000, f(1.000000) = 5.048526e-018
10 iterations.

Type double with limited iterations and quarter double bits.
Showing 13 bits precision with 5 decimal digits from tolerance 0.0156250
x at minimum = 0.99998, f(0.99998) = 2.0070e-009, after 7 iterations.

Type long double with limited iterations and all long double bits.
x at minimum = 1.00000000000137302, f(1.00000000000137302) = 7.54079013697311930e-024, after 10 iterations.


For type: f,
epsilon = 1.1921e-007,
the maximum theoretical precision from Brent's minimization is 0.00034527
Displaying to std::numeric_limits<T>::digits10 5, significant decimal digits.
x at minimum = 1.0002, f(1.0002) = 1.9017e-007,
met 12 bits precision, after 7 iterations.
x == 1 (compared to uncertainty 0.00034527) is true
f(x) == (0 compared to uncertainty 0.00034527) is true


For type: d,
epsilon = 2.220446e-016,
the maximum theoretical precision from Brent's minimization is 1.490116e-008
Displaying to std::numeric_limits<T>::digits10 7, significant decimal digits.
x at minimum = 1.000000, f(1.000000) = 5.048526e-018,
met 26 bits precision, after 10 iterations.
x == 1 (compared to uncertainty 1.490116e-008) is true
f(x) == (0 compared to uncertainty 1.490116e-008) is true


For type: e,
epsilon = 1.084202e-019,
the maximum theoretical precision from Brent's minimization is 3.292723e-010
Displaying to std::numeric_limits<T>::digits10 7, significant decimal digits.
x at minimum = 1.000000, f(1.000000) = 7.540790e-024,
met 32 bits precision, after 10 iterations.
x == 1 (compared to uncertainty 3.292723e-010) is true
f(x) == (0 compared to uncertainty 3.292723e-010) is true


For type: N5boost14multiprecision6numberINS0_8backends16float128_backendELNS0_26expression_template_optionE0EEE,
epsilon = 1.92592994e-34,
the maximum theoretical precision from Brent's minimization is 1.38777878e-17
Displaying to std::numeric_limits<T>::digits10 9, significant decimal digits.
x at minimum = 1.00000000, f(1.00000000) = 1.48695468e-43,
met 56 bits precision, after 12 iterations.
x == 1 (compared to uncertainty 1.38777878e-17) is true
f(x) == (0 compared to uncertainty 1.38777878e-17) is true
Bracketing -4.0000000000000000000000000000000000000000000000000 to 1.3333333333333333333333333333333333333333333333333
x at minimum = 0.99999999999999999999999999998813903221565569205253,
f(0.99999999999999999999999999998813903221565569205253) = 5.6273022712501408640665300316078046703496236636624e-58, after 14 iterations.


For type: N5boost14multiprecision6numberINS0_8backends13cpp_bin_floatILj50ELNS2_15digit_base_typeE10EviLi0ELi0EEELNS0_26expression_template_optionE1EEE,
epsilon = 5.3455294202e-51,
the maximum theoretical precision from Brent's minimization is 7.3113127550e-26
Displaying to std::numeric_limits<T>::digits10 11, significant decimal digits.
x at minimum = 1.0000000000, f(1.0000000000) = 5.6273022713e-58,
met 84 bits precision, after 14 iterations.
x == 1 (compared to uncertainty 7.3113127550e-26) is true
f(x) == (0 compared to uncertainty 7.3113127550e-26) is true
-4.0000000000000000000000000000000000000000000000000 1.3333333333333333333333333333333333333333333333333
x at minimum = 0.99999999999999999999999999998813903221565569205253, f(0.99999999999999999999999999998813903221565569205253) = 5.6273022712501408640665300316078046703496236636624e-58
14 iterations.


For type: N5boost14multiprecision6numberINS0_8backends13cpp_bin_floatILj50ELNS2_15digit_base_typeE10EviLi0ELi0EEELNS0_26expression_template_optionE1EEE,
epsilon = 5.3455294202e-51,
the maximum theoretical precision from Brent's minimization is 7.3113127550e-26
Displaying to std::numeric_limits<T>::digits10 11, significant decimal digits.
x at minimum = 1.0000000000, f(1.0000000000) = 5.6273022713e-58,
met 84 bits precision, after 14 iterations.
x == 1 (compared to uncertainty 7.3113127550e-26) is true
f(x) == (0 compared to uncertainty 7.3113127550e-26) is true

*/

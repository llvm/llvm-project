//  Copyright John Maddock 2016
//  Copyright Christopher Kormanyos 2016.
//  Copyright Paul A. Bristow 2016.

//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Contains Quickbook snippets as C++ comments - do not remove.

// http://gcc.gnu.org/onlinedocs/libquadmath/ GCC Quad-Precision Math Library
// https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format

// https://gcc.gnu.org/onlinedocs/gcc/C_002b_002b-Dialect-Options.html#C_002b_002b-Dialect-Options GNU 3.5 Options Controlling C++ Dialect
// https://gcc.gnu.org/onlinedocs/gcc/C-Dialect-Options.html#C-Dialect-Options 3.4 Options Controlling C Dialect

//[float128_includes_1

#include <boost/cstdfloat.hpp> // For float_64_t, float128_t. Must be first include!
//#include <boost/config.hpp>
#include <boost/multiprecision/float128.hpp>
#include <boost/math/special_functions.hpp> // For gamma function.
#include <boost/math/constants/constants.hpp> // For constants pi, e ...
#include <typeinfo> //

#include <cmath>  // for pow function.

// #include <quadmath.h>
// C:\program files\gcc-6-win64\lib\gcc\x86_64-w64-mingw32\6.1.1\include\quadmath.h

// i:\modular-boost\boost\multiprecision\float128.hpp|210|  undefined reference to `quadmath_snprintf'.

//] [/float128_includes_1]

//[float128_dialect_1
/*`To make float128 available it is vital to get the dialect and options on the command line correct.

Quad type is forbidden by all the strict C++ standards, so using or adding -std=c++11 and later standards will prevent its use.
so explicitly use -std=gnu++11, 1y, 14, 17, or 1z or ...

For GCC 6.1.1, for example, the default is if no C++ language dialect options are given, is -std=gnu++14.

See https://gcc.gnu.org/onlinedocs/gcc/C-Dialect-Options.html#C-Dialect-Options
https://gcc.gnu.org/onlinedocs/gcc/Standards.html#Standards 2 Language Standards Supported by GCC

 g++.exe -Wall -fexceptions -std=gnu++17 -g -fext-numeric-literals -fpermissive -lquadmath
  -II:\modular-boost\libs\math\include -Ii:\modular-boost -c J:\Cpp\float128\float128\float128_example.cpp -o obj\Debug\float128_example.o

Requires GCC linker option -lquadmath

If this is missing, then get errors like:

  \modular-boost\boost\multiprecision\float128.hpp|210|undefined reference to `quadmath_snprintf'|
  \modular-boost\boost\multiprecision\float128.hpp|351|undefined reference to `sqrtq'|

Requires compile option

  -fext-numeric-literals

If missing, then get errors like:

\modular-boost\libs\math\include/boost/math/cstdfloat/cstdfloat_types.hpp:229:43: error: unable to find numeric literal operator 'operator""Q'

A successful build log was:

  g++.exe -Wall -std=c++11 -fexceptions -std=gnu++17 -g -fext-numeric-literals -II:\modular-boost\libs\math\include -Ii:\modular-boost -c J:\Cpp\float128\float128\float128_example.cpp -o obj\Debug\float128_example.o
  g++.exe  -o bin\Debug\float128.exe obj\Debug\float128_example.o  -lquadmath
*/

//] [/float128_dialect_1]

void show_versions(std::string title)
{
  std::cout << title << std::endl;

  std::cout << "Platform: " << BOOST_PLATFORM << '\n'
    << "Compiler: " << BOOST_COMPILER << '\n'
    << "STL     : " << BOOST_STDLIB << '\n'
    << "Boost   : " << BOOST_VERSION / 100000 << "."
    << BOOST_VERSION / 100 % 1000 << "."
    << BOOST_VERSION % 100
    << std::endl;
#ifdef _MSC_VER
  std::cout << "_MSC_FULL_VER = " << _MSC_FULL_VER << std::endl; // VS 2015 190023026
#if defined _M_IX86
  std::cout << "(x86)" << std::endl;
#endif
#if defined _M_X64
  std::cout << " (x64)" << std::endl;
#endif
#if defined _M_IA64
  std::cout << " (Itanium)" << std::endl;
#endif
  // Something very wrong if more than one is defined (so show them in all just in case)!
#endif // _MSC_VER
#ifdef __GNUC__
//PRINT_MACRO(__GNUC__);
//PRINT_MACRO(__GNUC_MINOR__);
//PRINT_MACRO(__GNUC_PATCH__);
std::cout << "GCC " << __VERSION__  << std::endl;
//PRINT_MACRO(LONG_MAX);
#endif // __GNUC__
  return;
} // void show_version(std::string title)

int main()
{
  try
  {

//[float128_example_3
// Always use try'n'catch blocks to ensure any error messages are displayed.
//`Ensure that all possibly significant digits (17) including trailing zeros are shown.

    std::cout.precision(std::numeric_limits<boost::float64_t>::max_digits10);
    std::cout.setf(std::ios::showpoint); // Show all significant trailing zeros.
 //] [/ float128_example_3]

#ifdef BOOST_FLOAT128_C
  std::cout << "Floating-point type boost::float128_t is available." << std::endl;
    std::cout << "  std::numeric_limits<boost::float128_t>::digits10 == "
    << std::numeric_limits<boost::float128_t>::digits10 << std::endl;
  std::cout << "  std::numeric_limits<boost::float128_t>::max_digits10 == "
    << std::numeric_limits<boost::float128_t>::max_digits10 << std::endl;
#else
  std::cout << "Floating-point type boost::float128_t is NOT available." << std::endl;
#endif

  show_versions("");

  using boost::multiprecision::float128;  // Wraps, for example, __float128 or _Quad.
  // or
  //using namespace boost::multiprecision;

  std::cout.precision(std::numeric_limits<float128>::max_digits10);  // Show all potentially meaningful digits.
  std::cout.setf(std::ios::showpoint); // Show all significant trailing zeros.

  // float128 pi0 = boost::math::constants::pi(); // Compile fails - need to specify a type for the constant!

  float128 pi1 = boost::math::constants::pi<float128>();  // Returns a constant of type float128.
  std::cout << sqrt(pi1) << std::endl; // 1.77245385090551602729816748334114514

  float128 pi2 = boost::math::constants::pi<__float128>(); // Constant of type __float128 gets converted to float128 on the assignment.
  std::cout << sqrt(pi2) << std::endl; // 1.77245385090551602729816748334114514

  // DIY decimal digit literal constant, with suffix Q.
  float128 pi3 = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348Q;
  std::cout << sqrt(pi3) << std::endl; // 1.77245385090551602729816748334114514

  // Compare to ready-rolled sqrt(pi) constant from Boost.Math:
  std::cout << boost::math::constants::root_pi<float128>() << std::endl; // 1.77245385090551602729816748334114514

   // DIY decimal digit literal constant, without suffix Q, suffering seventeen silent digits loss of precision!
  float128 pi4 = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348;
  std::cout << sqrt(pi4) << std::endl; // 1.77245385090551599275151910313924857

  // float128 variables constructed from a quad-type literal can be declared constexpr if required:

#ifndef BOOST_NO_CXX11_CONSTEXPR
  constexpr float128 pi_constexpr = 3.1415926535897932384626433832795028841971693993751058Q;
#endif
  std::cout << pi_constexpr << std::endl; // 3.14159265358979323846264338327950280

  // But sadly functions like sqrt are not yet available constexpr for float128.

  // constexpr float128 root_pi_constexpr = sqrt(pi_constexpr);  // Fails - not constexpr (yet).
  // constexpr float128 root_pi_constexpr = std::sqrt(pi_constexpr);  // Fails - no known conversion for argument 1 from 'const float128'.
  // constexpr float128 root_pi_constexpr = sqrt(pi_constexpr); // Call to non-constexpr
  // constexpr float128 root_pi_constexpr = boost::math::constants::root_pi(); // Missing type for constant.

  // Best current way to get a constexpr is to use a Boost.Math constant if one is available.
  constexpr float128 root_pi_constexpr = boost::math::constants::root_pi<float128>();
  std::cout << root_pi_constexpr << std::endl; // 1.77245385090551602729816748334114514

  // Note that casts within the sqrt call are NOT NEEDED (nor allowed),
  // since all the variables are the correct type to begin with.
  // std::cout << sqrt<float128>(pi3) << std::endl;
  // But note examples of catastrophic (but hard to see) loss of precision below.

  // Note also that the library functions, here sqrt, is NOT defined using std::sqrt,
  // so that the correct overload is found using Argument Dependent LookUp (ADL).

  float128 ee = boost::math::constants::e<float128>();
   std::cout << ee << std::endl;  // 2.71828182845904523536028747135266231

  float128 e1 = exp(1.Q); // Note argument to exp is type float128.
  std::cout << e1 << std::endl; // 2.71828182845904523536028747135266231

  // Beware - it is all too easy to silently get a much lower precision by mistake.

  float128 e1d = exp(1.); // Caution - only double 17 decimal digits precision!
  std::cout << e1d << std::endl; // 2.71828182845904509079559829842764884

  float128 e1i = exp(1); // Caution int promoted to double so only 17 decimal digits precision!
  std::cout << e1i << std::endl; // 2.71828182845904509079559829842764884

  float f1 = 1.F;
  float128 e1f = exp(f1); // Caution float so only 6 decimal digits precision out of 36!
  std::cout << e1f << std::endl; // 2.71828174591064453125000000000000000

  // In all these cases you get what you asked for and not what you expected or wanted.

  // Casting is essential if you start with a lower precision type.

  float128 e1q = exp(static_cast<float128>(f1)); // Full 36 decimal digits precision!
  std::cout << e1q << std::endl; // 2.71828182845904523536028747135266231

  float128 e1qc = exp((float128)f1); // Full 36 decimal digits precision!
  std::cout << e1qc << std::endl; // 2.71828182845904523536028747135266231

  float128 e1qcc = exp(float128(f1)); // Full 36 decimal digits precision!
  std::cout << e1qcc << std::endl; // 2.71828182845904523536028747135266231

  //float128 e1q = exp<float128>(1.); // Compile fails.
  // std::cout << e1q << std::endl; //

// http://en.cppreference.com/w/cpp/language/typeid
// The name()is implementation-dependent mangled, and may not be able to be output.
// The example showing output using one of the implementations where type_info::name prints full type names;
// filter through c++filt -t if using gcc or similar.

//[float128_type_info
const std::type_info& tifu128 = typeid(__float128); // OK.
//std::cout << tifu128.name() << std::endl; // On GCC, aborts (because not printable string).
//std::cout << typeid(__float128).name() << std::endl; // Aborts -
//  string name cannot be output.

const std::type_info& tif128 = typeid(float128); // OK.
std::cout << tif128.name() << std::endl; // OK.
std::cout << typeid(float128).name() << std::endl; // OK.

const std::type_info& tpi = typeid(pi1); // OK using GCC 6.1.1.
// (from GCC 5 according to http://gcc.gnu.org/bugzilla/show_bug.cgi?id=43622)
std::cout << tpi.name() << std::endl; // OK, Output implementation-dependent mangled name:

// N5boost14multiprecision6numberINS0_8backends16float128_backendELNS0_26expression_template_optionE0EEE

//] [/float128_type_info]

  }
  catch (std::exception ex)
  { // Display details about why any exceptions are thrown.
    std::cout << "Thrown exception " << ex.what() << std::endl;
  }
} // int main()

/*
[float128_output

-std=c++11 or -std=c++17 don't work

Floating-point type boost::float128_t is NOT available.

Platform: Win32
Compiler: GNU C++ version 6.1.1 20160609
STL     : GNU libstdc++ version 20160609
Boost   : 1.62.0
GCC 6.1.1 20160609


Added -fext-numeric-literals to

-std=gnu++11 -fext-numeric-literals -lquadmath

Floating-point type boost::float128_t is available.
  std::numeric_limits<boost::float128_t>::digits10 == 33
  std::numeric_limits<boost::float128_t>::max_digits10 == 36

Platform: Win32
Compiler: GNU C++ version 6.1.1 20160609
STL     : GNU libstdc++ version 20160609
Boost   : 1.62.0
GCC 6.1.1 20160609
1.77245385090551602729816748334114514
1.77245385090551602729816748334114514
1.77245385090551602729816748334114514
1.77245385090551602729816748334114514
N5boost14multiprecision6numberINS0_8backends16float128_backendELNS0_26expression_template_optionE0EEE
N5boost14multiprecision6numberINS0_8backends16float128_backendELNS0_26expression_template_optionE0EEE
N5boost14multiprecision6numberINS0_8backends16float128_backendELNS0_26expression_template_optionE0EEE

Process returned 0 (0x0)   execution time : 0.033 s
Press any key to continue.



//] [/float128_output]

*/

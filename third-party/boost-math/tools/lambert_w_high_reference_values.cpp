// \modular-boost\libs\math\test\lambert_w_high_reference_values.cpp

// Copyright Paul A. Bristow 2017.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Write a C++ file \lambert_w_mp_hi_values.ipp
// containing arrays of z arguments and 100 decimal digit precision lambert_w0(z) reference values.
// These can be used in tests of precision of less-precise types like
// built-in float, double, long double and quad and cpp_dec_float_50.

// These cover the range from 0.5 to (std::numeric_limits<>::max)();
// The Fukushima algorithm changes from a series function for all z > 0.5.

// Multiprecision types:
//#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp> // boost::multiprecision::cpp_dec_float_100
using  boost::multiprecision::cpp_dec_float_100;

#include <boost/math/special_functions/lambert_w.hpp> //
using boost::math::lambert_w0;
using boost::math::lambert_wm1;

#include <iostream>
// using std::cout; using std::std::endl; using std::ios; using std::std::cerr;
#include <iomanip>
using std::setprecision;
using std::showpoint;
#include <fstream>
using std::ofstream;
#include <cassert>
#include <cfloat> // for DBL_EPS etc
#include <limits> // for numeric limits.
//#include <ctype>
#include <string>

const long double eps = std::numeric_limits<long double>::epsilon();

// Creates if no file exists, & uses default overwrite/ ios::replace.
const char filename[] = "lambert_w_high_reference_values.ipp"; //
std::ofstream fout(filename, std::ios::out);

typedef cpp_dec_float_100 RealType;  // 100 decimal digits for the value fed to macro BOOST_MATH_TEST_VALUE.
// Could also use cpp_dec_float_50 or cpp_bin_float types.

const int no_of_tests = 450; // 500 overflows float.

static const float min_z = 0.5F; // for element[0]

int main()
{  // Make C++ file containing Lambert W test values.
  std::cout << filename << " ";
  std::cout << std::endl;
  std::cout << "Lambert W0 decimal digit precision values for high z argument values." << std::endl;

  if (!fout.is_open())
  {  // File failed to open OK.
    std::cerr << "Open file " << filename << " failed!" << std::endl;
    std::cerr << "errno " << errno << std::endl;
    return -1;
  }
  try
  {
    int output_precision = std::numeric_limits<RealType>::digits10;
    // cpp_dec_float_100 is ample precision and
    // has several extra bits internally so max_digits10 are not needed.
    fout.precision(output_precision);
    fout << std::showpoint << std::endl; // Do show trailing zeros.

    // Intro for RealType values.
    std::cout << "Lambert W test values written to file " << filename << std::endl;
    fout <<
      "\n"
      "// A collection of big Lambert W test values computed using "
      << output_precision << " decimal digits precision.\n"
      "// C++ floating-point type is " << "RealType."  "\n"
      "\n"
      "// Written by " << __FILE__ << " " << __TIMESTAMP__ << "\n"

      "\n"
      "// Copyright Paul A. Bristow 2017."   "\n"
      "// Distributed under the Boost Software License, Version 1.0." "\n"
      "// (See accompanying file LICENSE_1_0.txt" "\n"
      "// or copy at http://www.boost.org/LICENSE_1_0.txt)" "\n"
      << std::endl;

    fout << "// Size of arrays of arguments z and Lambert W" << std::endl;
    fout << "static const unsigned int noof_tests = " << no_of_tests << ";" << std::endl;

    // Declare arrays of z and Lambert W.
    fout << "\n// Declare arrays of arguments z and Lambert W(z)" << std::endl;
    fout <<
      "\n"
      "template <typename RealType>""\n"
      "static RealType zs[" << no_of_tests << "];"
      << std::endl;

    fout <<
      "\n"
      "template <typename RealType>""\n"
      "static RealType ws[" << no_of_tests << "];"
      << std::endl;

    fout << "// The values are defined using the macro BOOST_MATH_TEST_VALUE to ensure\n"
      "// that both built-in and multiprecision types are correctly initialized with full precision.\n"
      "// built-in types like float, double require a floating-point literal like 3.14,\n"
      "// but multiprecision types require a decimal digit string like \"3.14\".\n"
      "// Numerical values are chosen to avoid exactly representable values."
      << std::endl;

    static const RealType min_z = 0.6; // for element[0]

    const RealType max_z = (std::numeric_limits<float>::max)() / 10; // (std::numeric_limits<float>::max)() to make sure is OK for all floating-point types.
    // Less a bit as lambert_w0(max) may be inaccurate.
    const RealType step_size = 0.5F; // Increment step size.
    const RealType step_factor = 2.f; // Multiple factor, typically 2, 5 or 10.
    const int step_modulo = 5;

    RealType z = min_z;

    // Output function to initialize array of arguments z and Lambert W.
    fout <<
      "\n"
      << "template <typename RealType>\n"
      "void init_zws()\n"
      "{\n";

    for (size_t index = 0; (index != no_of_tests); index++)
    {
      fout
        << "  zs<RealType>[" << index << "] = BOOST_MATH_TEST_VALUE(RealType, "
        << z // Since start with converting a float may get lots of usefully random digits.
        << ");"
        << std::endl;

      fout
        << "  ws<RealType>[" << index << "] = BOOST_MATH_TEST_VALUE(RealType, "
        << lambert_w0(z)
        << ");"
        << std::endl;

      if ((index % step_modulo) == 0)
      {
        z *= step_factor; //
      }
      z += step_size;
      if (z >= max_z)
      { // Don't go over max for float.
        std::cout << "too big z" << std::endl;
        break;
      }
    } // for index
    fout << "};" << std::endl;

    fout << "// End of lambert_w_mp_high_values.ipp " << std::endl;
  }
  catch (std::exception& ex)
  {
    std::cout << "Exception " << ex.what() << std::endl;
  }

  fout.close();

  std::cout << no_of_tests << " Lambert_w0 values written to files " << __TIMESTAMP__ << std::endl;
  return 0;
}  // main


/*
A few spot checks again Wolfram:

  zs<RealType>[1] = BOOST_MATH_TEST_VALUE(RealType, 1.6999999999999999555910790149937383830547332763671875);
  ws<RealType>[1] = BOOST_MATH_TEST_VALUE(RealType, 0.7796011225311008662356536916883580556792500749037209859530390902424444585607630246126725241921761054);
  Wolfram                                            0.7796011225311008662356536916883580556792500749037209859530390902424444585607630246126725241921761054

  zs<RealType>[99] = BOOST_MATH_TEST_VALUE(RealType, 3250582.599999999976716935634613037109375);
  ws<RealType>[99] = BOOST_MATH_TEST_VALUE(RealType, 12.47094339016839065212822905567651460418204106065566910956134121802725695306834966790193342511971825);
  Wolfram                                            12.47094339016839065212822905567651460418204106065566910956134121802725695306834966790193342511971825

*/


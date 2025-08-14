// lambert_w_test_values.cpp

// Copyright Paul A. Bristow 2017.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Write a C++ file J:\Cpp\Misc\lambert_w_1000_test_values\lambert_w_mp_values.ipp
// containing arrays of z arguments and 100 decimal digit precision lambert_w0(z) reference values.
// These can be used in tests of precision of less-precise types like
// built-in float, double, long double and quad and cpp_dec_float_50.

// Multiprecision types:
//#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp> // boost::multiprecision::cpp_dec_float_100
using  boost::multiprecision::cpp_dec_float_100;

#include <boost/math/special_functions/lambert_w.hpp> //
using boost::math::lambert_w0;

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
#include <algorithm>
using std::transform;

const char* prefix = "static "; // "static const" or "static constexpr" or just "const "" or "" even?
// But problems with VS2017 and GCC not accepting same format mean only static at present.

const long double eps = std::numeric_limits<long double>::epsilon();

/*

// Sample test values from Wolfram.
template <typename RealType>
static RealType zs[noof_tests];

template <typename RealType>
void init_zs()
{
  zs<RealType>[0] = BOOST_MATH_TEST_VALUE(RealType, -0.35);
  zs<RealType>[1] = BOOST_MATH_TEST_VALUE(RealType, -0.3);
  zs<RealType>[2] = BOOST_MATH_TEST_VALUE(RealType, -0.01);
  zs<RealType>[3] = BOOST_MATH_TEST_VALUE(RealType, +0.01);
  zs<RealType>[4] = BOOST_MATH_TEST_VALUE(RealType, 0.1);
  zs<RealType>[5] = BOOST_MATH_TEST_VALUE(RealType, 0.5);
  zs<RealType>[6] = BOOST_MATH_TEST_VALUE(RealType, 1.);
  zs<RealType>[7] = BOOST_MATH_TEST_VALUE(RealType, 2.);
  zs<RealType>[8] = BOOST_MATH_TEST_VALUE(RealType, 5.);
  zs<RealType>[9] = BOOST_MATH_TEST_VALUE(RealType, 10.);
  zs<RealType>[10] = BOOST_MATH_TEST_VALUE(RealType, 100.);
  zs<RealType>[11] = BOOST_MATH_TEST_VALUE(RealType, 1e+6);
};

// 'Known good' Lambert w values using N[productlog(-0.3), 50]
// evaluated to full precision of RealType (up to 50 decimal digits).
template <typename RealType>
static RealType ws[noof_tests];

template <typename RealType>
void init_ws()
{
  ws<RealType>[0] = BOOST_MATH_TEST_VALUE(RealType, -0.7166388164560738505881698000038650406110575701385055261614344530078353170171071547711151137001759321);
  ws<RealType>[1] = BOOST_MATH_TEST_VALUE(RealType, -0.4894022271802149690362312519962933689234100060163590345114659679736814083816206187318524147462752111);
  ws<RealType>[2] = BOOST_MATH_TEST_VALUE(RealType, -0.01010152719853875327292018767138623973670903993475235877290726369225412969738704722202330440072213641);
  ws<RealType>[3] = BOOST_MATH_TEST_VALUE(RealType, 0.009901473843595011885336326816570107953627746494917415482611387085655068978243229360100010886171970918);
  ws<RealType>[4] = BOOST_MATH_TEST_VALUE(RealType, 0.09127652716086226429989572142317956865311922405147203264830839460717224625441755165020664592995606710);
  ws<RealType>[5] = BOOST_MATH_TEST_VALUE(RealType, 0.3517337112491958260249093009299510651714642155171118040466438461099606107203387108968323038321915693);
  ws<RealType>[6] = BOOST_MATH_TEST_VALUE(RealType, 0.5671432904097838729999686622103555497538157871865125081351310792230457930866845666932194469617522946); // Output from https://www.wolframalpha.com/input/?i=lambert_w0(1)
  ws<RealType>[7] = BOOST_MATH_TEST_VALUE(RealType, 0.8526055020137254913464724146953174668984533001514035087721073946525150656742630448965773783502494847);
  ws<RealType>[8] = BOOST_MATH_TEST_VALUE(RealType, 1.326724665242200223635099297758079660128793554638047479789290393025342679920536226774469916608426789); // https://www.wolframalpha.com/input/?i=N%5Bproductlog(5),+100%5D
  ws<RealType>[9] = BOOST_MATH_TEST_VALUE(RealType, 1.745528002740699383074301264875389911535288129080941331322206048555557259941551704989523510778883075);
  ws<RealType>[10] = BOOST_MATH_TEST_VALUE(RealType, 3.385630140290050184888244364529726867491694170157806680386174654885206544913039277686735236213650781);
  ws<RealType>[11] = BOOST_MATH_TEST_VALUE(RealType, 11.38335808614005262200015678158500428903377470601886512143238610626898610768018867797709315493717650);
  ////W(1e35) = 76.256377207295812974093508663841808129811690961764 too big for float.
};

*/

// Global so accessible from output_value.
// Creates if no file exists, & uses default overwrite/ ios::replace.
const char filename[] = "lambert_w_low_reference values.ipp"; //
std::ofstream fout(filename, std::ios::out); ; //

// 100 decimal digits for the value fed to macro BOOST_MATH_TEST_VALUE
typedef cpp_dec_float_100 RealType;

void output_value(size_t index, RealType value)
{
  fout
    << "  zs<RealType>[" << index << "] = BOOST_MATH_TEST_VALUE(RealType, "
    << value
    << ");"
    << std::endl;
  return;
}

void output_lambert_w0(size_t index, RealType value)
{
  fout
    << "  ws<RealType>[" << index << "] = BOOST_MATH_TEST_VALUE(RealType, "
    << lambert_w0(value)
    << ");"
    << std::endl;
  return;
}

int main()
{  // Make C++ file containing Lambert W test values.
  std::cout << filename << " ";
#ifdef __TIMESTAMP__
  std::cout << __TIMESTAMP__;
#endif
  std::cout << std::endl;
  std::cout << "Lambert W0 decimal digit precision values." << std::endl;

  // Note __FILE__ & __TIMESTAMP__ are ANSI standard C & thus Std C++?

  if (!fout.is_open())
  {  // File failed to open OK.
    std::cerr << "Open file " << filename << " failed!" << std::endl;
    std::cerr << "errno " << errno << std::endl;
    return -1;
  }

  int output_precision = std::numeric_limits<cpp_dec_float_100>::digits10;
  // cpp_dec_float_100 is ample precision and
  // has several extra bits internally so max_digits10 are not needed.
  fout.precision(output_precision);

  int no_of_tests = 100;

  // Intro for RealType values.
  std::cout << "Lambert W test values written to file " << filename << std::endl;
  fout <<
    "\n"
    "// A collection of Lambert W test values computed using "
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

  fout << "// Declare arrays of arguments z and Lambert W(z)" << std::endl;
  fout << "// The values are defined using the macro BOOST_MATH_TEST_VALUE to ensure\n"
    "// that both built-in and multiprecision types are correctly initialized with full precision.\n"
    "// built-in types like double require a floating-point literal like 3.14,\n"
    "// but multiprecision types require a decimal digit string like \"3.14\".\n"
    << std::endl;
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

  RealType max_z("10");
  RealType min_z("-0.35");
  RealType step_size("0.01");
  size_t step_count = no_of_tests;

  // Output to initialize array of arguments z for Lambert W.
  fout <<
    "\n"
    << "template <typename RealType>\n"
    "void init_zs()\n"
    "{\n";

  RealType z = min_z;
  for (size_t i = 0; (i < no_of_tests); i++)
  {
    output_value(i, z);
    z += step_size;
  }
  fout << "};" << std::endl;

  // Output array of Lambert W values.
  fout <<
    "\n"
    << "template <typename RealType>\n"
    "void init_ws()\n"
    "{\n";

  z = min_z;
  for (size_t i = 0; (i < step_count); i++)
  {
    output_lambert_w0(i, z);
    z += step_size;
  }
  fout << "};" << std::endl;

  fout << "// End of lambert_w_mp_values.ipp " << std::endl;
  fout.close();

  std::cout << "Lambert_w0 values written to files " << __TIMESTAMP__ << std::endl;
  return 0;
}  // main


/*

start and finish checks again WolframAlpha:
ws<RealType>[0] = BOOST_MATH_TEST_VALUE(RealType, -0.7166388164560738505881698000038650406110575701385055261614344530078353170171071547711151137001759321);
Wolfram N[productlog(-0.35), 100]                 -0.7166388164560738505881698000038650406110575701385055261614344530078353170171071547711151137001759321


ws<RealType>[19] = BOOST_MATH_TEST_VALUE(RealType, 0.7397278549447991214587608743391115983469848985053641692586810406118264600667862543570373167046626221);
Wolfram N[productlog(1.55), 100]                   0.7397278549447991214587608743391115983469848985053641692586810406118264600667862543570373167046626221


*/


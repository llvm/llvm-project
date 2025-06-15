/** lexical_cast_nonfinite_facets.cpp
*
* Copyright (c) 2011 Paul A. Bristow
*
* Distributed under the Boost Software License, Version 1.0.
* (See accompanying file LICENSE_1_0.txt
* or copy at http://www.boost.org/LICENSE_1_0.txt)
*
* This very simple program illustrates how to use the
* `boost/math/nonfinite_num_facets.hpp' with lexical cast
* to obtain C99 representation of infinity and NaN.
* This example is from the original Floating Point  Utilities contribution by Johan Rade.
* Floating Point Utility library has been accepted into Boost,
* but the utilities are incorporated into Boost.Math library.
*
\file

\brief A very simple example of using lexical cast with
non_finite_num facet for C99 standard output of infinity and NaN.

\detail This example shows how to create a C99 non-finite locale,
and imbue input and output streams with the non_finite_num put and get facets.
This allows lexical_cast output and input of infinity and NaN in a Standard portable way,
This permits 'loop-back' of output back into input (and portably across different system too).

*/

#include <boost/math/special_functions/nonfinite_num_facets.hpp>
using boost::math::nonfinite_num_get;
using boost::math::nonfinite_num_put;

#include <boost/lexical_cast.hpp>
using boost::lexical_cast;

#include <iostream>
using std::cout;
using std::endl;
using std::cerr;

#include <iomanip>
using std::setw;
using std::left;
using std::right;
using std::internal;

#include <string>
using std::string;

#include <sstream>
using std::istringstream;

#include <limits>
using std::numeric_limits;

#include <locale>
using std::locale;

#include <boost/math/tools/assert.hpp>

int main ()
{
  std::cout << "lexical_cast example (NOT using finite_num_facet)." << std::endl;
    
  if((std::numeric_limits<double>::has_infinity == false) || (std::numeric_limits<double>::infinity() == 0))
  {
    std::cout << "Infinity not supported on this platform." << std::endl;
    return 0;
  }

  if((std::numeric_limits<double>::has_quiet_NaN == false) || (std::numeric_limits<double>::quiet_NaN() == 0))
  {
    std::cout << "NaN not supported on this platform." << std::endl;
    return 0;
  }
  
  // Some tests that are expected to fail on some platforms.
  // (But these tests are expected to pass using non_finite num_put and num_get facets).

  // Use the current 'native' default locale.
  std::locale default_locale (std::locale::classic ()); // Note the current (default C) locale.

  // Create plus and minus infinity.
  double plus_infinity = +std::numeric_limits<double>::infinity();
  double minus_infinity = -std::numeric_limits<double>::infinity();

  // and create a NaN (NotANumber).
  double NaN = +std::numeric_limits<double>::quiet_NaN ();

  // Output the nonfinite values using the current (default C) locale.
  // The default representations differ from system to system,
  // for example, using Microsoft compilers, 1.#INF, -1.#INF, and 1.#QNAN.
  cout << "Using default locale" << endl;
  cout << "+std::numeric_limits<double>::infinity() = " << plus_infinity << endl;
  cout << "-std::numeric_limits<double>::infinity() = " << minus_infinity << endl;
  cout << "+std::numeric_limits<double>::quiet_NaN () = " << NaN << endl;
      
  // Checks below are expected to fail on some platforms!

  // Now try some 'round-tripping', 'reading' "inf"
  double x = boost::lexical_cast<double>("inf");
  // and check we get a floating-point infinity.
  BOOST_MATH_ASSERT(x == std::numeric_limits<double>::infinity());

  // Check we can convert the other way from floating-point infinity,
  string s = boost::lexical_cast<string>(numeric_limits<double>::infinity());
  // to a C99 string representation as "inf".
  BOOST_MATH_ASSERT(s == "inf");

  // Finally try full 'round-tripping' (in both directions):
  BOOST_MATH_ASSERT(lexical_cast<double>(lexical_cast<string>(numeric_limits<double>::infinity()))
    == numeric_limits<double>::infinity());
  BOOST_MATH_ASSERT(lexical_cast<string>(lexical_cast<double>("inf")) == "inf");

  return 0;
} // int main()

/*

Output:

from MSVC 10, fails (as expected)

  lexical_cast_native.vcxproj -> J:\Cpp\fp_facet\fp_facet\Debug\lexical_cast_native.exe
  lexical_cast example (NOT using finite_num_facet).
  Using default locale
  +std::numeric_limits<double>::infinity() = 1.#INF
  -std::numeric_limits<double>::infinity() = -1.#INF
  +std::numeric_limits<double>::quiet_NaN () = 1.#QNAN
C:\Program Files\MSBuild\Microsoft.Cpp\v4.0\Microsoft.CppCommon.targets(183,5): error MSB3073: The command ""J:\Cpp\fp_facet\fp_facet\Debug\lexical_cast_native.exe"
C:\Program Files\MSBuild\Microsoft.Cpp\v4.0\Microsoft.CppCommon.targets(183,5): error MSB3073: :VCEnd" exited with code 3.


*/

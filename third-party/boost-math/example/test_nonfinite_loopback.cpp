// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Copyright (c) 2006 Johan Rade
// Copyright (c) 2011 Paul A. Bristow

/*!
\file
\brief Basic tests of native nonfinite loopback.

\detail Basic loopback test outputs using the platforms built-in facets
and reads back in, and checks if loopback OK.

Using MSVC this doesn't work OK:
input produces just "1" instead of "1.#QNAN", 1.#SNAN" or 1.#IND"!

*/

#include <iostream>
using std::cout;
using std::endl;
#include <locale>
using std::locale;
#include <string>
using std::string;
#include <sstream>
  using std::stringstream;
#include <limits>
using std::numeric_limits;

int main()
{
   locale default_locale; // Current global locale.
  // Try to use the default locale first.
  // On MSVC this doesn't work.

  { // Try infinity.
    stringstream ss; // Both input and output.
    ss.imbue(default_locale); // Redundant, of course.
    string infs;
    if(numeric_limits<double>::has_infinity)
    {  // Make sure infinity is specialised for type double.
      double inf = numeric_limits<double>::infinity();
      ss << inf; // Output infinity.
      infs = ss.str();  //
    }
    else
    { // Need to provide a suitable string for infinity.
     infs =  "1.#INF"; // Might suit MSVC?
      ss << infs;
    }
    double r;
    ss >> r; // Read back in.

    cout << "infinity output was " << infs << endl; // "1.#INF"
    cout << "infinity input was " << r << endl; // "1"
  }

    { // Try Quiet NaN
    stringstream ss; // Both input and output.
    ss.imbue(default_locale); // Redundant, of course.
    string infs;
    if(numeric_limits<double>::has_quiet_NaN)
    {  // Make sure quiet NaN is specialised for type double.
      double qnan = numeric_limits<double>::quiet_NaN();
      ss << qnan; // Output quiet_NaN.
      infs = ss.str();  //
    }
    else
    { // Need to provide a suitable string for quiet_NAN.
     infs =  "1.#QNAN";
      ss << infs;
    }
    double r;
    ss >> r; // Read back in.

    cout << "quiet_NaN output was " << infs << endl; // "1.#QNAN"
    cout << "quiet_NaN input was " << r << endl; // "1#"
  }


} // int main()

/*

Output (MSVC Version 10.0):


  infinity output was 1.#INF
  infinity input was 1
  quiet_NaN output was 1.#QNAN
  quiet_NaN input was 1


*/


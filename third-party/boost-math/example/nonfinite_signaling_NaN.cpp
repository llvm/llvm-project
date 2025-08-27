// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Copyright (c) 2006 Johan Rade
// Copyright (c) 2011 Paul A. Bristow

/*!
\file
\brief Tests of nonfinite signaling NaN loopback.

\detail  nonfinite signaling NaN
test outputs using nonfinite facets
(output and input) and reads back in, and checks if loopback OK.

Not expected to work on all platforms (if any).  But shows that on MSVC,
this legacy locale can ensure a consistent quiet NaN input from representations
"1.#QNAN", "1.#SNAN" and "1.#IND"

*/

#ifdef _MSC_VER
#   pragma warning(disable : 4702)
#endif

#include <boost/math/special_functions/nonfinite_num_facets.hpp>
using boost::math::nonfinite_num_get;
using boost::math::nonfinite_num_put;

#include <iostream>
using std::cout;
using std::endl;

#include <locale>
using std::locale;

#include <string>
using std::string;

#include <sstream>
  using std::stringstream;
  using std::istringstream;

#include <limits>
using std::numeric_limits;

int main()
{
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

  locale default_locale; // Current global locale.
  // Try to use the default locale first.
  // On MSVC this doesn't work.

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
    { // Need to provide a suitable string for quiet NaN.
     infs =  "1.#QNAN";
      ss << infs;
    }
    double r;
    ss >> r; // Read back in.

    cout << "quiet_NaN output was " << infs << endl; // "1.#QNAN"
    cout << "quiet_NaN input was " << r << endl; // "1"
  }

#if (!defined BOOST_BORLANDC && !defined BOOST_CODEGEARC)
  // These compilers trap when trying to create a signaling_NaN!
  { // Try Signaling NaN
    stringstream ss; // Both input and output.
    ss.imbue(default_locale); // Redundant, of course.
    string infs;
    if(numeric_limits<double>::has_signaling_NaN)
    {  // Make sure signaling NaN is specialised for type double.
      double qnan = numeric_limits<double>::signaling_NaN();
      ss << qnan; // Output signaling_NaN.
      infs = ss.str();  //
    }
    else
    { // Need to provide a suitable string for signaling NaN.
     infs =  "1.#SNAN";
      ss << infs;
    }
    double r;
    ss >> r; // Read back in.

    cout << "signaling_NaN output was " << infs << endl; // "1.#QNAN" (or "1.#SNAN"?)
    cout << "signaling_NaN input was " << r << endl; // "1"
  }
#endif // Not Borland or CodeGear.

  // Create legacy_locale and store the nonfinite_num_get facet (with legacy flag) in it.
  locale legacy_locale(default_locale, new nonfinite_num_get<char>(boost::math::legacy));
  // Note that the legacy flag has no effect on the nonfinite_num_put output facet.

  cout << "Use legacy locale." << endl;

  { // Try infinity.
    stringstream ss; // Both input and output.
    ss.imbue(legacy_locale);
    string infs;
    if(numeric_limits<double>::has_infinity)
    {  // Make sure infinity is specialised for type double.
      double inf = numeric_limits<double>::infinity();
      ss << inf; // Output infinity.
      infs = ss.str();  //
    }
    else
    { // Need to provide a suitable string for infinity.
     infs =  "1.#INF";
      ss << infs;
    }
    double r;
    ss >> r; // Read back in.

    cout << "infinity output was " << infs << endl; // "1.#INF"
    cout << "infinity input was " << r << endl; // "1.#INF"
  }

  { // Try input of "1.#SNAN".
    //double inf = numeric_limits<double>::signaling_NaN(); // Assigns "1.#QNAN" on MSVC.
    // So must use explicit string "1.#SNAN" instead.
    stringstream ss; // Both input and output.
    ss.imbue(legacy_locale);
    string s = "1.#SNAN";

    ss << s; // Write out.
    double r;

    ss >> r; // Read back in.

    cout << "SNAN output was " << s << endl; // "1.#SNAN"
    cout << "SNAN input was " << r << endl;  // "1.#QNAN"
  }

  { // Try input of "1.#IND" .
    stringstream ss; // Both input and output.
    ss.imbue(legacy_locale);
    string s = "1.#IND";
    ss << s; // Write out.
    double r;
    ss >> r; // Read back in.

    cout << "IND output was " << s << endl; // "1.#IND"
    cout << "IND input was " << r << endl;  // "1.#QNAN"
  }

} // int main()

/*

Output:
  nonfinite_signaling_NaN.vcxproj -> J:\Cpp\fp_facet\fp_facet\Debug\nonfinite_signaling_NaN.exe

  quiet_NaN output was 1.#QNAN
  quiet_NaN input was 1
  signaling_NaN output was 1.#QNAN
  signaling_NaN input was 1
  Use legacy locale.
  infinity output was 1.#INF
  infinity input was 1.#INF
  SNAN output was 1.#SNAN
  SNAN input was 1.#QNAN
  IND output was 1.#IND
  IND input was 1.#QNAN


*/


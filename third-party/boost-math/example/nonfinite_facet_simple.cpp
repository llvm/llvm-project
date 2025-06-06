/** nonfinite_num_facet.cpp
*
* Copyright (c) 2011 Paul A. Bristow
*
* Distributed under the Boost Software License, Version 1.0.
* (See accompanying file LICENSE_1_0.txt
* or copy at http://www.boost.org/LICENSE_1_0.txt)
*
* This very simple program illustrates how to use the
* `boost/math/nonfinite_num_facets.hpp' to obtain C99
* representation of infinity and NaN.
* (from the original
* Floating Point  Utilities contribution by Johan Rade.
* Floating Point Utility library has been accepted into Boost,
* but the utilities are incorporated into Boost.Math library.
*
\file

\brief A very simple example of using non_finite_num facet for
C99 standard output of infinity and NaN.

\detail Provided infinity and nan are supported,
this example shows how to create a C99 non-finite locale,
and imbue input and output streams with the non_finite_num put and get facets.
This allow output and input of infinity and NaN in a Standard portable way,
This permits 'loop-back' of output back into input (and portably across different system too).
This is particularly useful when used with Boost.Serialization so that non-finite NaNs and infinity
values in text and xml archives can be handled correctly and portably.

*/

#ifdef _MSC_VER
#  pragma warning (disable : 4127)  // conditional expression is constant.
#endif

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

#include <boost/math/special_functions/nonfinite_num_facets.hpp>
// from Johan Rade Floating Point Utilities.

int main ()
{
  std::cout << "Nonfinite_num_facet very simple example." << std::endl;

  if((std::numeric_limits<double>::has_infinity == 0) || (std::numeric_limits<double>::infinity() == 0))
  {
    std::cout << "Infinity not supported on this platform." << std::endl;
    return 0;
  }

  if((std::numeric_limits<double>::has_quiet_NaN == 0) || (std::numeric_limits<double>::quiet_NaN() == 0))
  {
    std::cout << "NaN not supported on this platform." << std::endl;
    return 0;
  }

  std::locale default_locale (std::locale::classic ()); // Note the current (default C) locale.

  // Create plus and minus infinity.
  double plus_infinity = +std::numeric_limits<double>::infinity();
  double minus_infinity = -std::numeric_limits<double>::infinity();

  // and create a NaN (NotANumber)
  double NaN = +std::numeric_limits<double>::quiet_NaN ();

  double negated_NaN = (boost::math::changesign)(std::numeric_limits<double>::quiet_NaN ());


  // Output the nonfinite values using the current (default C) locale.
  // The default representations differ from system to system,
  // for example, using Microsoft compilers, 1.#INF, -1.#INF, and 1.#QNAN,
  // Linux "inf", "-inf", "nan"
  cout << "Using C locale" << endl;
  cout << "+std::numeric_limits<double>::infinity() = " << plus_infinity << endl;
  cout << "-std::numeric_limits<double>::infinity() = " << minus_infinity << endl;
  cout << "+std::numeric_limits<double>::quiet_NaN () = " << NaN << endl;

  // Display negated NaN.
  cout << "negated NaN " << negated_NaN << endl; // "-1.IND" or "-nan".
  
  // Create a new output locale, and add the nonfinite_num_put facet
  std::locale C99_out_locale (default_locale, new boost::math::nonfinite_num_put<char>);
  // and imbue the cout stream with the new locale.
  cout.imbue (C99_out_locale);

  // Or for the same effect more concisely:
  cout.imbue (locale(locale(), new boost::math::nonfinite_num_put<char>));

  // Output using the new locale:
  cout << "Using C99_out_locale " << endl;
  cout << "+std::numeric_limits<double>::infinity() = " << plus_infinity << endl;
  cout << "-std::numeric_limits<double>::infinity() = " << minus_infinity << endl;
  cout << "+std::numeric_limits<double>::quiet_NaN () = " << NaN << endl;
  // Expect "inf", "-inf", "nan".

  // Display negated NaN.
  cout << "negated NaN " << negated_NaN << endl; // Expect "-nan".

  // Create a string with the expected C99 representation of plus infinity.
  std::string inf = "inf";
  { // Try to read an infinity value using the default C locale.
    // Create an input stream which will provide "inf"
    std::istringstream iss (inf);

     // Create a double ready to take the input,
    double infinity;
    // and read "inf" from the stringstream:
    iss >> infinity; 

    // This will not work on all platforms!  (Intel-Linux-13.0.1 fails EXIT STATUS: 139)
    if (! iss)
    { // Reading infinity went wrong!
      std::cerr << "C locale input format error!" << std::endl;
    }
  } // Using default C locale.

  { // Now retry using C99 facets.
  // Create a new input locale and add the nonfinite_num_get facet.
  std::locale C99_in_locale (default_locale, new boost::math::nonfinite_num_get<char>);

  // Create an input stream which will provide "inf".
  std::istringstream iss (inf);
  // Imbue the stream with the C99 input locale.
  iss.imbue (C99_in_locale);

  // Create a double ready to take the input,
  double infinity;
  // and read from the stringstream:
  iss >> infinity; 

  if (! iss)
  { // Reading infinity went wrong!
    std::cout << "C99 input format error!" << std::endl;
  }
  // Expect to get an infinity, which will display still using the C99 locale as "inf"
  cout << "infinity in C99 representation is " << infinity << endl; 

  // To check, we can switch back to the default C locale.
  cout.imbue (default_locale);
  cout <<  "infinity in default C representation is " << infinity << endl; 
  } // using C99 locale.

  {
    // A 'loop-back example, output to a stringstream, and reading it back in.
    // Create C99 input and output locales. 
    std::locale C99_out_locale (default_locale, new boost::math::nonfinite_num_put<char>);
    std::locale C99_in_locale (default_locale, new boost::math::nonfinite_num_get<char>);

    std::ostringstream oss;
    oss.imbue(C99_out_locale);
    oss << plus_infinity;

    std::istringstream iss(oss.str()); // So stream contains "inf".
    iss.imbue (C99_in_locale);

    std::string s;

    iss >> s;

    cout.imbue(C99_out_locale);
    if (oss.str() != s)
    {
      cout << plus_infinity << " != " << s << " loopback failed!" << endl;
    }
    else
    {
      cout << plus_infinity << " == " << s << " as expected." << endl;
    }
  }


  // Example varying the width and position of the nonfinite representations.
  // With the nonfinite_num_put and _get facets, the width of the output is constant.

  #ifdef BOOST_NO_CXX11_NUMERIC_LIMITS
  cout << "BOOST_NO_CXX11_NUMERIC_LIMITS is defined, so no max_digits10 available." << endl;
  std::streamsize  max_digits10 = 2 + std::numeric_limits<double>::digits * 30103UL / 100000UL;
#else
  // Can use new C++0X max_digits10 (the maximum potentially significant digits).
  std::streamsize  max_digits10 = std::numeric_limits<double>::max_digits10;
#endif
  cout << "std::numeric_limits<double>::max_digits10 is " << max_digits10 << endl;
  cout.precision(max_digits10);

  double pi = 3.141592653589793238462643383279502884197169399375105820974944;
  // Expect 17 (probably) decimal digits (regardless of locale).
  // cout has the default locale.
  cout << "pi = " << pi << endl; // pi = 3.1415926535897931
  cout.imbue (C99_out_locale); // Use cout with the C99 locale
  // (expect the same output for a double).
  cout << "pi = " << pi << endl; // pi = 3.1415926535897931

  cout << "infinity in C99 representation is " << plus_infinity << endl; 

  //int width = 2; // Check effect if width too small is OK.
  // (There was a disturbed layout on older MSVC?).
  int width = 20;

  // Similarly if we can switch back to the default C locale.
  cout.imbue (default_locale);
  cout <<  "infinity in default C representation is " << plus_infinity << endl; 
  cout <<  "infinity in default C representation (setw(" << width << ") is |" << setw(width) << plus_infinity <<'|' << endl; 
  cout <<  "infinity in default C representation (setw(" << width << ") is |" << left << setw(width) << plus_infinity <<'|' << endl; 
  cout <<  "infinity in default C representation (setw(" << width << ") is |" << internal << setw(width) << plus_infinity <<'|' << endl; 

  cout.imbue (C99_out_locale);
  cout << "infinity in C99 representation (setw(" << width << ") is |" << right << setw(width) << plus_infinity <<'|'<< endl; 
  cout << "infinity in C99 representation (setw(" << width << ") is |" << left << setw(width) << plus_infinity <<'|'<< endl; 
  cout << "infinity in C99 representation (setw(" << width << ") is |" << internal << setw(width) << plus_infinity <<'|'<< endl; 

  return 0;
} // int main()

// end of test_nonfinite_num_facets.cpp

/*

Output:

simple_nonfinite_facet.vcxproj -> J:\Cpp\MathToolkit\test\Math_test\Release\nonfinite_facet_simple.exe
  Nonfinite_num_facet very simple example.
  Using C locale
  +std::numeric_limits<double>::infinity() = 1.#INF
  -std::numeric_limits<double>::infinity() = -1.#INF
  +std::numeric_limits<double>::quiet_NaN () = 1.#QNAN
  Using C99_out_locale 
  +std::numeric_limits<double>::infinity() = inf
  -std::numeric_limits<double>::infinity() = -inf
  +std::numeric_limits<double>::quiet_NaN () = nan
  infinity in C99 representation is inf
  infinity in default C representation is 1.#INF
  3
  3
  inf == inf as expected.
  std::numeric_limits<double>::max_digits10 is 17
  pi = 3.1415926535897931
  C locale input format error!
  pi = 3.1415926535897931
  infinity in C99 representation is inf
  infinity in default C representation is 1.#INF
  infinity in default C representation (setw(20) is               1.#INF|
  infinity in default C representation (setw(20) is 1.#INF              |
  infinity in default C representation (setw(20) is               1.#INF|
  infinity in C99 representation (setw(20) is                  inf|
  infinity in C99 representation (setw(20) is inf                 |
  infinity in C99 representation (setw(20) is                  inf|

*/

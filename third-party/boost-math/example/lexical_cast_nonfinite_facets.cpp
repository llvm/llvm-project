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

See also lexical_cast_native.cpp which is expected to fail on many systems,
but might succeed if the default locale num_put and num_get facets
comply with C99 nonfinite input and output specification.

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
  std::cout << "finite_num_facet with lexical_cast example." << std::endl;

  // Example of using non_finite num_put and num_get facets with lexical_cast.
  //locale old_locale;
  //locale tmp_locale(old_locale, new nonfinite_num_put<char>);
  //// Create a new temporary output locale, and add the output nonfinite_num_put facet.

  //locale new_locale(tmp_locale, new nonfinite_num_get<char>);
  // Create a new output locale (from the tmp locale), and add the input nonfinite_num_get facet.

  // Note that you can only add facets one at a time, 
  // unless you chain thus:
  
  std::locale new_locale(std::locale(std::locale(),
    new boost::math::nonfinite_num_put<char>),
    new boost::math::nonfinite_num_get<char>);
  
  locale::global(new_locale); // Newly constructed streams
  // (including those streams inside lexical_cast)
  // now use new_locale with nonfinite facets.

  // Output using the new locale.
  cout << "Using C99_out_locale " << endl;
  cout.imbue(new_locale);
  // Necessary because cout already constructed using default C locale,
  // and default facets for nonfinites.

    // Create plus and minus infinity.
  double plus_infinity = +std::numeric_limits<double>::infinity();
  double minus_infinity = -std::numeric_limits<double>::infinity();

  // and create a NaN (NotANumber)
  double NaN = +std::numeric_limits<double>::quiet_NaN ();
  cout << "+std::numeric_limits<double>::infinity() = " << plus_infinity << endl;
  cout << "-std::numeric_limits<double>::infinity() = " << minus_infinity << endl;
  cout << "+std::numeric_limits<double>::quiet_NaN () = " << NaN << endl;

  // Now try some 'round-tripping', 'reading' "inf".
  double x = boost::lexical_cast<double>("inf");
  // and check we get a floating-point infinity.
  BOOST_MATH_ASSERT(x == std::numeric_limits<double>::infinity());
  cout << "boost::lexical_cast<double>(\"inf\") = " << x << endl;

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
  finite_num_facet with lexical_cast example.
  Using C99_out_locale 
  +std::numeric_limits<double>::infinity() = inf
  -std::numeric_limits<double>::infinity() = -inf
  +std::numeric_limits<double>::quiet_NaN () = nan
  boost::lexical_cast<double>("inf") = inf


*/

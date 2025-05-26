/** nonfinite_serialization_archives.cpp
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
* From the original Floating Point  Utilities contribution by Johan Rade.
* Floating Point Utility library has been accepted into Boost,
* but the utilities are incorporated into Boost.Math library.
*
\file

\brief A simple example of using non_finite_num facet for
C99 standard output of infinity and NaN in serialization archives.

\detail This example shows how to create a C99 non-finite locale,
and imbue input and output streams with the non_finite_num put and get facets.
This allow output and input of infinity and NaN in a Standard portable way,
This permits 'loop-back' of output back into input (and portably across different system too).
This is particularly useful when used with Boost.Serialization so that non-finite NaNs and infinity
values in text and xml archives can be handled correctly and portably.

*/


#ifdef _MSC_VER
#   pragma warning(disable : 4127) // conditional expression is constant.
#endif


#include <boost/archive/text_oarchive.hpp>
using boost::archive::text_oarchive;
#include <boost/archive/codecvt_null.hpp>
using boost::archive::codecvt_null;
using boost::archive::no_codecvt;

#include <boost/math/special_functions/nonfinite_num_facets.hpp>
using boost::math::nonfinite_num_get;
using boost::math::nonfinite_num_put;

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

#include <fstream>
using std::ofstream;

#include <limits>
using std::numeric_limits;

#include <locale>
using std::locale;


/*
Use with serialization archives.

It is important that the same locale is used
when an archive is saved and when it is loaded.
Otherwise, loading the archive may fail.

By default, archives are saved and loaded with a classic C locale with a
`boost::archive::codecvt_null` facet added.
Normally you do not have to worry about that.
The constructors for the archive classes, as a side-effect,
imbue the stream with such a locale.

However, if you want to use the facets `nonfinite_num_put` and `nonfinite_num_get`
with archives,`then you have to manage the locale manually.

That is done by calling the archive constructor with the flag `boost::archive::no_codecvt`.
Then the archive constructor will not imbue the stream with a new locale.

The following code shows how to use `nonfinite_num_put` with a `text_oarchive`:

*/

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

  locale default_locale(locale::classic(), new boost::archive::codecvt_null<char>);
  // codecvt_null so the archive constructor will not imbue the stream with a new locale.

  locale my_locale(default_locale, new nonfinite_num_put<char>);
  // Add nonfinite_num_put facet to locale.

  // Use a temporary folder /.temps (which contains "boost-no-inspect" so that it will not be inspected, and made 'hidden' too).
  ofstream ofs("./.temps/test.txt");
  ofs.imbue(my_locale);

  boost::archive::text_oarchive oa(ofs, no_codecvt);

  double x = numeric_limits<double>::infinity();
  oa & x;

} // int main()


/* The same method works with nonfinite_num_get  and text_iarchive.

If you use the trap_infinity and trap_nan flags with a serialization archive,
then you must set the exception mask of the stream.
Serialization archives do not check the stream state.


*/

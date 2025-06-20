// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Copyright (c) 2006 Johan Rade
// Copyright (c) 2011 Paul A. Bristow

/*!
\file
\brief Basic tests of nonfinite loopback.

\detail Basic loopback test outputs using nonfinite facets
(output and input) and reads back in, and checks if loopback OK.

Expected to work portably on all platforms.

*/

#ifdef _MSC_VER
#   pragma warning(disable : 4702)
#   pragma warning(disable : 4127) // conditional expression is constant.
#endif

#include <boost/math/special_functions/nonfinite_num_facets.hpp>
using boost::math::nonfinite_num_get;
using boost::math::nonfinite_num_put;

#include <iostream>
using std::cout;
using std::endl;

#include <locale>
using std::locale;

#include <sstream>
using std::stringstream;
#include <limits>
using std::numeric_limits;

#include <assert.h>

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
  //locale old_locale; // Current global locale.
  //  Create tmp_locale and store the output nonfinite_num_put facet in it.
  //locale tmp_locale(old_locale, new nonfinite_num_put<char>);
  //  Create new_locale and store the input nonfinite_num_get facet in it.
  //locale new_locale(tmp_locale, new nonfinite_num_get<char>);
  //  Can only add one facet at a time, hence need a tmp_locale,
  //  unless we write:

  std::locale new_locale(std::locale(std::locale(std::locale(),
    new boost::math::nonfinite_num_put<char>),
    new boost::math::nonfinite_num_get<char>));

  stringstream ss; // Both input and output, so need both get and put facets.

  ss.imbue(new_locale);

  double inf = numeric_limits<double>::infinity();
  ss << inf; // Write out.
  double r;
  ss >> r; // Read back in.

  BOOST_MATH_ASSERT(inf == r); // OK MSVC <= 10.0!

} // int main()

/*

Output:

nonfinite_loopback_ok.vcxproj -> J:\Cpp\fp_facet\fp_facet\Debug\nonfinite_loopback_ok.exe

*/



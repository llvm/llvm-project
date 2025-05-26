// nonfinite_legacy.cpp

// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Copyright (c) 2006 Johan Rade
// Copyright (c) 2011 Paul A. Bristow

/*!
\file
\brief Basic tests of nonfinite loopback with output and input facet.

\detail Basic loopback test outputs using the so-called 'legacy' facets,
"1.#INF"  and "1.#QNAN".

and reads back in using nonfinite input 'legacy' facet, and
(if possible) checks if loopback OK.

*/

#include <boost/math/special_functions/nonfinite_num_facets.hpp>
using boost::math::nonfinite_num_put;
using boost::math::nonfinite_num_get;

using boost::math::legacy;

#include <iostream>
using std::cout;
using std::endl;

#include <iomanip>
using std::setfill;
using std::setw;

#include <locale>
using std::locale;

#include <sstream>
using std::stringstream;
#include <limits>
using std::numeric_limits;

#include <assert.h>

int main()
{
  // Create a new locale with both the nonfinite facets.
  std::locale new_locale(std::locale(std::locale(),
    new boost::math::nonfinite_num_put<char>),
    new boost::math::nonfinite_num_get<char>);

  {
    stringstream ss;
    ss.imbue(new_locale);
    double inf = numeric_limits<double>::infinity();
    ss << inf; // Write out.
    double r;
    ss >> r; // Read back in.

    cout << "infinity output was " << inf << endl;
    cout << "infinity input was " << r << endl;

    BOOST_MATH_ASSERT(inf == r);
  }
  {
    stringstream ss;
    ss.imbue(new_locale);

    double nan = numeric_limits<double>::quiet_NaN();
    ss << nan; // Write out.
    double v;
    ss >> v; // Read back in.

    cout << "NaN output was " << nan << endl;
    cout << "NaN input was " << v << endl;

    // BOOST_MATH_ASSERT(nan == v); // Always fails because NaN == NaN fails!
    // BOOST_MATH_ASSERT(nan == numeric_limits<double>::quiet_NaN()); asserts!
  }

} // int main()

/*

Output:

infinity output was 1.#INF
infinity input was 1.#INF
NaN output was 1.#QNAN
NaN input was 1.#QNAN

*/


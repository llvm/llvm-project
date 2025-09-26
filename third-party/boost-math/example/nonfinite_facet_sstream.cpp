// nonfinite_facet_sstream.cpp

// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Copyright (c) 2006 Johan Rade
// Copyright (c) 2011 Paul A. Bristow

/*!
\file
\brief Examples of nonfinite with output and input facets and stringstreams.

\detail Construct a new locale with the nonfinite_num_put and nonfinite_num_get
facets and imbue istringstream, ostringstream and stringstreams,
showing output and input (and loopback for the stringstream).

*/

#include <boost/math/special_functions/nonfinite_num_facets.hpp>
using boost::math::nonfinite_num_put;
using boost::math::nonfinite_num_get;

using boost::math::legacy;

#include <iostream>
using std::cout;
using std::endl;
#include <locale>
using std::locale;

#include <sstream>
using std::stringstream;
using std::istringstream;
using std::ostringstream;

#include <limits>
using std::numeric_limits;

#include <assert.h>

int main()
{
  //[nonfinite_facets_sstream_1
  locale old_locale;
  locale tmp_locale(old_locale, new nonfinite_num_put<char>);
  locale new_locale(tmp_locale, new nonfinite_num_get<char>);
  //] [/nonfinite_facets_sstream_1]

  // Note that to add two facets,  nonfinite_num_put and nonfinite_num_get,
  // you have to add one at a time, using a temporary locale.

  {
    ostringstream oss;
    oss.imbue(new_locale);
    double inf = numeric_limits<double>::infinity();
    oss << inf; // Write out.
    cout << "infinity output was " << oss.str() << endl;
    BOOST_MATH_ASSERT(oss.str() == "inf");
  }
  {
    istringstream iss;
    iss.str("inf");
    iss.imbue(new_locale);
    double inf;
    iss >> inf; // Read from "inf"
    cout << "Infinity input was " << iss.str() << endl;
    BOOST_MATH_ASSERT(inf == numeric_limits<double>::infinity());
  }

  {
    //[nonfinite_facets_sstream_2
    stringstream ss;
    ss.imbue(new_locale);
    double inf = numeric_limits<double>::infinity();
    ss << inf; // Write out.
    BOOST_MATH_ASSERT(ss.str() == "inf");
    double r;
    ss >> r; // Read back in.
    BOOST_MATH_ASSERT(inf == r); // Confirms that the double values really are identical.

    cout << "infinity output was " << ss.str() << endl;
    cout << "infinity input was " << r << endl;
    // But the string representation of r displayed will be the native type
    // because, when it was constructed, cout had NOT been imbued
    // with the new locale containing the nonfinite_numput facet.
    // So the cout output will be "1.#INF on MS platforms
    // and may be "inf" or other string representation on other platforms.

    //] [/nonfinite_facets_sstream_2]
  }

  {
    stringstream ss;
    ss.imbue(new_locale);

    double nan = numeric_limits<double>::quiet_NaN();
    ss << nan; // Write out.
    BOOST_MATH_ASSERT(ss.str() == "nan");

    double v;
    ss >> v; // Read back in.

    cout << "NaN output was " << ss.str() << endl;
    cout << "NaN input was " << v << endl;

    // assert(nan == v); // Always fails because NaN == NaN fails!
    // assert(nan == numeric_limits<double>::quiet_NaN()); asserts!

    // And the string representation will be the native type
    // because cout has NOT been imbued with a locale containing
    // the nonfinite_numput facet.
    // So the output will be "1.#QNAN on MS platforms
    // and may be "nan" or other string representation on other platforms.
  }

} // int main()


/*
//[nonfinite_facet_sstream_output

infinity output was inf
Infinity input was inf
infinity output was inf
infinity input was 1.#INF
NaN output was nan
NaN input was 1.#QNAN

//] [nonfinite_facet_sstream_output]
*/


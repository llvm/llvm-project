
/** nonfinite_num_facet_trap.cpp
*
* Copyright (c) 2012 Paul A. Bristow
*
* Distributed under the Boost Software License, Version 1.0.
* (See accompanying file LICENSE_1_0.txt
* or copy at http://www.boost.org/LICENSE_1_0.txt)
*
* This very simple program illustrates how to use the
* `boost/math/nonfinite_num_facets.hpp` trapping output of infinity and/or NaNs.
*
\file

\brief A very simple example of using non_finite_num facet for
trapping output of infinity and/or NaNs.

\note To actually get an exception throw by the iostream library
one must enable exceptions.
  `oss.exceptions(std::ios_base::failbit | std::ios_base::badbit);`
\note Which bit is set is implementation dependent, so enable exceptions for both.

This is a fairly brutal method of catching nonfinites on output,
but may suit some applications.

*/

#ifdef _MSC_VER
#   pragma warning(disable : 4127) // conditional expression is constant.
// assumes C++ exceptions enabled /EHsc
#endif

#include <boost/cstdint.hpp>
#include <boost/math/special_functions/nonfinite_num_facets.hpp>

#include <iostream>
#include <iomanip>
using std::cout;
using std::endl;
using std::hex;
#include <exception>
#include <limits> // numeric_limits
using std::numeric_limits;

int main()
{
  using namespace boost::math;

  std::cout << "nonfinite_num_facet_trap.cpp" << std::endl;

  const double inf = +std::numeric_limits<double>::infinity ();
  const double nan = +std::numeric_limits<double>::quiet_NaN ();

  { // Output infinity and NaN with default flags (no trapping).
    std::ostringstream oss;
    std::locale default_locale (std::locale::classic ());
    std::locale C99_out_locale (default_locale, new boost::math::nonfinite_num_put<char>);
    oss.imbue (C99_out_locale);
    oss.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    oss << inf <<  ' ' << nan;
    cout << "oss.rdstate()  = " << hex << oss.rdstate() << endl; // 0
    cout << "os.str() = " << oss.str() << endl; // os.str() = inf nan
  }

  try
  { // // Output infinity with flags set to trap and catch any infinity.
    std::ostringstream oss;
    std::locale default_locale (std::locale::classic ());
    std::locale C99_out_locale (default_locale, new boost::math::nonfinite_num_put<char>(trap_infinity));
    oss.imbue (C99_out_locale);
    oss.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    // Note that which bit is set is implementation dependent, so enable exceptions for both.
    oss << inf;
    cout << "oss.rdstate()  = " << hex << oss.rdstate() << endl;
    cout << "oss.str() = " << oss.str() << endl;
  }
  catch(const std::ios_base::failure& e)
  { // Expect "Infinity".
    std::cout << "\n""Message from thrown exception was: " << e.what() << std::endl;
  }

  try
  { // // Output NaN with flags set to catch any NaNs.
    std::ostringstream oss;
    std::locale default_locale (std::locale::classic ());
    std::locale C99_out_locale (default_locale, new boost::math::nonfinite_num_put<char>(trap_nan));
    oss.imbue (C99_out_locale);
    oss.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    // Note that which bit is set is implementation dependent, so enable exceptions for both.
    oss << nan;
    cout << "oss.str() = " << oss.str() << endl;
  }
  catch(const std::ios_base::failure& e)
  { // Expect "Infinity".
    std::cout << "\n""Message from thrown exception was: " << e.what() << std::endl;
 }


  return 0; // end of nonfinite_num_facet_trap.cpp
} // int main()


/*

Output:

  nonfinite_num_facet_trap.cpp
  oss.rdstate()  = 0
  os.str() = inf nan

  Message from thrown exception was: Infinity

  Message from thrown exception was: NaN

*/

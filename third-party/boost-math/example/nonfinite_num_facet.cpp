/** nonfinite_num_facet.cpp
 *
 * Copyright (c) 2011 Francois Mauger
 * Copyright (c) 2011 Paul A. Bristow
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying file LICENSE_1_0.txt
 * or copy at http://www.boost.org/LICENSE_1_0.txt)
 *
 * This simple program illustrates how to use the
 * `boost/math/nonfinite_num_facets.hpp' material from the original
 * Floating Point  Utilities contribution by Johan Rade.
 * Floating Point Utility library has been accepted into Boost,
 * but the utilities have been/will be incorporated into Boost.Math library.
 *
\file

\brief A fairly simple example of using non_finite_num facet for
C99 standard output of infinity and NaN.

\detail  This program illustrates how to use the
 `boost/math/nonfinite_num_facets.hpp' material from the original
  Floating Point  Utilities contribution by Johan Rade.
  Floating Point Utility library has been accepted into Boost,
  but the utilities have been/will be incorporated into Boost.Math library.

  Based on an example from Francois Mauger.

  Double and float variables are assigned ordinary finite values (pi),
  and nonfinite like infinity and NaN.

  These values are then output and read back in, and then redisplayed.
  
*/

#ifdef _MSC_VER
#   pragma warning(disable : 4127) // conditional expression is constant.
#endif

#include <iostream>
#include <iomanip>
using std::cout;
using std::endl;

#include <limits> // numeric_limits
using std::numeric_limits;

#include <boost/cstdint.hpp>

#include <boost/math/special_functions/nonfinite_num_facets.hpp>

static const char sep = ','; // Separator of bracketed float and double values.

// Use max_digits10 (or equivalent) to obtain 
// all potentially significant decimal digits for the floating-point types.
    
#ifdef BOOST_NO_CXX11_NUMERIC_LIMITS
  std::streamsize  max_digits10_float = 2 + std::numeric_limits<float>::digits * 30103UL / 100000UL;
  std::streamsize  max_digits10_double = 2 + std::numeric_limits<double>::digits * 30103UL / 100000UL;
#else
  // Can use new C++0X max_digits10 (the maximum potentially significant digits).
  std::streamsize  max_digits10_float = std::numeric_limits<float>::max_digits10;
  std::streamsize  max_digits10_double = std::numeric_limits<double>::max_digits10;
#endif


/* A class with a float and a double */
struct foo
{
  foo () : fvalue (3.1415927F), dvalue (3.1415926535897931)
  {
  }
  // Set both the values to -infinity :
  void minus_infinity ()
  {
    fvalue = -std::numeric_limits<float>::infinity ();
    dvalue = -std::numeric_limits<double>::infinity ();
    return;
  }
  // Set the values to +infinity :
  void plus_infinity ()
  {
    fvalue = +std::numeric_limits<float>::infinity ();
    dvalue = +std::numeric_limits<double>::infinity ();
    return;
  }
  // Set the values to NaN :
  void nan ()
  {
    fvalue = +std::numeric_limits<float>::quiet_NaN ();
    dvalue = +std::numeric_limits<double>::quiet_NaN ();
    return;
  }
  // Print a foo:
  void print (std::ostream & a_out, const std::string & a_title)
  {
    if (a_title.empty ()) a_out << "foo";
    else a_out << a_title;
    a_out << " : " << std::endl;
    a_out << "|-- " << "fvalue = ";
    
    a_out.precision (max_digits10_float);
    a_out << fvalue << std::endl;
    a_out << "`-- " << "dvalue = ";
    a_out.precision (max_digits10_double);
    a_out << dvalue << std::endl;
    return;
  }

  // I/O operators for a foo structure of a float and a double :
  friend std::ostream & operator<< (std::ostream & a_out, const foo & a_foo);
  friend std::istream & operator>> (std::istream & a_in, foo & a_foo);

  // Attributes :
  float  fvalue; // Single precision floating number.
  double dvalue; // Double precision floating number.
};

std::ostream & operator<< (std::ostream & a_out, const foo & a_foo)
{ // Output bracketed FPs, for example "(3.1415927,3.1415926535897931)"
  a_out.precision (max_digits10_float);
  a_out << "(" << a_foo.fvalue << sep ;
  a_out.precision (max_digits10_double);
  a_out << a_foo.dvalue << ")";
  return a_out;
}

std::istream & operator>> (std::istream & a_in, foo & a_foo)
{ // Input bracketed floating-point values into a foo structure,
  // for example from "(3.1415927,3.1415926535897931)"
  char c = 0;
  a_in.get (c);
  if (c != '(')
  {
    std::cerr << "ERROR: operator>> No ( " << std::endl;
    a_in.setstate(std::ios::failbit);
    return a_in;
  }
  float f;
  a_in >> std::ws >> f;
  if (! a_in)
  {
    return a_in;
  }
  a_in >> std::ws;
  a_in.get (c);
  if (c != sep)
  {
    std::cerr << "ERROR: operator>> c='" << c << "'" << std::endl;
    std::cerr << "ERROR: operator>> No '" << sep << "'" << std::endl;
    a_in.setstate(std::ios::failbit);
    return a_in;
  }
  double d;
  a_in >> std::ws >> d;
  if (! a_in)
  {
    return a_in;
  }
  a_in >> std::ws;
  a_in.get (c);
  if (c != ')')
  {
    std::cerr << "ERROR: operator>> No ) " << std::endl;
    a_in.setstate(std::ios::failbit);
    return a_in;
  }
  a_foo.fvalue = f;
  a_foo.dvalue = d;
  return a_in;
} // std::istream & operator>> (std::istream & a_in, foo & a_foo)

int main ()
{
  std::cout << "nonfinite_num_facet simple example." << std::endl;

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

#ifdef BOOST_NO_CXX11_NUMERIC_LIMITS
  cout << "BOOST_NO_CXX11_NUMERIC_LIMITS is defined, so no max_digits10 available either:"
     "\n we'll have to calculate our own version." << endl;
#endif
  std::cout << "std::numeric_limits<float>::max_digits10 is " << max_digits10_float << endl;
  std::cout << "std::numeric_limits<double>::max_digits10 is " << max_digits10_double << endl;

   std::locale the_default_locale (std::locale::classic ());

  {
    std::cout << "Write to a string buffer (using default locale) :" << std::endl;
    foo f0; // pi
    foo f1; f1.minus_infinity ();
    foo f2; f2.plus_infinity ();
    foo f3; f3.nan ();

    f0.print (std::cout, "f0"); // pi
    f1.print (std::cout, "f1"); // +inf
    f2.print (std::cout, "f2"); // -inf
    f3.print (std::cout, "f3"); // NaN

    std::ostringstream oss;
    std::locale C99_out_locale (the_default_locale, new boost::math::nonfinite_num_put<char>);
    oss.imbue (C99_out_locale);
    oss.precision (15);
    oss << f0 << f1 << f2 << f3;
    std::cout << "Output in C99 format is: \"" << oss.str () << "\"" << std::endl;
    std::cout << "Output done." << std::endl;
  }

  {
    std::string the_string = "(3.1415927,3.1415926535897931)(-inf,-inf)(inf,inf)(nan,nan)"; // C99 format
    // Must have correct separator!
    std::cout << "Read C99 format from a string buffer containing \"" << the_string << "\""<< std::endl;

    std::locale C99_in_locale (the_default_locale, new boost::math::nonfinite_num_get<char>);
    std::istringstream iss (the_string);
    iss.imbue (C99_in_locale);

    foo f0, f1, f2, f3;
    iss >> f0 >> f1 >> f2 >> f3;
    if (! iss)
    {
       std::cerr << "Input Format error !" << std::endl;
    }
    else
    {
      std::cerr << "Input OK." << std::endl;
      cout << "Display in default locale format " << endl;
      f0.print (std::cout, "f0");
      f1.print (std::cout, "f1");
      f2.print (std::cout, "f2");
      f3.print (std::cout, "f3");
    }
    std::cout << "Input done." << std::endl;
  }
  
  std::cout << "End nonfinite_num_facet.cpp" << std::endl;
  return 0;
} // int main()

 // end of test_nonfinite_num_facets.cpp

/*

Output:

nonfinite_num_facet simple example.
  std::numeric_limits<float>::max_digits10 is 8
  std::numeric_limits<double>::max_digits10 is 17
  Write to a string buffer (using default locale) :
  f0 : 
  |-- fvalue = 3.1415927
  `-- dvalue = 3.1415926535897931
  f1 : 
  |-- fvalue = -1.#INF
  `-- dvalue = -1.#INF
  f2 : 
  |-- fvalue = 1.#INF
  `-- dvalue = 1.#INF
  f3 : 
  |-- fvalue = 1.#QNAN
  `-- dvalue = 1.#QNAN
  Output in C99 format is: "(3.1415927,3.1415926535897931)(-inf,-inf)(inf,inf)(nan,nan)"
  Output done.
  Read C99 format from a string buffer containing "(3.1415927,3.1415926535897931)(-inf,-inf)(inf,inf)(nan,nan)"
  Display in default locale format 
  f0 : 
  |-- fvalue = 3.1415927
  `-- dvalue = 3.1415926535897931
  f1 : 
  |-- fvalue = -1.#INF
  `-- dvalue = -1.#INF
  f2 : 
  |-- fvalue = 1.#INF
  `-- dvalue = 1.#INF
  f3 : 
  |-- fvalue = 1.#QNAN
  `-- dvalue = 1.#QNAN
  Input done.
  End nonfinite_num_facet.cpp

*/

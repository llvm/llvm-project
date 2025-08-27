/** nonfinite_num_facet_serialization.cpp
 *
 * Copyright (c) 2011 Francois Mauger
 * Copyright (c) 2011 Paul A. Bristow
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying file LICENSE_1_0.txt
 * or copy at http://www.boost.org/LICENSE_1_0.txt)
 *
 * This sample program by Francois Mauger illustrates how to use the
 * `boost/math/nonfinite_num_facets.hpp'  material  from the  original
 * Floating Point  Utilities contribution by  Johan Rade.  Here  it is
 * shown  how  non  finite  floating  number  can  be  serialized  and
 * deserialized from  I/O streams and/or Boost  text/XML archives.  It
 * produces two archives stored in `test.txt' and `test.xml' files.
 *
 * Tested with Boost 1.44, gcc 4.4.1, Linux/i686 (32bits).
 * Tested with Boost.1.46.1  MSVC 10.0 32 bit.
 */

#ifdef _MSC_VER
#   pragma warning(push)
//#   pragma warning(disable : 4100) // unreferenced formal parameter.
#endif

#include <iostream>
#include <sstream>
#include <fstream>
#include <limits>

#include <boost/cstdint.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/codecvt_null.hpp>

// from the Floating Point Utilities :
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
  { // Construct using 32 and 64-bit max_digits10 decimal digits value of pi.
  }
  // Set the values at -infinity :
  void minus_infinity ()
  {
    fvalue = -std::numeric_limits<float>::infinity ();
    dvalue = -std::numeric_limits<double>::infinity ();
    return;
  }
  // Set the values at +infinity :
  void plus_infinity ()
  {
    fvalue = +std::numeric_limits<float>::infinity ();
    dvalue = +std::numeric_limits<double>::infinity ();
    return;
  }
  // Set the values at NaN :
  void nan ()
  {
    fvalue = +std::numeric_limits<float>::quiet_NaN ();
    dvalue = +std::numeric_limits<double>::quiet_NaN ();
    return;
  }
  // Print :
  void print (std::ostream & a_out, const std::string & a_title)
  {
    if (a_title.empty ()) a_out << "foo";
    else a_out << a_title;
    a_out << " : " << std::endl;
    a_out << "|-- " << "fvalue = ";
    a_out.precision (7);
    a_out << fvalue << std::endl;
    a_out << "`-- " << "dvalue = ";
    a_out.precision (15);
    a_out << dvalue << std::endl;
    return;
  }

  // I/O operators :
  friend std::ostream & operator<< (std::ostream & a_out, const foo & a_foo);
  friend std::istream & operator>> (std::istream & a_in, foo & a_foo);

  // Boost serialization :
  template <class Archive>
  void serialize (Archive & ar, int /*version*/)
  {
    ar & BOOST_SERIALIZATION_NVP (fvalue);
    ar & BOOST_SERIALIZATION_NVP (dvalue);
    return;
  }

  // Attributes :
  float  fvalue; // Single precision floating-point number.
  double dvalue; // Double precision floating-point number.
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
}

int main (void)
{
  std::clog << std::endl
      << "Nonfinite_serialization.cpp' example program." << std::endl;

#ifdef BOOST_NO_CXX11_NUMERIC_LIMITS
  std::cout << "BOOST_NO_CXX11_NUMERIC_LIMITS is defined, so no max_digits10 available either,"
     "using our own version instead." << std::endl;
#endif
  std::cout << "std::numeric_limits<float>::max_digits10 is " << max_digits10_float << std::endl;
  std::cout << "std::numeric_limits<double>::max_digits10 is " << max_digits10_double << std::endl;

  std::locale the_default_locale (std::locale::classic (),
          new boost::archive::codecvt_null<char>);

  // Demonstrate use of nonfinite facets with stringstreams.
  {
    std::clog << "Construct some foo structures with a finite and nonfinites." << std::endl;
    foo f0;
    foo f1; f1.minus_infinity ();
    foo f2; f2.plus_infinity ();
    foo f3; f3.nan ();
    // Display them.
    f0.print (std::clog, "f0");
    f1.print (std::clog, "f1");
    f2.print (std::clog, "f2");
    f3.print (std::clog, "f3");
    std::clog << " Write to a string buffer." << std::endl;

    std::ostringstream oss;
    std::locale the_out_locale (the_default_locale, new boost::math::nonfinite_num_put<char>);
    oss.imbue (the_out_locale);
    oss.precision (max_digits10_double);
    oss << f0 << f1 << f2 << f3;
    std::clog << "Output is: `" << oss.str () << "'" << std::endl;
    std::clog << "Done output to ostringstream." << std::endl;
  }

  {
    std::clog << "Read foo structures from a string buffer." << std::endl;

    std::string the_string = "(3.1415927,3.1415926535897931)(-inf,-inf)(inf,inf)(nan,nan)";
    std::clog << "Input is: `" << the_string << "'" << std::endl;

    std::locale the_in_locale (the_default_locale, new boost::math::nonfinite_num_get<char>);
    std::istringstream iss (the_string);
    iss.imbue (the_in_locale);

    foo f0, f1, f2, f3;
    iss >> f0 >> f1 >> f2 >> f3;
    if (! iss)
    {
      std::cerr << "Format error !" << std::endl;
    }
    else
    {
      std::cerr << "Read OK." << std::endl;
      f0.print (std::clog, "f0");
      f1.print (std::clog, "f1");
      f2.print (std::clog, "f2");
      f3.print (std::clog, "f3");
    }
    std::clog << "Done input from istringstream." << std::endl;
  }

  {  // Demonstrate use of nonfinite facets for Serialization with Boost text archives.
    std::clog << "Serialize (using Boost text archive)." << std::endl;
    // Construct some foo structures with a finite and nonfinites.
    foo f0;
    foo f1; f1.minus_infinity ();
    foo f2; f2.plus_infinity ();
    foo f3; f3.nan ();
    // Display them.
    f0.print (std::clog, "f0");
    f1.print (std::clog, "f1");
    f2.print (std::clog, "f2");
    f3.print (std::clog, "f3");

    std::locale the_out_locale (the_default_locale, new boost::math::nonfinite_num_put<char>);
    // Use a temporary folder .temps (which contains "boost-no-inspect" so that it will not be inspected, and made 'hidden' too).
    std::ofstream fout ("./.temps/nonfinite_archive_test.txt");
    fout.imbue (the_out_locale);
    boost::archive::text_oarchive toar (fout, boost::archive::no_codecvt);
    // Write to archive.
    toar & f0;
    toar & f1;
    toar & f2;
    toar & f3;
    std::clog << "Done." << std::endl;
  }

  {
    std::clog << "Deserialize (Boost text archive)..." << std::endl;
    std::locale the_in_locale (the_default_locale, new boost::math::nonfinite_num_get<char>);
     // Use a temporary folder .temps (which contains "boost-no-inspect" so that it will not be inspected, and made 'hidden' too).
    std::ifstream fin ("./.temps/nonfinite_archive_test.txt");
    fin.imbue (the_in_locale);
    boost::archive::text_iarchive tiar (fin, boost::archive::no_codecvt);
    foo f0, f1, f2, f3;
    // Read from archive.
    tiar & f0;
    tiar & f1;
    tiar & f2;
    tiar & f3;
    // Display foos.
    f0.print (std::clog, "f0");
    f1.print (std::clog, "f1");
    f2.print (std::clog, "f2");
    f3.print (std::clog, "f3");

    std::clog << "Done." << std::endl;
  }

  {   // Demonstrate use of nonfinite facets for Serialization with Boost XML Archive.
    std::clog << "Serialize (Boost XML archive)..." << std::endl;
    // Construct some foo structures with a finite and nonfinites.
    foo f0;
    foo f1; f1.minus_infinity ();
    foo f2; f2.plus_infinity ();
    foo f3; f3.nan ();
     // Display foos.
    f0.print (std::clog, "f0");
    f1.print (std::clog, "f1");
    f2.print (std::clog, "f2");
    f3.print (std::clog, "f3");

    std::locale the_out_locale (the_default_locale, new boost::math::nonfinite_num_put<char>);
      // Use a temporary folder /.temps (which contains "boost-no-inspect" so that it will not be inspected, and made 'hidden' too).
    std::ofstream fout ("./.temps/nonfinite_XML_archive_test.txt");
    fout.imbue (the_out_locale);
    boost::archive::xml_oarchive xoar (fout, boost::archive::no_codecvt);

    xoar & BOOST_SERIALIZATION_NVP (f0);
    xoar & BOOST_SERIALIZATION_NVP (f1);
    xoar & BOOST_SERIALIZATION_NVP (f2);
    xoar & BOOST_SERIALIZATION_NVP (f3);
    std::clog << "Done." << std::endl;
  }

  {
    std::clog << "Deserialize (Boost XML archive)..." << std::endl;
    std::locale the_in_locale (the_default_locale, new boost::math::nonfinite_num_get<char>);
    // Use a temporary folder /.temps (which contains "boost-no-inspect" so that it will not be inspected, and made 'hidden' too).
    std::ifstream fin ("./.temps/nonfinite_XML_archive_test.txt");  // Previously written above.
    fin.imbue (the_in_locale);
    boost::archive::xml_iarchive xiar (fin, boost::archive::no_codecvt);
    foo f0, f1, f2, f3;

    xiar & BOOST_SERIALIZATION_NVP (f0);
    xiar & BOOST_SERIALIZATION_NVP (f1);
    xiar & BOOST_SERIALIZATION_NVP (f2);
    xiar & BOOST_SERIALIZATION_NVP (f3);

    f0.print (std::clog, "f0");
    f1.print (std::clog, "f1");
    f2.print (std::clog, "f2");
    f3.print (std::clog, "f3");

    std::clog << "Done." << std::endl;
  }

  std::clog << "End nonfinite_serialization.cpp' example program." << std::endl;
  return 0;
}

/*

Output:

  Nonfinite_serialization.cpp' example program.
  std::numeric_limits<float>::max_digits10 is 8
  std::numeric_limits<double>::max_digits10 is 17
  Construct some foo structures with a finite and nonfinites.
  f0 :
  |-- fvalue = 3.141593
  `-- dvalue = 3.14159265358979
  f1 :
  |-- fvalue = -1.#INF
  `-- dvalue = -1.#INF
  f2 :
  |-- fvalue = 1.#INF
  `-- dvalue = 1.#INF
  f3 :
  |-- fvalue = 1.#QNAN
  `-- dvalue = 1.#QNAN
   Write to a string buffer.
  Output is: `(3.1415927,3.1415926535897931)(-inf,-inf)(inf,inf)(nan,nan)'
  Done output to ostringstream.
  Read foo structures from a string buffer.
  Input is: `(3.1415927,3.1415926535897931)(-inf,-inf)(inf,inf)(nan,nan)'
  Read OK.
  f0 :
  |-- fvalue = 3.141593
  `-- dvalue = 3.14159265358979
  f1 :
  |-- fvalue = -1.#INF
  `-- dvalue = -1.#INF
  f2 :
  |-- fvalue = 1.#INF
  `-- dvalue = 1.#INF
  f3 :
  |-- fvalue = 1.#QNAN
  `-- dvalue = 1.#QNAN
  Done input from istringstream.
  Serialize (using Boost text archive).
  f0 :
  |-- fvalue = 3.141593
  `-- dvalue = 3.14159265358979
  f1 :
  |-- fvalue = -1.#INF
  `-- dvalue = -1.#INF
  f2 :
  |-- fvalue = 1.#INF
  `-- dvalue = 1.#INF
  f3 :
  |-- fvalue = 1.#QNAN
  `-- dvalue = 1.#QNAN
  Done.
  Deserialize (Boost text archive)...
  f0 :
  |-- fvalue = 3.141593
  `-- dvalue = 3.14159265358979
  f1 :
  |-- fvalue = -1.#INF
  `-- dvalue = -1.#INF
  f2 :
  |-- fvalue = 1.#INF
  `-- dvalue = 1.#INF
  f3 :
  |-- fvalue = 1.#QNAN
  `-- dvalue = 1.#QNAN
  Done.
  Serialize (Boost XML archive)...
  f0 :
  |-- fvalue = 3.141593
  `-- dvalue = 3.14159265358979
  f1 :
  |-- fvalue = -1.#INF
  `-- dvalue = -1.#INF
  f2 :
  |-- fvalue = 1.#INF
  `-- dvalue = 1.#INF
  f3 :
  |-- fvalue = 1.#QNAN
  `-- dvalue = 1.#QNAN
  Done.
  Deserialize (Boost XML archive)...
  f0 :
  |-- fvalue = 3.141593
  `-- dvalue = 3.14159265358979
  f1 :
  |-- fvalue = -1.#INF
  `-- dvalue = -1.#INF
  f2 :
  |-- fvalue = 1.#INF
  `-- dvalue = 1.#INF
  f3 :
  |-- fvalue = 1.#QNAN
  `-- dvalue = 1.#QNAN
  Done.
  End nonfinite_serialization.cpp' example program.

  */

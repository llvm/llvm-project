// Copyright 2006 Johan Rade
// Copyright 2011 Paul A. Bristow  To incorporate into Boost.Math
// Copyright 2012 Paul A. Bristow with new tests.

// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef _MSC_VER
#  pragma warning(disable : 4127) // Expression is constant.
#endif

#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include <boost/math/special_functions/nonfinite_num_facets.hpp>
#include "s_.ipp" // To create test strings like std::basic_string<CharType> s = S_("0 -0");

#include <iomanip>
#include <locale>
#include <sstream>
#include <ostream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <iostream>

namespace {

  // Using an anonymous namespace resolves ambiguities on platforms
  // with fpclassify etc functions at global scope.

  using namespace boost::math;
  using boost::math::signbit;
  using boost::math::changesign;
  using boost::math::isnan;

  //------------------------------------------------------------------------------

  template<class CharType, class ValType> void signed_zero_test_impl();
  // Loopback tests using all built-in char and floating-point types.

  BOOST_AUTO_TEST_CASE(signed_zero_test)
  {
    std::cout 
      << "BuildInfo:" << '\n'
      << "  platform "  << BOOST_PLATFORM << '\n'
      << "  compiler "  << BOOST_COMPILER << '\n'
      << "  STL "       << BOOST_STDLIB << '\n'
      << "  Boost version " << BOOST_VERSION/100000     << "."
                            << BOOST_VERSION/100 % 1000 << "."
                            << BOOST_VERSION % 100     
    << std::endl;

    signed_zero_test_impl<char, float>();
    signed_zero_test_impl<char, double>();
    signed_zero_test_impl<char, long double>();
    signed_zero_test_impl<wchar_t, float>();
    signed_zero_test_impl<wchar_t, double>();
    signed_zero_test_impl<wchar_t, long double>();
  }

  template<class CharType, class ValType> void signed_zero_test_impl()
  {


    if (signbit(static_cast<CharType>(-1e-6f) / (std::numeric_limits<CharType>::max)()) != -0)
    {
      BOOST_TEST_MESSAGE("Signed zero is not supported on this platform!");
      return;
    }

    std::locale old_locale;
    std::locale tmp_locale(
      old_locale, new nonfinite_num_put<CharType>(signed_zero));
    std::locale new_locale(tmp_locale, new nonfinite_num_get<CharType>);

    std::basic_stringstream<CharType> ss;
    ss.imbue(new_locale);

    std::basic_string<CharType> null = S_("");

    std::basic_string<CharType> s1 = S_("123");
    ss << s1 << std::endl;
    ss.str(null);


    BOOST_CHECK(ss.str() == null);  //

    ValType a1 = static_cast<ValType>(0); // zero.
    ValType a2 = (changesign)(static_cast<ValType>(0)); // negative signed zero.
    BOOST_CHECK(!(signbit)(a1)); //
    BOOST_CHECK((signbit)(a2));

    ss << a1 << ' ' << a2;

    std::basic_string<CharType> s = S_("0 -0"); // Expected.
    BOOST_CHECK(ss.str() == s);

    ValType b1, b2;
    ss >> b1 >> b2;  // Read back in.

    BOOST_CHECK(b1 == a1);
    BOOST_CHECK(b2 == a2);
    BOOST_CHECK(!(signbit)(b1));
    BOOST_CHECK((signbit)(b2));
    BOOST_CHECK(ss.rdstate() == std::ios_base::eofbit);
  }  //   template<class CharType, class ValType> void signed_zero_test_impl()

  // Checking output of types char using first default & then using signed_zero flag.
#define CHECKOUT(manips, expected)\
  {\
    {\
      std::locale old_locale;\
      std::locale tmp_locale(old_locale, new nonfinite_num_put<char>(0));\
      std::locale new_locale(tmp_locale, new nonfinite_num_get<char>);\
      std::ostringstream ss;\
      ss.imbue(new_locale);\
      ss << manips;\
      std::basic_string<char> s = S_(expected);\
      BOOST_CHECK_EQUAL(ss.str(), s);\
    }\
    {\
      std::locale old_locale;\
      std::locale tmp_locale(old_locale, new nonfinite_num_put<char>(signed_zero));\
      std::locale new_locale(tmp_locale, new nonfinite_num_get<char>);\
      std::ostringstream ss;\
      ss.imbue(new_locale);\
      ss << manips;\
      std::basic_string<char> s = S_(expected);\
      BOOST_CHECK_EQUAL(ss.str(), s);\
    }\
  }\

  BOOST_AUTO_TEST_CASE(misc_output_tests)
  { // Tests of output using a variety of output options.

     //
     // STD libraries don't all format zeros the same, 
     // so figure out what the library-specific formatting is
     // and then make sure that our facet produces the same...
     //
     bool precision_after = false;  // Prints N digits after the point rather than N digits total
     bool triple_exponent = false;  // Has 3 digits in the exponent rather than 2.
     std::stringstream ss;
     ss << std::showpoint << std::setprecision(6) << 0.0;
     if(ss.str().size() == 8)
        precision_after = true;
     ss.str("");
     ss << std::scientific << 0.0;
     triple_exponent = ss.str().size() - ss.str().find_first_of('e') == 5;



    // Positive zero.
    CHECKOUT(0, "0"); // integer zero.
    CHECKOUT(0., "0"); // double zero.
    CHECKOUT(std::setw(2) << 0., " 0");
    CHECKOUT(std::setw(4) << 0., "   0");
    CHECKOUT(std::right << std::setw(4) << 0., "   0");
    CHECKOUT(std::left << std::setw(4) << 0., "0   ");
    CHECKOUT(std::setw(4) << std::setfill('*') << 0., "***0");
    CHECKOUT(std::setw(4) << std::internal << std::setfill('*') << 0., "***0"); // left adjust sign and right adjust value.
    CHECKOUT(std::showpos << std::setw(4) << std::internal << std::setfill('*') << 0., "+**0"); // left adjust sign and right adjust value.

   if(precision_after)
   {
// BOOST_STDLIB == ("Dinkumware standard library version" BOOST_STRINGIZE(_CPPLIB_VER)) )
    CHECKOUT(std::showpoint << 0., "0.000000"); //  std::setprecision(6)
    CHECKOUT(std::setprecision(2) << std::showpoint << 0., "0.00");
   }
   else
   {
    CHECKOUT(std::showpoint << 0., "0.00000"); //  std::setprecision(6)
    CHECKOUT(std::setprecision(2) << std::showpoint << 0., "0.0");
   }
    CHECKOUT(std::fixed << std::setw(5) << std::setfill('0') << std::setprecision(2) << 0., "00.00");
    CHECKOUT(std::fixed << std::setw(6) << std::setfill('0') << std::setprecision(2) << 0., "000.00");
    CHECKOUT(std::fixed << std::setw(6) << std::setfill('0') << std::setprecision(3) << 0., "00.000");
    CHECKOUT(std::fixed << std::setw(6) << std::setfill('*') << std::setprecision(3) << 0., "*0.000");
    CHECKOUT(std::fixed << std::setw(6) << std::setfill('*') << std::setprecision(2) << std::left << 0.0, "0.00**");

    CHECKOUT(std::showpos << 0., "+0");
    CHECKOUT(std::showpos << std::fixed << std::setw(6) << std::setfill('*') << std::setprecision(2) << std::left << 0.0, "+0.00*");
   if(triple_exponent)
   {
    CHECKOUT(std::scientific << std::showpoint << std::setw(10) << std::setfill('*') << std::setprecision(1) << std::left << 0., "0.0e+000**");
   }
   else
   {
    CHECKOUT(std::scientific << std::showpoint << std::setw(10) << std::setfill('*') << std::setprecision(1) << std::left << 0., "0.0e+00***"); 
   }
    CHECKOUT(std::fixed << std::showpoint << std::setw(6) << std::setfill('*') << std::setprecision(3) << std::left << 0., "0.000*");

    double nz = (changesign)(static_cast<double>(0)); // negative signed zero.
    CHECKOUT(nz, "-0");
    // CHECKOUT(std::defaultfloat << nz, "-0"); Only for C++11
    CHECKOUT(std::showpos << nz, "-0"); // Ignore showpos because is negative.
    CHECKOUT(std::setw(2) << nz, "-0");
    CHECKOUT(std::setw(4) << nz, "  -0");
    CHECKOUT(std::right << std::setw(4) << nz, "  -0");
    CHECKOUT(std::left << std::setw(4) << nz, "-0  ");
    CHECKOUT(std::setw(4) << std::setfill('*') << nz, "**-0");
    CHECKOUT(std::setw(4) << std::internal << std::setfill('*') << nz, "-**0"); // Use std::internal to left adjust sign and right adjust value.
    CHECKOUT(std::showpos << std::setw(4) << std::internal << std::setfill('*') << nz, "-**0");

    CHECKOUT(std::fixed << std::setw(5) << std::setfill('0') << std::setprecision(2) << 0., "00.00");
    CHECKOUT(std::fixed << std::setw(6) << std::setfill('0') << std::setprecision(2) << 0., "000.00");
    CHECKOUT(std::fixed << std::setw(6) << std::setfill('0') << std::setprecision(3) << 0., "00.000");
    CHECKOUT(std::fixed << std::setw(6) << std::setfill('*') << std::setprecision(3) << 0., "*0.000");
    CHECKOUT(std::fixed << std::setw(6) << std::setfill('*') << std::setprecision(2) << std::left << 0.0, "0.00**");
    CHECKOUT(std::setprecision(2) << nz, "-0"); // No showpoint, so no decimal point nor trailing zeros.
   if(precision_after)
   {
    CHECKOUT(std::setprecision(2) << std::showpoint << nz, "-0.00"); // or "-0.0"
    CHECKOUT(std::setw(1) << std::setprecision(3) << std::showpoint << nz, "-0.000"); // Not enough width for precision overflows width.  or "-0.00"
   }
   else
   {
    CHECKOUT(std::setprecision(2) << std::showpoint << nz, "-0.0"); // or "-0.00"
    CHECKOUT(std::setw(1) << std::setprecision(3) << std::showpoint << nz, "-0.00"); // Not enough width for precision overflows width.  or "-0.000"
   }
   if(triple_exponent)
   {
    CHECKOUT(std::scientific << std::showpoint << std::setw(10) << std::setfill('*') << std::setprecision(1) << std::left << nz, "-0.0e+000*"); // -0.0e+00**
   }
   else
   {
    CHECKOUT(std::scientific << std::showpoint << std::setw(10) << std::setfill('*') << std::setprecision(1) << std::left << nz, "-0.0e+00**"); // -0.0e+000*
   }
    CHECKOUT(std::fixed << std::showpoint << std::setw(6) << std::setfill('*') << std::setprecision(3) << std::left << 0., "0.000*");

    // Non zero values.

    CHECKOUT(std::showpos << std::fixed << std::setw(6) << std::setfill('*') << std::setprecision(2) << std::left << 42., "+42.00");
    CHECKOUT(std::showpos << std::fixed << std::setw(6) << std::setfill('*') << std::setprecision(2) << std::left << 4.2, "+4.20*");
    CHECKOUT(std::showpos << std::fixed << std::setw(6) << std::setfill('*') << std::setprecision(2) << std::left << 1.22, "+1.22*");
    CHECKOUT(std::showpos << std::fixed << std::setw(6) << std::setfill('*') << std::setprecision(2) << std::left << 0.12, "+0.12*");

    CHECKOUT(std::setprecision(4) << std::showpoint << 1.2, "1.200");

  }

}   // anonymous namespace

/*

Output:

test_signed_zero.cpp
  Running 2 test cases...
  Platform: Win32
  Compiler: Microsoft Visual C++ version 10.0
  STL     : Dinkumware standard library version 520
  Boost   : 1.49.0
  Entering test suite "Master Test Suite"
  Entering test case "signed_zero_test"
  Leaving test case "signed_zero_test"; testing time: 2ms
  Entering test case "misc_output_tests"
  Leaving test case "misc_output_tests"; testing time: 15ms
  Leaving test suite "Master Test Suite"

  *** No errors detected

*/


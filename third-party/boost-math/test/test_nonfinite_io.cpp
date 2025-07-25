
// Copyright 2011 Paul A. Bristow 

// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// test_nonfinite_trap.cpp

#ifdef _MSC_VER
#  pragma warning(disable : 4127) // Expression is constant.
#endif

#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include <almost_equal.ipp> // Similar to BOOST_CLOSE_FRACTION.
#include <s_.ipp> // To create test strings like std::basic_string<CharType> s = S_("0 -0"); 
#include <boost/math/special_functions/nonfinite_num_facets.hpp>

#include <locale>
#include <sstream>
#include <iomanip>

namespace {

// Using an anonymous namespace resolves ambiguities on platforms
// with fpclassify etc functions at global scope.

using namespace boost::math;
using boost::math::signbit;
using boost::math::changesign;
using (boost::math::isnan)(;

//------------------------------------------------------------------------------
// Test nonfinite_num_put and nonfinite_num_get facets by checking
// loopback (output and re-input) of a few values,
// but using all the built-in char and floating-point types.
// Only the default output is used but various ostream options are tested separately below.
// Finite, infinite and NaN values (positive and negative) are used for the test.

void trap_test_finite();
void trap_test_inf();
void trap_test_nan();

BOOST_AUTO_TEST_CASE(trap_test)
{
    trap_test_finite();
    trap_test_inf();
    trap_test_nan();
}

//------------------------------------------------------------------------------

template<class CharType, class ValType> void trap_test_finite_impl();

void trap_test_finite()
{ // Test finite using all the built-in char and floating-point types.
    trap_test_finite_impl<char, float>();
    trap_test_finite_impl<char, double>();
    trap_test_finite_impl<char, long double>();
    trap_test_finite_impl<wchar_t, float>();
    trap_test_finite_impl<wchar_t, double>();
    trap_test_finite_impl<wchar_t, long double>();
}

template<class CharType, class ValType> void trap_test_finite_impl()
{
    std::locale old_locale;
    std::locale tmp_locale(old_locale,
        new nonfinite_num_put<CharType>(trap_infinity | trap_nan));
    std::locale new_locale(tmp_locale,
        new nonfinite_num_get<CharType>(trap_infinity | trap_nan));

    std::basic_stringstream<CharType> ss;
    ss.imbue(new_locale);

    ValType a1 = (ValType)1.2;
    ValType a2 = (ValType)-3.5;
    ValType a3 = (std::numeric_limits<ValType>::max)();
    ValType a4 = -(std::numeric_limits<ValType>::max)();
    ss << a1 << ' ' << a2 << ' ' << a3 << ' ' << a4; // 1.2, -3.5, max, -max

    ValType b1, b2, b3, b4;
    ss >> b1 >> b2 >> b3 >> b4;

    BOOST_CHECK(almost_equal(b1, a1));
    BOOST_CHECK(almost_equal(b2, a2));
    BOOST_CHECK(almost_equal(b3, a3));
    BOOST_CHECK(almost_equal(b4, a4));
    BOOST_CHECK(b3 != std::numeric_limits<ValType>::infinity());
    BOOST_CHECK(b4 != -std::numeric_limits<ValType>::infinity());
    BOOST_CHECK(ss.rdstate() == std::ios_base::eofbit);

    ss.clear();
    ss.str(S_(""));

    ss << "++5";
    ValType b5;
    ss >> b5;
    BOOST_CHECK(ss.rdstate() == std::ios_base::failbit);
}

//------------------------------------------------------------------------------

template<class CharType, class ValType> void trap_test_inf_impl();
template<class CharType, class ValType> void trap_test_put_inf_impl();
template<class CharType, class ValType> void trap_test_get_inf_impl();

void trap_test_inf()
{ // Test infinity using all the built-in char and floating-point types.
    trap_test_inf_impl<char, float>();
    trap_test_inf_impl<char, double>();
    trap_test_inf_impl<char, long double>();
    trap_test_inf_impl<wchar_t, float>();
    trap_test_inf_impl<wchar_t, double>();
    trap_test_inf_impl<wchar_t, long double>();
}

template<class CharType, class ValType> void trap_test_inf_impl()
{
    trap_test_put_inf_impl<CharType, ValType>();
    trap_test_get_inf_impl<CharType, ValType>();
}

template<class CharType, class ValType> void trap_test_put_inf_impl()
{
    std::locale old_locale;
    std::locale new_locale(old_locale,
        new nonfinite_num_put<CharType>(trap_infinity));

    std::basic_stringstream<CharType> ss;
    ss.imbue(new_locale);

    ValType a1 = std::numeric_limits<ValType>::infinity();
    ss << a1;
    BOOST_CHECK(ss.rdstate() == std::ios_base::failbit
        || ss.rdstate() == std::ios_base::badbit);
    ss.clear();

    ValType a2 = -std::numeric_limits<ValType>::infinity();
    ss << a2;
    BOOST_CHECK(ss.rdstate() == std::ios_base::failbit
        || ss.rdstate() == std::ios_base::badbit);
}

template<class CharType, class ValType> void trap_test_get_inf_impl()
{
    std::locale old_locale;
    std::locale tmp_locale(old_locale, new nonfinite_num_put<CharType>);
    std::locale new_locale(tmp_locale,
        new nonfinite_num_get<CharType>(trap_infinity));

    std::basic_stringstream<CharType> ss;
    ss.imbue(new_locale);

    ValType a1 = std::numeric_limits<ValType>::infinity();
    ss << a1;
    ValType b1;
    ss >> b1;
    BOOST_CHECK(ss.rdstate() == std::ios_base::failbit);

    ss.clear();
    ss.str(S_(""));

    ValType a2 = -std::numeric_limits<ValType>::infinity();
    ss << a2;
    ValType b2;
    ss >> b2;
    BOOST_CHECK(ss.rdstate() == std::ios_base::failbit);
}

//------------------------------------------------------------------------------

template<class CharType, class ValType> void trap_test_nan_impl();
template<class CharType, class ValType> void trap_test_put_nan_impl();
template<class CharType, class ValType> void trap_test_get_nan_impl();

void trap_test_nan()
{ // Test NaN using all the built-in char and floating-point types.
    trap_test_nan_impl<char, float>();
    trap_test_nan_impl<char, double>();
    trap_test_nan_impl<char, long double>();
    trap_test_nan_impl<wchar_t, float>();
    trap_test_nan_impl<wchar_t, double>();
    trap_test_nan_impl<wchar_t, long double>();
}

template<class CharType, class ValType> void trap_test_nan_impl()
{
    trap_test_put_nan_impl<CharType, ValType>();
    trap_test_get_nan_impl<CharType, ValType>();
}

template<class CharType, class ValType> void trap_test_put_nan_impl()
{
    std::locale old_locale;
    std::locale new_locale(old_locale,
        new nonfinite_num_put<CharType>(trap_nan));

    std::basic_stringstream<CharType> ss;
    ss.imbue(new_locale);

    ValType a1 = std::numeric_limits<ValType>::quiet_NaN();
    ss << a1;
    BOOST_CHECK(ss.rdstate() == std::ios_base::failbit
        || ss.rdstate() == std::ios_base::badbit);
    ss.clear();

    ValType a2 = std::numeric_limits<ValType>::signaling_NaN();
    ss << a2;
    BOOST_CHECK(ss.rdstate() == std::ios_base::failbit
        || ss.rdstate() == std::ios_base::badbit);
}

template<class CharType, class ValType> void trap_test_get_nan_impl()
{
    std::locale old_locale;
    std::locale tmp_locale(old_locale, new nonfinite_num_put<CharType>);
    std::locale new_locale(tmp_locale,
        new nonfinite_num_get<CharType>(trap_nan));

    std::basic_stringstream<CharType> ss;
    ss.imbue(new_locale);

    ValType a1 = std::numeric_limits<ValType>::quiet_NaN();
    ss << a1;
    ValType b1;
    ss >> b1;
    BOOST_CHECK(ss.rdstate() == std::ios_base::failbit);

    ss.clear();
    ss.str(S_(""));

    ValType a2 = std::numeric_limits<ValType>::signaling_NaN();
    ss << a2;
    ValType b2;
    ss >> b2;
    BOOST_CHECK(ss.rdstate() == std::ios_base::failbit);
}

//------------------------------------------------------------------------------

// Test a selection of stream output options comparing result with expected string.
// Only uses CharType = char and ValType  = double.
// Other types have already been tested above.

#define CHECKOUT(manips, expected)\
  {\
  std::locale old_locale;\
  std::locale tmp_locale(old_locale, new nonfinite_num_put<char>(0)); /* default flags. */\
  std::locale new_locale(tmp_locale, new nonfinite_num_get<char>);\
  std::ostringstream ss;\
  ss.imbue(new_locale);\
  ss << manips;\
  std::basic_string<char> s = S_(expected);\
  BOOST_CHECK_EQUAL(ss.str(), s);\
  }\

 BOOST_AUTO_TEST_CASE(check_trap_nan)
  { // Check that with trap_nan set, it really does throw exception.
  std::locale old_locale;
  std::locale tmp_locale(old_locale, new nonfinite_num_put<char>(trap_nan));
  std::locale new_locale(tmp_locale, new nonfinite_num_get<char>);
  std::ostringstream os;
  os.imbue(new_locale);
  os.exceptions(std::ios_base::badbit | std::ios_base::failbit); // Enable throwing exceptions.
  double nan =  std::numeric_limits<double>::quiet_NaN();
  BOOST_MATH_CHECK_THROW((os << nan), std::runtime_error);
  // warning : in "check_trap_nan": exception std::runtime_error is expected
 } //  BOOST_AUTO_TEST_CASE(check_trap_nan)

 BOOST_AUTO_TEST_CASE(check_trap_inf)
  { // Check that with trap_nan set, it really does throw exception.
  std::locale old_locale;
  std::locale tmp_locale(old_locale, new nonfinite_num_put<char>(trap_infinity));
  std::locale new_locale(tmp_locale, new nonfinite_num_get<char>);
  std::ostringstream os;
  os.imbue(new_locale);
  os.exceptions(std::ios_base::badbit | std::ios_base::failbit); // Enable throwing exceptions.
  double inf =  std::numeric_limits<double>::infinity();
  BOOST_MATH_CHECK_THROW((os << inf), std::runtime_error);
  // warning : in "check_trap_inf": exception std::runtime_error is expected.
 
 } //  BOOST_AUTO_TEST_CASE(check_trap_nan_inf)

 BOOST_AUTO_TEST_CASE(output_tests)
  {
    // Positive zero.
    CHECKOUT(0, "0"); // integer zero.
    CHECKOUT(0., "0"); // double zero.

    double nan =  std::numeric_limits<double>::quiet_NaN();
    double inf =  std::numeric_limits<double>::infinity();

    CHECKOUT(inf, "inf"); // infinity.
    CHECKOUT(-inf, "-inf"); // infinity.
    CHECKOUT(std::showpos << inf, "+inf"); // infinity.

    CHECKOUT(std::setw(6) << std::showpos << inf, "  +inf"); // infinity.
    CHECKOUT(std::right << std::setw(6) << std::showpos << inf, "  +inf"); // infinity.
    CHECKOUT(std::left << std::setw(6) << std::showpos << inf, "+inf  "); // infinity.
    CHECKOUT(std::left << std::setw(6) << std::setprecision(6) << inf, "inf   "); // infinity.
    CHECKOUT(std::left << std::setw(6) << std::setfill('*') << std::setprecision(6) << inf, "inf***"); // infinity.
    CHECKOUT(std::right << std::setw(6) << std::setfill('*') << std::setprecision(6) << inf, "***inf"); // infinity.
    CHECKOUT(std::internal<< std::setw(6) << std::showpos << inf, "+  inf"); // infinity.
    CHECKOUT(std::internal<< std::setw(6) << std::setfill('*') << std::showpos << inf, "+**inf"); // infinity.
    CHECKOUT(std::internal<< std::setw(6) << std::setfill('*') << std::showpos << -inf, "-**inf"); // infinity.

    CHECKOUT(nan, "nan"); // nan
    CHECKOUT(std::setw(1) << nan, "nan"); // nan, even if width was too small.
    CHECKOUT(std::setprecision(10) << nan, "nan"); // setprecision has no effect.
 }  //  BOOST_AUTO_TEST_CASE(output_tests)
 
}   // anonymous namespace

/*

Output:

test_nonfinite_io.cpp
  test_nonfinite_io.vcxproj -> J:\Cpp\MathToolkit\test\Math_test\Debug\test_nonfinite_io.exe
  Running 4 test cases...
  Platform: Win32
  Compiler: Microsoft Visual C++ version 10.0
  STL     : Dinkumware standard library version 520
  Boost   : 1.49.0
  Entering test suite "Master Test Suite"
  Entering test case "trap_test"
  Leaving test case "trap_test"; testing time: 7ms
  Entering test case "check_trap_nan"
  Leaving test case "check_trap_nan"
  Entering test case "check_trap_inf"
  Leaving test case "check_trap_inf"; testing time: 1ms
  Entering test case "output_tests"
  Leaving test case "output_tests"; testing time: 3ms
  Leaving test suite "Master Test Suite"
  
  *** No errors detected

*/


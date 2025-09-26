// Copyright (c) 2006 Johan Rade
// Copyright (c) 2011 Paul A. Bristow comments
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

/*!
\file
\brief Legacy (non-C99) tests of the nonfinite num facets.

\detail legacy_test outputs using nonfinite_num_put facet
with legacy flag, and reads back in using nonfinite_num_ facet,
and checks loopback OK.

Also checks that output of infinity, -infinity and NaN are as expected,
including the 'legacy' "1.#IND",  "1.#QNAN", "1.#SNAN" representations
(was used by MSVC but now all represented on output by "1.#QNAN") 
and qnan snan nanq nans (used by other systems)
excluding C99 specification  "nan -nan nan -nan" and "inf -inf".
*/

#ifdef _MSC_VER
#   pragma warning(disable : 4702)
#endif

#include <iomanip>
#include <locale>
#include <sstream>

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

//#include "almost_equal.hpp"
//#include "S_.hpp"

#include <boost/math/special_functions/nonfinite_num_facets.hpp>

namespace {

// The anonymous namespace resolves ambiguities on platforms
// with fpclassify etc functions declared at global scope.

using namespace boost::math;
using boost::math::signbit;
using boost::math::changesign;
using boost::math::isnan;

//------------------------------------------------------------------------------

void legacy_test_inf();
void legacy_test_nan();

BOOST_AUTO_TEST_CASE(legacy_test)
{
    legacy_test_inf();
    legacy_test_nan();
}

//------------------------------------------------------------------------------

template<class CharType, class ValType> void legacy_test_inf_impl();

void legacy_test_inf()
{
    legacy_test_inf_impl<char, float>();
    legacy_test_inf_impl<char, double>();
    legacy_test_inf_impl<char, long double>();
    legacy_test_inf_impl<wchar_t, float>();
    legacy_test_inf_impl<wchar_t, double>();
    legacy_test_inf_impl<wchar_t, long double>();
}

template<class CharType, class ValType> void legacy_test_inf_impl()
{
    std::locale old_locale;
    std::locale new_locale(old_locale, new nonfinite_num_get<CharType>(legacy));

    std::basic_stringstream<CharType> ss;
    ss.imbue(new_locale);

    ValType a1 = std::numeric_limits<ValType>::infinity();
    ValType a2 = -std::numeric_limits<ValType>::infinity();
    ss << a1 << ' ' << a2;

    ss << " 1.#INF";

    ValType b1, b2, b3;
    ss >> b1 >> b2 >> b3;

    BOOST_CHECK(b1 == a1);
    BOOST_CHECK(b2 == a2);
    BOOST_CHECK(b3 == std::numeric_limits<ValType>::infinity());
    BOOST_CHECK(ss.rdstate() == std::ios_base::eofbit);
}

//------------------------------------------------------------------------------

template<class CharType, class ValType> void legacy_test_nan_impl();

void legacy_test_nan()
{
    legacy_test_nan_impl<char, float>();
    legacy_test_nan_impl<char, double>();
    legacy_test_nan_impl<char, long double>();
    legacy_test_nan_impl<wchar_t, float>();
    legacy_test_nan_impl<wchar_t, double>();
    legacy_test_nan_impl<wchar_t, long double>();
}

template<class CharType, class ValType> void legacy_test_nan_impl()
{
    std::locale old_locale;
    std::locale new_locale(old_locale, new nonfinite_num_get<CharType>(legacy));

    std::basic_stringstream<CharType> ss;
    ss.imbue(new_locale);

    ValType a1 = std::numeric_limits<ValType>::quiet_NaN();
    ValType a2 = -std::numeric_limits<ValType>::quiet_NaN();
    ValType a3 = std::numeric_limits<ValType>::signaling_NaN();
    ValType a4 = -std::numeric_limits<ValType>::signaling_NaN();
    ss << a1 << ' ' << a2 << ' ' << a3 << ' ' << a4; 

    ss << " qnan snan nanq nans 1.#IND 1.#QNAN 1.#SNAN";

    ValType b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11;
    ss >> b1 >> b2 >> b3 >> b4 >> b5 >> b6 >> b7 >> b8 >> b9 >> b10 >> b11;

    // std::cout << b11 << std::endl; // Confirms that legacy
    // IND, SNAN and QNAN are considered the same,
    // and both output the legacy string "1.#QNAN".

    BOOST_CHECK((isnan)(b1));
    BOOST_CHECK((isnan)(b2));
    BOOST_CHECK((isnan)(b3));
    BOOST_CHECK((isnan)(b4));
    BOOST_CHECK((isnan)(b5));
    BOOST_CHECK((isnan)(b6));
    BOOST_CHECK((isnan)(b7));
    BOOST_CHECK((isnan)(b8));
    BOOST_CHECK((isnan)(b9));
    BOOST_CHECK((isnan)(b10));
    BOOST_CHECK((isnan)(b11));  //  Johan V3 1.#SNAN failed on MSVC 10.
    // Change in nonfinite_num_facet.hpp Paul A. Bristow 11 Apr 11 makes work OK.
/*
    // These tests fail on platforms, such as gcc,
    // that use the same representation of +nan and -nan.

    BOOST_CHECK(!(signbit)(b1));
    BOOST_CHECK((signbit)(b2));
    BOOST_CHECK(!(signbit)(b3));
    BOOST_CHECK((signbit)(b4));
*/
    BOOST_CHECK(!(signbit)(b5));
    BOOST_CHECK(!(signbit)(b6));
    BOOST_CHECK(!(signbit)(b7));
    BOOST_CHECK(!(signbit)(b8));
    BOOST_CHECK(!(signbit)(b9));
    BOOST_CHECK(!(signbit)(b10));
    BOOST_CHECK(!(signbit)(b11));  // Johan V3 1.#SNAN failed MSVC 10.

    BOOST_CHECK(ss.rdstate() == std::ios_base::eofbit); // Fails if SNAN test fails.
}

//------------------------------------------------------------------------------

}   // anonymous namespace

/*

Output:

  legacy_test.vcxproj -> J:\Cpp\fp_facet\fp_facet\Debug\legacy_test.exe
  Running 1 test case...
  1.#QNAN
  1.#QNAN
  1.#QNAN
  1.#QNAN
  1.#QNAN
  1.#QNAN
  
  *** No errors detected


*/

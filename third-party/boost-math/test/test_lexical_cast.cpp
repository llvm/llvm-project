
// Copyright (c) 2006 Johan Rade

// Copyright (c) 2011 Paul A. Bristow incorporated Boost.Math

// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

//#ifdef _MSC_VER
//#   pragma warning(disable : 4127 4511 4512 4701 4702)
//#endif

#define BOOST_TEST_MAIN

#include <limits>
#include <locale>
#include <string>
#include <boost/lexical_cast.hpp>
#include <boost/test/unit_test.hpp>

#include <boost/math/special_functions/nonfinite_num_facets.hpp>
#include <boost/math/special_functions/sign.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include "almost_equal.ipp"
#include "s_.ipp"

namespace {

// the anonymous namespace resolves ambiguities on platforms
// with fpclassify etc functions at global scope

using boost::lexical_cast;

using namespace boost::math;
using boost::math::signbit;
using boost::math::changesign;
using boost::math::isnan;

//------------------------------------------------------------------------------

template<class CharType, class ValType> void lexical_cast_test_impl();

BOOST_AUTO_TEST_CASE(lexical_cast_test)
{
    lexical_cast_test_impl<char, float>();
    lexical_cast_test_impl<char, double>();
    lexical_cast_test_impl<char, long double>();
    lexical_cast_test_impl<wchar_t, float>();
    lexical_cast_test_impl<wchar_t, double>();
    lexical_cast_test_impl<wchar_t, long double>();
}

template<class CharType, class ValType> void lexical_cast_test_impl()
{
    if((std::numeric_limits<ValType>::has_infinity == 0) || (std::numeric_limits<ValType>::infinity() == 0))
       return;
    if((std::numeric_limits<ValType>::has_quiet_NaN == 0) || (std::numeric_limits<ValType>::quiet_NaN() == 0))
       return;

    std::locale old_locale;
    std::locale tmp_locale(old_locale,
        new nonfinite_num_put<CharType>(signed_zero));
    std::locale new_locale(tmp_locale, new nonfinite_num_get<CharType>);
    std::locale::global(new_locale);

    ValType a1 = static_cast<ValType>(0);
    ValType a2 = static_cast<ValType>(13);
    ValType a3 = std::numeric_limits<ValType>::infinity();
    ValType a4 = std::numeric_limits<ValType>::quiet_NaN();
    ValType a5 = std::numeric_limits<ValType>::signaling_NaN();
    ValType a6 = (changesign)(static_cast<ValType>(0));
    ValType a7 = static_cast<ValType>(-57);
    ValType a8 = -std::numeric_limits<ValType>::infinity();
    ValType a9 = (changesign)(std::numeric_limits<ValType>::quiet_NaN()); // -NaN
    ValType a10 = (changesign)(std::numeric_limits<ValType>::signaling_NaN()); // -NaN

    std::basic_string<CharType> s1 = S_("0");
    std::basic_string<CharType> s2 = S_("13");
    std::basic_string<CharType> s3 = S_("inf");
    std::basic_string<CharType> s4 = S_("nan");
    std::basic_string<CharType> s5 = S_("nan");
    std::basic_string<CharType> s6 = S_("-0");
    std::basic_string<CharType> s7 = S_("-57");
    std::basic_string<CharType> s8 = S_("-inf");
    std::basic_string<CharType> s9 = S_("-nan");
    std::basic_string<CharType> s10 = S_("-nan");

    BOOST_CHECK(lexical_cast<std::basic_string<CharType> >(a1) == s1);
    BOOST_CHECK(lexical_cast<std::basic_string<CharType> >(a2) == s2);
    BOOST_CHECK(lexical_cast<std::basic_string<CharType> >(a3) == s3);
    BOOST_CHECK(lexical_cast<std::basic_string<CharType> >(a4) == s4);
    BOOST_CHECK(lexical_cast<std::basic_string<CharType> >(a5) == s5);
    BOOST_CHECK(lexical_cast<std::basic_string<CharType> >(a6) == s6);
    BOOST_CHECK(lexical_cast<std::basic_string<CharType> >(a7) == s7);
    BOOST_CHECK(lexical_cast<std::basic_string<CharType> >(a8) == s8);
    BOOST_CHECK(lexical_cast<std::basic_string<CharType> >(a9) == s9);
    BOOST_CHECK(lexical_cast<std::basic_string<CharType> >(a10) == s10);

    BOOST_CHECK(lexical_cast<ValType>(s1) == a1);
    BOOST_CHECK(!(signbit)(lexical_cast<ValType>(s1)));
    BOOST_CHECK(lexical_cast<ValType>(s2) == a2);
    BOOST_CHECK(lexical_cast<ValType>(s3) == a3);
    BOOST_CHECK((isnan)(lexical_cast<ValType>(s4)));
    BOOST_CHECK(!(signbit)(lexical_cast<ValType>(s4)));
    BOOST_CHECK((isnan)(lexical_cast<ValType>(s5)));
    BOOST_CHECK(!(signbit)(lexical_cast<ValType>(s5)));
    BOOST_CHECK(lexical_cast<ValType>(a6) == a6);
    BOOST_CHECK((signbit)(lexical_cast<ValType>(s6)));
    BOOST_CHECK(lexical_cast<ValType>(s7) == a7);
    BOOST_CHECK(lexical_cast<ValType>(s8) == a8);
    BOOST_CHECK((isnan)(lexical_cast<ValType>(s9)));
    BOOST_CHECK((signbit)(lexical_cast<ValType>(s9)));
    BOOST_CHECK((isnan)(lexical_cast<ValType>(s10)));
    BOOST_CHECK((signbit)(lexical_cast<ValType>(s10)));

    std::locale::global(old_locale);
}

//------------------------------------------------------------------------------

}   // anonymous namespace

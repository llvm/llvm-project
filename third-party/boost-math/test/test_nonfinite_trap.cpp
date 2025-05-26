// Copyright (c) 2006 Johan Rade
// Copyright (c) 2011 Paul A. Bristow To incorporate into Boost.Math

// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// test_nonfinite_trap.cpp

#ifdef _MSC_VER
#   pragma warning(disable : 4702)
#endif

#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include "almost_equal.ipp" // Similar to BOOST_CLOSE_FRACTION.
#include "s_.ipp" // To create test strings like std::basic_string<CharType> s = S_("0 -0"); 
#include <boost/math/special_functions/nonfinite_num_facets.hpp>

#include <locale>
#include <sstream>

namespace {

// Using an anonymous namespace resolves ambiguities on platforms
// with fpclassify etc functions at global scope.

using namespace boost::math;
using boost::math::signbit;
using boost::math::changesign;
using boost::math::isnan;

//------------------------------------------------------------------------------

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
{
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
    ss << a1 << ' ' << a2 << ' ' << a3 << ' ' << a4;

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
{
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
{
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

}   // anonymous namespace


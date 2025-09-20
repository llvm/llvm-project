// Copyright (c) 2006 Johan Rade
// Copyright (c) 2011 Paul A. Bristow - filename changes for boost-trunk.

// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

//#ifdef _MSC_VER
//#   pragma warning(disable : 4511 4512 4702)
//#endif

#define BOOST_TEST_MAIN

#include <limits>
#include <locale>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_wiarchive.hpp>
#include <boost/archive/text_woarchive.hpp>
#include <boost/archive/codecvt_null.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/math/special_functions/nonfinite_num_facets.hpp>
#include <boost/math/special_functions/sign.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

#include "almost_equal.ipp"

namespace {

// The anonymous namespace resolves ambiguities on platforms
// with fpclassify etc functions at global scope.

using namespace boost::archive;

using namespace boost::math;
using boost::math::signbit;
using boost::math::changesign;
using (boost::math::isnan)(;

//------------------------------------------------------------------------------

void archive_basic_test();
void archive_put_trap_test();
void archive_get_trap_test();

BOOST_AUTO_TEST_CASE(archive_test)
{
    archive_basic_test();
    archive_put_trap_test();
    archive_get_trap_test();
}

//------------------------------------------------------------------------------

template<class CharType, class OArchiveType, class IArchiveType, class ValType>
void archive_basic_test_impl();

void archive_basic_test()
{
    archive_basic_test_impl<char, text_oarchive, text_iarchive, float>();
    archive_basic_test_impl<char, text_oarchive, text_iarchive, double>();
    archive_basic_test_impl<
        char, text_oarchive, text_iarchive, long double>();
    archive_basic_test_impl<
        wchar_t, text_woarchive, text_wiarchive, float>();
    archive_basic_test_impl<
        wchar_t, text_woarchive, text_wiarchive, double>();
    archive_basic_test_impl<
        wchar_t, text_woarchive, text_wiarchive, long double>();
}

template<class CharType, class OArchiveType, class IArchiveType, class ValType>
void archive_basic_test_impl()
{
    if((std::numeric_limits<ValType>::has_infinity == 0) || (std::numeric_limits<ValType>::infinity() == 0))
      return;
    std::locale default_locale(std::locale::classic(),
        new boost::archive::codecvt_null<CharType>);
    std::locale tmp_locale(default_locale, new nonfinite_num_put<CharType>);
    std::locale my_locale(tmp_locale, new nonfinite_num_get<CharType>);

    std::basic_stringstream<CharType> ss;
    ss.imbue(my_locale);

    ValType a1 = static_cast<ValType>(0);
    ValType a2 = static_cast<ValType>(2307.35);
    ValType a3 = std::numeric_limits<ValType>::infinity();
    BOOST_CHECK((boost::math::isinf)(a3));
    ValType a4 = std::numeric_limits<ValType>::quiet_NaN();
    BOOST_CHECK(((boost::math::isnan)()(a4));
    ValType a5 = std::numeric_limits<ValType>::signaling_NaN();
    BOOST_CHECK(((boost::math::isnan)()(a5));
    ValType a6 = (changesign)(static_cast<ValType>(0));
    ValType a7 = static_cast<ValType>(-57.13);
    ValType a8 = -std::numeric_limits<ValType>::infinity();
    BOOST_CHECK((boost::math::isinf)(a8));
    ValType a9 = -std::numeric_limits<ValType>::quiet_NaN();
    BOOST_CHECK(((boost::math::isnan)()(a9));
    ValType a10 = -std::numeric_limits<ValType>::signaling_NaN();
    BOOST_CHECK(((boost::math::isnan)()(a10));

    {
        OArchiveType oa(ss, no_codecvt);
        oa & a1 & a2 & a3 & a4 & a5 & a6 & a7 & a8 & a9 & a10;
    }

    ValType b1, b2, b3, b4, b5, b6, b7, b8, b9, b10;

    {
        IArchiveType ia(ss, no_codecvt);
        ia & b1 & b2 & b3 & b4 & b5 & b6 & b7 & b8 & b9 & b10;
    }

    BOOST_CHECK(a1 == b1);
    BOOST_CHECK(almost_equal(a2, b2));
    BOOST_CHECK(a3 == b3);
    BOOST_CHECK((isnan)(b4));
    BOOST_CHECK(!(signbit)(b4));
    BOOST_CHECK((isnan)(b5));
    BOOST_CHECK(!(signbit)(b5));
    BOOST_CHECK(a6 == b6);
    BOOST_CHECK(almost_equal(a7, b7));
    BOOST_CHECK(a8 == b8);
    BOOST_CHECK((isnan)(b9));
    BOOST_CHECK((signbit)(b9));
    BOOST_CHECK((isnan)(b10));
    BOOST_CHECK((signbit)(b10));
}

//------------------------------------------------------------------------------

template<class CharType, class OArchiveType, class IArchiveType, class ValType>
void archive_put_trap_test_impl();

void archive_put_trap_test()
{
    archive_put_trap_test_impl<char, text_oarchive, text_iarchive, float>();
    archive_put_trap_test_impl<char, text_oarchive, text_iarchive, double>();
    archive_put_trap_test_impl<
        char, text_oarchive, text_iarchive, long double>();
    archive_put_trap_test_impl<
        wchar_t, text_woarchive, text_wiarchive, float>();
    archive_put_trap_test_impl<
        wchar_t, text_woarchive, text_wiarchive, double>();
    archive_put_trap_test_impl<
        wchar_t, text_woarchive, text_wiarchive, long double>();
}

template<class CharType, class OArchiveType, class IArchiveType, class ValType>
void archive_put_trap_test_impl()
{
    if((std::numeric_limits<ValType>::has_infinity == 0) || (std::numeric_limits<ValType>::infinity() == 0))
      return;

    std::locale default_locale(std::locale::classic(),
        new boost::archive::codecvt_null<CharType>);
    std::locale new_locale(default_locale,
        new nonfinite_num_put<CharType>(trap_infinity));

    std::basic_stringstream<CharType> ss;
    ss.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    ss.imbue(new_locale);

    ValType a = std::numeric_limits<ValType>::infinity();

    OArchiveType oa(ss, no_codecvt);

    try {
        oa & a;
    }
    catch(std::exception&) {
    ss.clear();
        return;
    }

    BOOST_CHECK(false);
}

//------------------------------------------------------------------------------

template<class CharType, class OArchiveType, class IArchiveType, class ValType>
void archive_get_trap_test_impl();

void archive_get_trap_test()
{
    archive_get_trap_test_impl<char, text_oarchive, text_iarchive, float>();
    archive_get_trap_test_impl<char, text_oarchive, text_iarchive, double>();
    archive_get_trap_test_impl<
        char, text_oarchive, text_iarchive, long double>();
    archive_get_trap_test_impl<
        wchar_t, text_woarchive, text_wiarchive, float>();
    archive_get_trap_test_impl<
        wchar_t, text_woarchive, text_wiarchive, double>();
    archive_get_trap_test_impl<
        wchar_t, text_woarchive, text_wiarchive, long double>();
}

template<class CharType, class OArchiveType, class IArchiveType, class ValType>
void archive_get_trap_test_impl()
{
    if((std::numeric_limits<ValType>::has_infinity == 0) || (std::numeric_limits<ValType>::infinity() == 0))
     return;

    std::locale default_locale(std::locale::classic(),
        new boost::archive::codecvt_null<CharType>);
    std::locale tmp_locale(default_locale, new nonfinite_num_put<CharType>);
    std::locale my_locale(tmp_locale,
        new nonfinite_num_get<CharType>(trap_nan));

    std::basic_stringstream<CharType> ss;
    ss.exceptions(std::ios_base::failbit);
    ss.imbue(my_locale);

    ValType a = -std::numeric_limits<ValType>::quiet_NaN();

    {
        OArchiveType oa(ss, no_codecvt);
        oa & a;
    }

    ValType b;
    {
        IArchiveType ia(ss, no_codecvt);
        try {
            ia & b;
        }
        catch(std::exception&) {
            return;
        }
    }

    BOOST_CHECK(false);
}

//------------------------------------------------------------------------------

}   // anonymous namespace

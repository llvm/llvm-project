#ifndef BOOST_MATH_S__HPP
#define BOOST_MATH_S__HPP

// Copyright (c) 2006 Johan Rade
// Copyright (c) 2012 Paul A. Bristow

// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// The macro S_ lets you write
//
//     basic_string<CharType> s = S_("foo");
// and
//     CharType c = S_('a');
//
// provided that CharType is either char or wchar_t.
// Used by tests of output of signed zero and others.

#ifdef _MSC_VER
#   pragma warning(push)
#   pragma warning(disable : 4512) // conditional expression is constant.
#endif

#include <string>

//------------------------------------------------------------------------------

#define S_(a) make_literal_helper(a, L##a)

class char_literal_helper {
public:
    char_literal_helper(char c, wchar_t wc) : c_(c), wc_(wc) {}
    operator char() { return c_; }
    operator wchar_t() { return wc_; }
private:
    const char c_;
    const wchar_t wc_;
};

class string_literal_helper {
public:
    string_literal_helper(const char* s, const wchar_t* ws) : s_(s), ws_(ws) {}
    operator std::string() { return s_; }
    operator std::wstring() { return ws_; }
private:
    const char* s_;
    const wchar_t* ws_;
};

inline char_literal_helper make_literal_helper(char c, wchar_t wc)
{
    return char_literal_helper(c, wc);
}

inline string_literal_helper make_literal_helper(const char* s, const wchar_t* ws)
{
    return string_literal_helper(s, ws);
}

//------------------------------------------------------------------------------

#ifdef _MSC_VER
#   pragma warning(pop)
#endif

#endif // BOOST_MATH_S__HPP


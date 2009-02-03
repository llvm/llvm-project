// Copyright 2005, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Authors: wan@google.com (Zhanyong Wan), eefacm@gmail.com (Sean Mcafee)
//
// The Google C++ Testing Framework (Google Test)
//
// This header file declares the String class and functions used internally by
// Google Test.  They are subject to change without notice. They should not used
// by code external to Google Test.
//
// This header file is #included by testing/base/internal/gtest-internal.h.
// It should not be #included by other files.

#ifndef GTEST_INCLUDE_GTEST_INTERNAL_GTEST_STRING_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_GTEST_STRING_H_

#include <string.h>
#include <gtest/internal/gtest-port.h>

#if GTEST_HAS_GLOBAL_STRING || GTEST_HAS_STD_STRING
#include <string>
#endif  // GTEST_HAS_GLOBAL_STRING || GTEST_HAS_STD_STRING

namespace testing {
namespace internal {

// String - a UTF-8 string class.
//
// We cannot use std::string as Microsoft's STL implementation in
// Visual C++ 7.1 has problems when exception is disabled.  There is a
// hack to work around this, but we've seen cases where the hack fails
// to work.
//
// Also, String is different from std::string in that it can represent
// both NULL and the empty string, while std::string cannot represent
// NULL.
//
// NULL and the empty string are considered different.  NULL is less
// than anything (including the empty string) except itself.
//
// This class only provides minimum functionality necessary for
// implementing Google Test.  We do not intend to implement a full-fledged
// string class here.
//
// Since the purpose of this class is to provide a substitute for
// std::string on platforms where it cannot be used, we define a copy
// constructor and assignment operators such that we don't need
// conditional compilation in a lot of places.
//
// In order to make the representation efficient, the d'tor of String
// is not virtual.  Therefore DO NOT INHERIT FROM String.
class String {
 public:
  // Static utility methods

  // Returns the input if it's not NULL, otherwise returns "(null)".
  // This function serves two purposes:
  //
  // 1. ShowCString(NULL) has type 'const char *', instead of the
  // type of NULL (which is int).
  //
  // 2. In MSVC, streaming a null char pointer to StrStream generates
  // an access violation, so we need to convert NULL to "(null)"
  // before streaming it.
  static inline const char* ShowCString(const char* c_str) {
    return c_str ? c_str : "(null)";
  }

  // Returns the input enclosed in double quotes if it's not NULL;
  // otherwise returns "(null)".  For example, "\"Hello\"" is returned
  // for input "Hello".
  //
  // This is useful for printing a C string in the syntax of a literal.
  //
  // Known issue: escape sequences are not handled yet.
  static String ShowCStringQuoted(const char* c_str);

  // Clones a 0-terminated C string, allocating memory using new.  The
  // caller is responsible for deleting the return value using
  // delete[].  Returns the cloned string, or NULL if the input is
  // NULL.
  //
  // This is different from strdup() in string.h, which allocates
  // memory using malloc().
  static const char* CloneCString(const char* c_str);

#ifdef _WIN32_WCE
  // Windows CE does not have the 'ANSI' versions of Win32 APIs. To be
  // able to pass strings to Win32 APIs on CE we need to convert them
  // to 'Unicode', UTF-16.

  // Creates a UTF-16 wide string from the given ANSI string, allocating
  // memory using new. The caller is responsible for deleting the return
  // value using delete[]. Returns the wide string, or NULL if the
  // input is NULL.
  //
  // The wide string is created using the ANSI codepage (CP_ACP) to
  // match the behaviour of the ANSI versions of Win32 calls and the
  // C runtime.
  static LPCWSTR AnsiToUtf16(const char* c_str);

  // Creates an ANSI string from the given wide string, allocating
  // memory using new. The caller is responsible for deleting the return
  // value using delete[]. Returns the ANSI string, or NULL if the
  // input is NULL.
  //
  // The returned string is created using the ANSI codepage (CP_ACP) to
  // match the behaviour of the ANSI versions of Win32 calls and the
  // C runtime.
  static const char* Utf16ToAnsi(LPCWSTR utf16_str);
#endif

  // Compares two C strings.  Returns true iff they have the same content.
  //
  // Unlike strcmp(), this function can handle NULL argument(s).  A
  // NULL C string is considered different to any non-NULL C string,
  // including the empty string.
  static bool CStringEquals(const char* lhs, const char* rhs);

  // Converts a wide C string to a String using the UTF-8 encoding.
  // NULL will be converted to "(null)".  If an error occurred during
  // the conversion, "(failed to convert from wide string)" is
  // returned.
  static String ShowWideCString(const wchar_t* wide_c_str);

  // Similar to ShowWideCString(), except that this function encloses
  // the converted string in double quotes.
  static String ShowWideCStringQuoted(const wchar_t* wide_c_str);

  // Compares two wide C strings.  Returns true iff they have the same
  // content.
  //
  // Unlike wcscmp(), this function can handle NULL argument(s).  A
  // NULL C string is considered different to any non-NULL C string,
  // including the empty string.
  static bool WideCStringEquals(const wchar_t* lhs, const wchar_t* rhs);

  // Compares two C strings, ignoring case.  Returns true iff they
  // have the same content.
  //
  // Unlike strcasecmp(), this function can handle NULL argument(s).
  // A NULL C string is considered different to any non-NULL C string,
  // including the empty string.
  static bool CaseInsensitiveCStringEquals(const char* lhs,
                                           const char* rhs);

  // Compares two wide C strings, ignoring case.  Returns true iff they
  // have the same content.
  //
  // Unlike wcscasecmp(), this function can handle NULL argument(s).
  // A NULL C string is considered different to any non-NULL wide C string,
  // including the empty string.
  // NB: The implementations on different platforms slightly differ.
  // On windows, this method uses _wcsicmp which compares according to LC_CTYPE
  // environment variable. On GNU platform this method uses wcscasecmp
  // which compares according to LC_CTYPE category of the current locale.
  // On MacOS X, it uses towlower, which also uses LC_CTYPE category of the
  // current locale.
  static bool CaseInsensitiveWideCStringEquals(const wchar_t* lhs,
                                               const wchar_t* rhs);

  // Formats a list of arguments to a String, using the same format
  // spec string as for printf.
  //
  // We do not use the StringPrintf class as it is not universally
  // available.
  //
  // The result is limited to 4096 characters (including the tailing
  // 0).  If 4096 characters are not enough to format the input,
  // "<buffer exceeded>" is returned.
  static String Format(const char* format, ...);

  // C'tors

  // The default c'tor constructs a NULL string.
  String() : c_str_(NULL) {}

  // Constructs a String by cloning a 0-terminated C string.
  String(const char* c_str) : c_str_(NULL) {  // NOLINT
    *this = c_str;
  }

  // Constructs a String by copying a given number of chars from a
  // buffer.  E.g. String("hello", 3) will create the string "hel".
  String(const char* buffer, size_t len);

  // The copy c'tor creates a new copy of the string.  The two
  // String objects do not share content.
  String(const String& str) : c_str_(NULL) {
    *this = str;
  }

  // D'tor.  String is intended to be a final class, so the d'tor
  // doesn't need to be virtual.
  ~String() { delete[] c_str_; }

  // Allows a String to be implicitly converted to an ::std::string or
  // ::string, and vice versa.  Converting a String containing a NULL
  // pointer to ::std::string or ::string is undefined behavior.
  // Converting a ::std::string or ::string containing an embedded NUL
  // character to a String will result in the prefix up to the first
  // NUL character.
#if GTEST_HAS_STD_STRING
  String(const ::std::string& str) : c_str_(NULL) { *this = str.c_str(); }

  operator ::std::string() const { return ::std::string(c_str_); }
#endif  // GTEST_HAS_STD_STRING

#if GTEST_HAS_GLOBAL_STRING
  String(const ::string& str) : c_str_(NULL) { *this = str.c_str(); }

  operator ::string() const { return ::string(c_str_); }
#endif  // GTEST_HAS_GLOBAL_STRING

  // Returns true iff this is an empty string (i.e. "").
  bool empty() const {
    return (c_str_ != NULL) && (*c_str_ == '\0');
  }

  // Compares this with another String.
  // Returns < 0 if this is less than rhs, 0 if this is equal to rhs, or > 0
  // if this is greater than rhs.
  int Compare(const String& rhs) const;

  // Returns true iff this String equals the given C string.  A NULL
  // string and a non-NULL string are considered not equal.
  bool operator==(const char* c_str) const {
    return CStringEquals(c_str_, c_str);
  }

  // Returns true iff this String is less than the given C string.  A NULL
  // string is considered less than "".
  bool operator<(const String& rhs) const { return Compare(rhs) < 0; }

  // Returns true iff this String doesn't equal the given C string.  A NULL
  // string and a non-NULL string are considered not equal.
  bool operator!=(const char* c_str) const {
    return !CStringEquals(c_str_, c_str);
  }

  // Returns true iff this String ends with the given suffix.  *Any*
  // String is considered to end with a NULL or empty suffix.
  bool EndsWith(const char* suffix) const;

  // Returns true iff this String ends with the given suffix, not considering
  // case. Any String is considered to end with a NULL or empty suffix.
  bool EndsWithCaseInsensitive(const char* suffix) const;

  // Returns the length of the encapsulated string, or -1 if the
  // string is NULL.
  int GetLength() const {
    return c_str_ ? static_cast<int>(strlen(c_str_)) : -1;
  }

  // Gets the 0-terminated C string this String object represents.
  // The String object still owns the string.  Therefore the caller
  // should NOT delete the return value.
  const char* c_str() const { return c_str_; }

  // Sets the 0-terminated C string this String object represents.
  // The old string in this object is deleted, and this object will
  // own a clone of the input string.  This function copies only up to
  // length bytes (plus a terminating null byte), or until the first
  // null byte, whichever comes first.
  //
  // This function works even when the c_str parameter has the same
  // value as that of the c_str_ field.
  void Set(const char* c_str, size_t length);

  // Assigns a C string to this object.  Self-assignment works.
  const String& operator=(const char* c_str);

  // Assigns a String object to this object.  Self-assignment works.
  const String& operator=(const String &rhs) {
    *this = rhs.c_str_;
    return *this;
  }

 private:
  const char* c_str_;
};

// Streams a String to an ostream.
inline ::std::ostream& operator <<(::std::ostream& os, const String& str) {
  // We call String::ShowCString() to convert NULL to "(null)".
  // Otherwise we'll get an access violation on Windows.
  return os << String::ShowCString(str.c_str());
}

// Gets the content of the StrStream's buffer as a String.  Each '\0'
// character in the buffer is replaced with "\\0".
String StrStreamToString(StrStream* stream);

// Converts a streamable value to a String.  A NULL pointer is
// converted to "(null)".  When the input value is a ::string,
// ::std::string, ::wstring, or ::std::wstring object, each NUL
// character in it is replaced with "\\0".

// Declared here but defined in gtest.h, so that it has access
// to the definition of the Message class, required by the ARM
// compiler.
template <typename T>
String StreamableToString(const T& streamable);

}  // namespace internal
}  // namespace testing

#endif  // GTEST_INCLUDE_GTEST_INTERNAL_GTEST_STRING_H_

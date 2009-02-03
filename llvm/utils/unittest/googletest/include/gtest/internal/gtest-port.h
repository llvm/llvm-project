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
// Authors: wan@google.com (Zhanyong Wan)
//
// Low-level types and utilities for porting Google Test to various
// platforms.  They are subject to change without notice.  DO NOT USE
// THEM IN USER CODE.

#ifndef GTEST_INCLUDE_GTEST_INTERNAL_GTEST_PORT_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_GTEST_PORT_H_

// The user can define the following macros in the build script to
// control Google Test's behavior.  If the user doesn't define a macro
// in this list, Google Test will define it.
//
//   GTEST_HAS_CLONE          - Define it to 1/0 to indicate that clone(2)
//                              is/isn't available.
//   GTEST_HAS_GLOBAL_STRING  - Define it to 1/0 to indicate that ::string
//                              is/isn't available (some systems define
//                              ::string, which is different to std::string).
//   GTEST_HAS_GLOBAL_WSTRING - Define it to 1/0 to indicate that ::string
//                              is/isn't available (some systems define
//                              ::wstring, which is different to std::wstring).
//   GTEST_HAS_PTHREAD        - Define it to 1/0 to indicate that <pthread.h>
//                              is/isn't available.
//   GTEST_HAS_RTTI           - Define it to 1/0 to indicate that RTTI is/isn't
//                              enabled.
//   GTEST_HAS_STD_STRING     - Define it to 1/0 to indicate that
//                              std::string does/doesn't work (Google Test can
//                              be used where std::string is unavailable).
//   GTEST_HAS_STD_WSTRING    - Define it to 1/0 to indicate that
//                              std::wstring does/doesn't work (Google Test can
//                              be used where std::wstring is unavailable).
//   GTEST_HAS_TR1_TUPLE 1    - Define it to 1/0 to indicate tr1::tuple
//                              is/isn't available.

// This header defines the following utilities:
//
// Macros indicating the name of the Google C++ Testing Framework project:
//   GTEST_NAME              - a string literal of the project name.
//   GTEST_FLAG_PREFIX       - a string literal of the prefix all Google
//                             Test flag names share.
//   GTEST_FLAG_PREFIX_UPPER - a string literal of the prefix all Google
//                             Test flag names share, in upper case.
//
// Macros indicating the current platform:
//   GTEST_OS_CYGWIN   - defined iff compiled on Cygwin.
//   GTEST_OS_LINUX    - defined iff compiled on Linux.
//   GTEST_OS_MAC      - defined iff compiled on Mac OS X.
//   GTEST_OS_SOLARIS  - defined iff compiled on Sun Solaris.
//   GTEST_OS_SYMBIAN  - defined iff compiled for Symbian.
//   GTEST_OS_WINDOWS  - defined iff compiled on Windows.
//   GTEST_OS_ZOS      - defined iff compiled on IBM z/OS.
//
// Among the platforms, Cygwin, Linux, Max OS X, and Windows have the
// most stable support.  Since core members of the Google Test project
// don't have access to other platforms, support for them may be less
// stable.  If you notice any problems on your platform, please notify
// googletestframework@googlegroups.com (patches for fixing them are
// even more welcome!).
//
// Note that it is possible that none of the GTEST_OS_ macros are defined.
//
// Macros indicating available Google Test features:
//   GTEST_HAS_COMBINE      - defined iff Combine construct is supported
//                            in value-parameterized tests.
//   GTEST_HAS_DEATH_TEST   - defined iff death tests are supported.
//   GTEST_HAS_PARAM_TEST   - defined iff value-parameterized tests are
//                            supported.
//   GTEST_HAS_TYPED_TEST   - defined iff typed tests are supported.
//   GTEST_HAS_TYPED_TEST_P - defined iff type-parameterized tests are
//                            supported.
//
// Macros for basic C++ coding:
//   GTEST_AMBIGUOUS_ELSE_BLOCKER_ - for disabling a gcc warning.
//   GTEST_ATTRIBUTE_UNUSED_  - declares that a class' instances don't have to
//                              be used.
//   GTEST_DISALLOW_COPY_AND_ASSIGN_ - disables copy ctor and operator=.
//   GTEST_MUST_USE_RESULT_   - declares that a function's result must be used.
//
// Synchronization:
//   Mutex, MutexLock, ThreadLocal, GetThreadCount()
//                  - synchronization primitives.
//   GTEST_IS_THREADSAFE - defined to 1 to indicate that the above
//                         synchronization primitives have real implementations
//                         and Google Test is thread-safe; or 0 otherwise.
//
// Template meta programming:
//   is_pointer     - as in TR1; needed on Symbian and IBM XL C/C++ only.
//
// Smart pointers:
//   scoped_ptr     - as in TR2.
//
// Regular expressions:
//   RE             - a simple regular expression class using the POSIX
//                    Extended Regular Expression syntax.  Not available on
//                    Windows.
//
// Logging:
//   GTEST_LOG_()   - logs messages at the specified severity level.
//   LogToStderr()  - directs all log messages to stderr.
//   FlushInfoLog() - flushes informational log messages.
//
// Stderr capturing:
//   CaptureStderr()     - starts capturing stderr.
//   GetCapturedStderr() - stops capturing stderr and returns the captured
//                         string.
//
// Integer types:
//   TypeWithSize   - maps an integer to a int type.
//   Int32, UInt32, Int64, UInt64, TimeInMillis
//                  - integers of known sizes.
//   BiggestInt     - the biggest signed integer type.
//
// Command-line utilities:
//   GTEST_FLAG()       - references a flag.
//   GTEST_DECLARE_*()  - declares a flag.
//   GTEST_DEFINE_*()   - defines a flag.
//   GetArgvs()         - returns the command line as a vector of strings.
//
// Environment variable utilities:
//   GetEnv()             - gets the value of an environment variable.
//   BoolFromGTestEnv()   - parses a bool environment variable.
//   Int32FromGTestEnv()  - parses an Int32 environment variable.
//   StringFromGTestEnv() - parses a string environment variable.

#include <stdlib.h>
#include <stdio.h>
#include <iostream>  // Used for GTEST_CHECK_

#define GTEST_NAME "Google Test"
#define GTEST_FLAG_PREFIX "gtest_"
#define GTEST_FLAG_PREFIX_UPPER "GTEST_"

// Determines the version of gcc that is used to compile this.
#ifdef __GNUC__
// 40302 means version 4.3.2.
#define GTEST_GCC_VER_ \
    (__GNUC__*10000 + __GNUC_MINOR__*100 + __GNUC_PATCHLEVEL__)
#endif  // __GNUC__

// Determines the platform on which Google Test is compiled.
#ifdef __CYGWIN__
#define GTEST_OS_CYGWIN
#elif __SYMBIAN32__
#define GTEST_OS_SYMBIAN
#elif defined _MSC_VER
// TODO(kenton@google.com): GTEST_OS_WINDOWS is currently used to mean
//   both "The OS is Windows" and "The compiler is MSVC".  These
//   meanings really should be separated in order to better support
//   Windows compilers other than MSVC.
#define GTEST_OS_WINDOWS
#elif defined __APPLE__
#define GTEST_OS_MAC
#elif defined __linux__
#define GTEST_OS_LINUX
#elif defined __MVS__
#define GTEST_OS_ZOS
#elif defined(__sun) && defined(__SVR4)
#define GTEST_OS_SOLARIS
#endif  // _MSC_VER

// Determines whether ::std::string and ::string are available.

#ifndef GTEST_HAS_STD_STRING
// The user didn't tell us whether ::std::string is available, so we
// need to figure it out.

#ifdef GTEST_OS_WINDOWS
// Assumes that exceptions are enabled by default.
#ifndef _HAS_EXCEPTIONS
#define _HAS_EXCEPTIONS 1
#endif  // _HAS_EXCEPTIONS
// GTEST_HAS_EXCEPTIONS is non-zero iff exceptions are enabled.  It is
// always defined, while _HAS_EXCEPTIONS is defined only on Windows.
#define GTEST_HAS_EXCEPTIONS _HAS_EXCEPTIONS
// On Windows, we can use ::std::string if the compiler version is VS
// 2005 or above, or if exceptions are enabled.
#define GTEST_HAS_STD_STRING ((_MSC_VER >= 1400) || GTEST_HAS_EXCEPTIONS)
#else  // We are on Linux or Mac OS.
#define GTEST_HAS_EXCEPTIONS 0
#define GTEST_HAS_STD_STRING 1
#endif  // GTEST_OS_WINDOWS

#endif  // GTEST_HAS_STD_STRING

#ifndef GTEST_HAS_GLOBAL_STRING
// The user didn't tell us whether ::string is available, so we need
// to figure it out.

#define GTEST_HAS_GLOBAL_STRING 0

#endif  // GTEST_HAS_GLOBAL_STRING

#ifndef GTEST_HAS_STD_WSTRING
// The user didn't tell us whether ::std::wstring is available, so we need
// to figure it out.
// TODO(wan@google.com): uses autoconf to detect whether ::std::wstring
//   is available.

#if defined(GTEST_OS_CYGWIN) || defined(GTEST_OS_SOLARIS)
// At least some versions of cygwin don't support ::std::wstring.
// Solaris' libc++ doesn't support it either.
#define GTEST_HAS_STD_WSTRING 0
#else
#define GTEST_HAS_STD_WSTRING GTEST_HAS_STD_STRING
#endif  // defined(GTEST_OS_CYGWIN) || defined(GTEST_OS_SOLARIS)

#endif  // GTEST_HAS_STD_WSTRING

#ifndef GTEST_HAS_GLOBAL_WSTRING
// The user didn't tell us whether ::wstring is available, so we need
// to figure it out.
#define GTEST_HAS_GLOBAL_WSTRING GTEST_HAS_GLOBAL_STRING
#endif  // GTEST_HAS_GLOBAL_WSTRING

#if GTEST_HAS_STD_STRING || GTEST_HAS_GLOBAL_STRING || \
    GTEST_HAS_STD_WSTRING || GTEST_HAS_GLOBAL_WSTRING
#include <string>  // NOLINT
#endif  // GTEST_HAS_STD_STRING || GTEST_HAS_GLOBAL_STRING ||
        // GTEST_HAS_STD_WSTRING || GTEST_HAS_GLOBAL_WSTRING

#if GTEST_HAS_STD_STRING
#include <sstream>  // NOLINT
#else
#include <strstream>  // NOLINT
#endif  // GTEST_HAS_STD_STRING

// Determines whether RTTI is available.
#ifndef GTEST_HAS_RTTI
// The user didn't tell us whether RTTI is enabled, so we need to
// figure it out.

#ifdef _MSC_VER

#ifdef _CPPRTTI  // MSVC defines this macro iff RTTI is enabled.
#define GTEST_HAS_RTTI 1
#else
#define GTEST_HAS_RTTI 0
#endif  // _CPPRTTI

#elif defined(__GNUC__)

// Starting with version 4.3.2, gcc defines __GXX_RTTI iff RTTI is enabled.
#if GTEST_GCC_VER_ >= 40302
#ifdef __GXX_RTTI
#define GTEST_HAS_RTTI 1
#else
#define GTEST_HAS_RTTI 0
#endif  // __GXX_RTTI
#else
// For gcc versions smaller than 4.3.2, we assume RTTI is enabled.
#define GTEST_HAS_RTTI 1
#endif  // GTEST_GCC_VER >= 40302

#else

// Unknown compiler - assume RTTI is enabled.
#define GTEST_HAS_RTTI 1

#endif  // _MSC_VER

#endif  // GTEST_HAS_RTTI

// Determines whether <pthread.h> is available.
#ifndef GTEST_HAS_PTHREAD
// The user didn't tell us, so we need to figure it out.

#if defined(GTEST_OS_LINUX) || defined(GTEST_OS_MAC)
#define GTEST_HAS_PTHREAD 1
#else
#define GTEST_HAS_PTHREAD 0
#endif  // GTEST_OS_LINUX || GTEST_OS_MAC

#endif  // GTEST_HAS_PTHREAD

// Determines whether tr1/tuple is available.  If you have tr1/tuple
// on your platform, define GTEST_HAS_TR1_TUPLE=1 for both the Google
// Test project and your tests. If you would like Google Test to detect
// tr1/tuple on your platform automatically, please open an issue
// ticket at http://code.google.com/p/googletest.
#ifndef GTEST_HAS_TR1_TUPLE
// The user didn't tell us, so we need to figure it out.

// GCC provides <tr1/tuple> since 4.0.0.
#if defined(__GNUC__) && (GTEST_GCC_VER_ >= 40000)
#define GTEST_HAS_TR1_TUPLE 1
#else
#define GTEST_HAS_TR1_TUPLE 0
#endif  // __GNUC__
#endif  // GTEST_HAS_TR1_TUPLE

// To avoid conditional compilation everywhere, we make it
// gtest-port.h's responsibility to #include the header implementing
// tr1/tuple.
#if GTEST_HAS_TR1_TUPLE
#if defined(__GNUC__)
// GCC implements tr1/tuple in the <tr1/tuple> header.  This does not
// conform to the TR1 spec, which requires the header to be <tuple>.
#include <tr1/tuple>
#else
// If the compiler is not GCC, we assume the user is using a
// spec-conforming TR1 implementation.
#include <tuple>
#endif  // __GNUC__
#endif  // GTEST_HAS_TR1_TUPLE

// Determines whether clone(2) is supported.
// Usually it will only be available on Linux, excluding
// Linux on the Itanium architecture.
// Also see http://linux.die.net/man/2/clone.
#ifndef GTEST_HAS_CLONE
// The user didn't tell us, so we need to figure it out.

#if defined(GTEST_OS_LINUX) && !defined(__ia64__)
#define GTEST_HAS_CLONE 1
#else
#define GTEST_HAS_CLONE 0
#endif  // defined(GTEST_OS_LINUX) && !defined(__ia64__)

#endif  // GTEST_HAS_CLONE

// Determines whether to support death tests.
#if GTEST_HAS_STD_STRING && GTEST_HAS_CLONE
#define GTEST_HAS_DEATH_TEST
// On some platforms, <regex.h> needs someone to define size_t, and
// won't compile otherwise.  We can #include it here as we already
// included <stdlib.h>, which is guaranteed to define size_t through
// <stddef.h>.
#include <regex.h>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#endif  // GTEST_HAS_STD_STRING && GTEST_HAS_CLONE

// Determines whether to support value-parameterized tests.

#if defined(__GNUC__) || (_MSC_VER >= 1400)
// TODO(vladl@google.com): get the implementation rid of vector and list
// to compile on MSVC 7.1.
#define GTEST_HAS_PARAM_TEST
#endif  // defined(__GNUC__) || (_MSC_VER >= 1400)

// Determines whether to support type-driven tests.

// Typed tests need <typeinfo> and variadic macros, which gcc and VC
// 8.0+ support.
#if defined(__GNUC__) || (_MSC_VER >= 1400)
#define GTEST_HAS_TYPED_TEST
#define GTEST_HAS_TYPED_TEST_P
#endif  // defined(__GNUC__) || (_MSC_VER >= 1400)

// Determines whether to support Combine(). This only makes sense when
// value-parameterized tests are enabled.
#if defined(GTEST_HAS_PARAM_TEST) && GTEST_HAS_TR1_TUPLE
#define GTEST_HAS_COMBINE
#endif  // defined(GTEST_HAS_PARAM_TEST) && GTEST_HAS_TR1_TUPLE

// Determines whether the system compiler uses UTF-16 for encoding wide strings.
#if defined(GTEST_OS_WINDOWS) || defined(GTEST_OS_CYGWIN) || \
        defined(GTEST_OS_SYMBIAN)
#define GTEST_WIDE_STRING_USES_UTF16_ 1
#endif

// Defines some utility macros.

// The GNU compiler emits a warning if nested "if" statements are followed by
// an "else" statement and braces are not used to explicitly disambiguate the
// "else" binding.  This leads to problems with code like:
//
//   if (gate)
//     ASSERT_*(condition) << "Some message";
//
// The "switch (0) case 0:" idiom is used to suppress this.
#ifdef __INTEL_COMPILER
#define GTEST_AMBIGUOUS_ELSE_BLOCKER_
#else
#define GTEST_AMBIGUOUS_ELSE_BLOCKER_ switch (0) case 0:  // NOLINT
#endif

// Use this annotation at the end of a struct / class definition to
// prevent the compiler from optimizing away instances that are never
// used.  This is useful when all interesting logic happens inside the
// c'tor and / or d'tor.  Example:
//
//   struct Foo {
//     Foo() { ... }
//   } GTEST_ATTRIBUTE_UNUSED_;
#if defined(__GNUC__) && !defined(COMPILER_ICC)
#define GTEST_ATTRIBUTE_UNUSED_ __attribute__ ((unused))
#else
#define GTEST_ATTRIBUTE_UNUSED_
#endif

// A macro to disallow the evil copy constructor and operator= functions
// This should be used in the private: declarations for a class.
#define GTEST_DISALLOW_COPY_AND_ASSIGN_(type)\
  type(const type &);\
  void operator=(const type &)

// Tell the compiler to warn about unused return values for functions declared
// with this macro.  The macro should be used on function declarations
// following the argument list:
//
//   Sprocket* AllocateSprocket() GTEST_MUST_USE_RESULT_;
#if defined(__GNUC__) && (GTEST_GCC_VER_ >= 30400) && !defined(COMPILER_ICC)
#define GTEST_MUST_USE_RESULT_ __attribute__ ((warn_unused_result))
#else
#define GTEST_MUST_USE_RESULT_
#endif  // __GNUC__ && (GTEST_GCC_VER_ >= 30400) && !COMPILER_ICC

namespace testing {

class Message;

namespace internal {

class String;

// std::strstream is deprecated.  However, we have to use it on
// Windows as std::stringstream won't compile on Windows when
// exceptions are disabled.  We use std::stringstream on other
// platforms to avoid compiler warnings there.
#if GTEST_HAS_STD_STRING
typedef ::std::stringstream StrStream;
#else
typedef ::std::strstream StrStream;
#endif  // GTEST_HAS_STD_STRING

// Defines scoped_ptr.

// This implementation of scoped_ptr is PARTIAL - it only contains
// enough stuff to satisfy Google Test's need.
template <typename T>
class scoped_ptr {
 public:
  explicit scoped_ptr(T* p = NULL) : ptr_(p) {}
  ~scoped_ptr() { reset(); }

  T& operator*() const { return *ptr_; }
  T* operator->() const { return ptr_; }
  T* get() const { return ptr_; }

  T* release() {
    T* const ptr = ptr_;
    ptr_ = NULL;
    return ptr;
  }

  void reset(T* p = NULL) {
    if (p != ptr_) {
      if (sizeof(T) > 0) {  // Makes sure T is a complete type.
        delete ptr_;
      }
      ptr_ = p;
    }
  }
 private:
  T* ptr_;

  GTEST_DISALLOW_COPY_AND_ASSIGN_(scoped_ptr);
};

#ifdef GTEST_HAS_DEATH_TEST

// Defines RE.

// A simple C++ wrapper for <regex.h>.  It uses the POSIX Enxtended
// Regular Expression syntax.
class RE {
 public:
  // Constructs an RE from a string.
#if GTEST_HAS_STD_STRING
  RE(const ::std::string& regex) { Init(regex.c_str()); }  // NOLINT
#endif  // GTEST_HAS_STD_STRING

#if GTEST_HAS_GLOBAL_STRING
  RE(const ::string& regex) { Init(regex.c_str()); }  // NOLINT
#endif  // GTEST_HAS_GLOBAL_STRING

  RE(const char* regex) { Init(regex); }  // NOLINT
  ~RE();

  // Returns the string representation of the regex.
  const char* pattern() const { return pattern_; }

  // FullMatch(str, re) returns true iff regular expression re matches
  // the entire str.
  // PartialMatch(str, re) returns true iff regular expression re
  // matches a substring of str (including str itself).
  //
  // TODO(wan@google.com): make FullMatch() and PartialMatch() work
  // when str contains NUL characters.
#if GTEST_HAS_STD_STRING
  static bool FullMatch(const ::std::string& str, const RE& re) {
    return FullMatch(str.c_str(), re);
  }
  static bool PartialMatch(const ::std::string& str, const RE& re) {
    return PartialMatch(str.c_str(), re);
  }
#endif  // GTEST_HAS_STD_STRING

#if GTEST_HAS_GLOBAL_STRING
  static bool FullMatch(const ::string& str, const RE& re) {
    return FullMatch(str.c_str(), re);
  }
  static bool PartialMatch(const ::string& str, const RE& re) {
    return PartialMatch(str.c_str(), re);
  }
#endif  // GTEST_HAS_GLOBAL_STRING

  static bool FullMatch(const char* str, const RE& re);
  static bool PartialMatch(const char* str, const RE& re);

 private:
  void Init(const char* regex);

  // We use a const char* instead of a string, as Google Test may be used
  // where string is not available.  We also do not use Google Test's own
  // String type here, in order to simplify dependencies between the
  // files.
  const char* pattern_;
  regex_t full_regex_;     // For FullMatch().
  regex_t partial_regex_;  // For PartialMatch().
  bool is_valid_;
};

#endif  // GTEST_HAS_DEATH_TEST

// Defines logging utilities:
//   GTEST_LOG_()   - logs messages at the specified severity level.
//   LogToStderr()  - directs all log messages to stderr.
//   FlushInfoLog() - flushes informational log messages.

enum GTestLogSeverity {
  GTEST_INFO,
  GTEST_WARNING,
  GTEST_ERROR,
  GTEST_FATAL
};

void GTestLog(GTestLogSeverity severity, const char* file,
              int line, const char* msg);

#define GTEST_LOG_(severity, msg)\
    ::testing::internal::GTestLog(\
        ::testing::internal::GTEST_##severity, __FILE__, __LINE__, \
        (::testing::Message() << (msg)).GetString().c_str())

inline void LogToStderr() {}
inline void FlushInfoLog() { fflush(NULL); }

// Defines the stderr capturer:
//   CaptureStderr     - starts capturing stderr.
//   GetCapturedStderr - stops capturing stderr and returns the captured string.

#ifdef GTEST_HAS_DEATH_TEST

// A copy of all command line arguments.  Set by InitGoogleTest().
extern ::std::vector<String> g_argvs;

void CaptureStderr();
// GTEST_HAS_DEATH_TEST implies we have ::std::string.
::std::string GetCapturedStderr();
const ::std::vector<String>& GetArgvs();

#endif  // GTEST_HAS_DEATH_TEST

// Defines synchronization primitives.

// A dummy implementation of synchronization primitives (mutex, lock,
// and thread-local variable).  Necessary for compiling Google Test where
// mutex is not supported - using Google Test in multiple threads is not
// supported on such platforms.

class Mutex {
 public:
  Mutex() {}
  explicit Mutex(int /*unused*/) {}
  void AssertHeld() const {}
  enum { NO_CONSTRUCTOR_NEEDED_FOR_STATIC_MUTEX = 0 };
};

// We cannot call it MutexLock directly as the ctor declaration would
// conflict with a macro named MutexLock, which is defined on some
// platforms.  Hence the typedef trick below.
class GTestMutexLock {
 public:
  explicit GTestMutexLock(Mutex*) {}  // NOLINT
};

typedef GTestMutexLock MutexLock;

template <typename T>
class ThreadLocal {
 public:
  ThreadLocal() : value_() {}
  explicit ThreadLocal(const T& value) : value_(value) {}
  T* pointer() { return &value_; }
  const T* pointer() const { return &value_; }
  const T& get() const { return value_; }
  void set(const T& value) { value_ = value; }
 private:
  T value_;
};

// There's no portable way to detect the number of threads, so we just
// return 0 to indicate that we cannot detect it.
inline size_t GetThreadCount() { return 0; }

// The above synchronization primitives have dummy implementations.
// Therefore Google Test is not thread-safe.
#define GTEST_IS_THREADSAFE 0

#if defined(__SYMBIAN32__) || defined(__IBMCPP__)

// Passing non-POD classes through ellipsis (...) crashes the ARM
// compiler.  The Nokia Symbian and the IBM XL C/C++ compiler try to
// instantiate a copy constructor for objects passed through ellipsis
// (...), failing for uncopyable objects.  We define this to indicate
// the fact.
#define GTEST_ELLIPSIS_NEEDS_COPY_ 1

// The Nokia Symbian and IBM XL C/C++ compilers cannot decide between
// const T& and const T* in a function template.  These compilers
// _can_ decide between class template specializations for T and T*,
// so a tr1::type_traits-like is_pointer works.
#define GTEST_NEEDS_IS_POINTER_ 1

#endif  // defined(__SYMBIAN32__) || defined(__IBMCPP__)

template <bool bool_value>
struct bool_constant {
  typedef bool_constant<bool_value> type;
  static const bool value = bool_value;
};
template <bool bool_value> const bool bool_constant<bool_value>::value;

typedef bool_constant<false> false_type;
typedef bool_constant<true> true_type;

template <typename T>
struct is_pointer : public false_type {};

template <typename T>
struct is_pointer<T*> : public true_type {};

// Defines BiggestInt as the biggest signed integer type the compiler
// supports.

#ifdef GTEST_OS_WINDOWS
typedef __int64 BiggestInt;
#else
typedef long long BiggestInt;  // NOLINT
#endif  // GTEST_OS_WINDOWS

// The maximum number a BiggestInt can represent.  This definition
// works no matter BiggestInt is represented in one's complement or
// two's complement.
//
// We cannot rely on numeric_limits in STL, as __int64 and long long
// are not part of standard C++ and numeric_limits doesn't need to be
// defined for them.
const BiggestInt kMaxBiggestInt =
    ~(static_cast<BiggestInt>(1) << (8*sizeof(BiggestInt) - 1));

// This template class serves as a compile-time function from size to
// type.  It maps a size in bytes to a primitive type with that
// size. e.g.
//
//   TypeWithSize<4>::UInt
//
// is typedef-ed to be unsigned int (unsigned integer made up of 4
// bytes).
//
// Such functionality should belong to STL, but I cannot find it
// there.
//
// Google Test uses this class in the implementation of floating-point
// comparison.
//
// For now it only handles UInt (unsigned int) as that's all Google Test
// needs.  Other types can be easily added in the future if need
// arises.
template <size_t size>
class TypeWithSize {
 public:
  // This prevents the user from using TypeWithSize<N> with incorrect
  // values of N.
  typedef void UInt;
};

// The specialization for size 4.
template <>
class TypeWithSize<4> {
 public:
  // unsigned int has size 4 in both gcc and MSVC.
  //
  // As base/basictypes.h doesn't compile on Windows, we cannot use
  // uint32, uint64, and etc here.
  typedef int Int;
  typedef unsigned int UInt;
};

// The specialization for size 8.
template <>
class TypeWithSize<8> {
 public:
#ifdef GTEST_OS_WINDOWS
  typedef __int64 Int;
  typedef unsigned __int64 UInt;
#else
  typedef long long Int;  // NOLINT
  typedef unsigned long long UInt;  // NOLINT
#endif  // GTEST_OS_WINDOWS
};

// Integer types of known sizes.
typedef TypeWithSize<4>::Int Int32;
typedef TypeWithSize<4>::UInt UInt32;
typedef TypeWithSize<8>::Int Int64;
typedef TypeWithSize<8>::UInt UInt64;
typedef TypeWithSize<8>::Int TimeInMillis;  // Represents time in milliseconds.

// Utilities for command line flags and environment variables.

// A wrapper for getenv() that works on Linux, Windows, and Mac OS.
inline const char* GetEnv(const char* name) {
#ifdef _WIN32_WCE  // We are on Windows CE.
  // CE has no environment variables.
  return NULL;
#elif defined(GTEST_OS_WINDOWS)  // We are on Windows proper.
  // MSVC 8 deprecates getenv(), so we want to suppress warning 4996
  // (deprecated function) there.
#pragma warning(push)          // Saves the current warning state.
#pragma warning(disable:4996)  // Temporarily disables warning 4996.
  return getenv(name);
#pragma warning(pop)           // Restores the warning state.
#else  // We are on Linux or Mac OS.
  return getenv(name);
#endif
}

#ifdef _WIN32_WCE
// Windows CE has no C library. The abort() function is used in
// several places in Google Test. This implementation provides a reasonable
// imitation of standard behaviour.
void abort();
#else
inline void abort() { ::abort(); }
#endif  // _WIN32_WCE

// INTERNAL IMPLEMENTATION - DO NOT USE.
//
// GTEST_CHECK_ is an all-mode assert. It aborts the program if the condition
// is not satisfied.
//  Synopsys:
//    GTEST_CHECK_(boolean_condition);
//     or
//    GTEST_CHECK_(boolean_condition) << "Additional message";
//
//    This checks the condition and if the condition is not satisfied
//    it prints message about the condition violation, including the
//    condition itself, plus additional message streamed into it, if any,
//    and then it aborts the program. It aborts the program irrespective of
//    whether it is built in the debug mode or not.
class GTestCheckProvider {
 public:
  GTestCheckProvider(const char* condition, const char* file, int line) {
    FormatFileLocation(file, line);
    ::std::cerr << " ERROR: Condition " << condition << " failed. ";
  }
  ~GTestCheckProvider() {
    ::std::cerr << ::std::endl;
    abort();
  }
  void FormatFileLocation(const char* file, int line) {
    if (file == NULL)
      file = "unknown file";
    if (line < 0) {
      ::std::cerr << file << ":";
    } else {
#if _MSC_VER
      ::std::cerr << file << "(" << line << "):";
#else
      ::std::cerr << file << ":" << line << ":";
#endif
    }
  }
  ::std::ostream& GetStream() { return ::std::cerr; }
};
#define GTEST_CHECK_(condition) \
    GTEST_AMBIGUOUS_ELSE_BLOCKER_ \
    if (condition) \
      ; \
    else \
      ::testing::internal::GTestCheckProvider(\
          #condition, __FILE__, __LINE__).GetStream()

// Macro for referencing flags.
#define GTEST_FLAG(name) FLAGS_gtest_##name

// Macros for declaring flags.
#define GTEST_DECLARE_bool_(name) extern bool GTEST_FLAG(name)
#define GTEST_DECLARE_int32_(name) \
    extern ::testing::internal::Int32 GTEST_FLAG(name)
#define GTEST_DECLARE_string_(name) \
    extern ::testing::internal::String GTEST_FLAG(name)

// Macros for defining flags.
#define GTEST_DEFINE_bool_(name, default_val, doc) \
    bool GTEST_FLAG(name) = (default_val)
#define GTEST_DEFINE_int32_(name, default_val, doc) \
    ::testing::internal::Int32 GTEST_FLAG(name) = (default_val)
#define GTEST_DEFINE_string_(name, default_val, doc) \
    ::testing::internal::String GTEST_FLAG(name) = (default_val)

// Parses 'str' for a 32-bit signed integer.  If successful, writes the result
// to *value and returns true; otherwise leaves *value unchanged and returns
// false.
// TODO(chandlerc): Find a better way to refactor flag and environment parsing
// out of both gtest-port.cc and gtest.cc to avoid exporting this utility
// function.
bool ParseInt32(const Message& src_text, const char* str, Int32* value);

// Parses a bool/Int32/string from the environment variable
// corresponding to the given Google Test flag.
bool BoolFromGTestEnv(const char* flag, bool default_val);
Int32 Int32FromGTestEnv(const char* flag, Int32 default_val);
const char* StringFromGTestEnv(const char* flag, const char* default_val);

}  // namespace internal
}  // namespace testing

#endif  // GTEST_INCLUDE_GTEST_INTERNAL_GTEST_PORT_H_

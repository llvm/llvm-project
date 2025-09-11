//===- llvm/Support/DebugLog.h - Logging like debug output ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file contains macros for logging like debug output. It builds upon the
// support in Debug.h but provides a utility function for common debug output
// style.
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_DEBUGLOG_H
#define LLVM_SUPPORT_DEBUGLOG_H

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
#ifndef NDEBUG

/// LDBG() is a macro that can be used as a raw_ostream for debugging.
/// It will stream the output to the dbgs() stream, with a prefix of the
/// debug type and the file and line number. A trailing newline is added to the
/// output automatically. If the streamed content contains a newline, the prefix
/// is added to each beginning of a new line. Nothing is printed if the debug
/// output is not enabled or the debug type does not match.
///
/// E.g.,
///   LDBG() << "Bitset contains: " << Bitset;
/// is equivalent to
///   LLVM_DEBUG(dbgs() << "[" << DEBUG_TYPE << "] " << __FILE__ << ":" <<
///   __LINE__ << " "
///              << "Bitset contains: " << Bitset << "\n");
///
// An optional `level` argument can be provided to control the verbosity of the
/// output. The default level is 1, and is in increasing level of verbosity.
///
/// The `level` argument can be a literal integer, or a macro that evaluates to
/// an integer.
///
/// An optional `type` argument can be provided to control the debug type. The
/// default type is DEBUG_TYPE. The `type` argument can be a literal string, or
/// a macro that evaluates to a string.
///
/// E.g.,
///   LDBG(2) << "Bitset contains: " << Bitset;
///   LDBG("debug_type") << "Bitset contains: " << Bitset;
///   LDBG("debug_type", 2) << "Bitset contains: " << Bitset;
#define LDBG(...) _GET_LDBG_MACRO(__VA_ARGS__)(__VA_ARGS__)

/// LDBG_OS() is a macro that behaves like LDBG() but instead of directly using
/// it to stream the output, it takes a callback function that will be called
/// with a raw_ostream.
/// This is useful when you need to pass a `raw_ostream` to a helper function to
/// be able to print (when the `<<` operator is not available).
///
/// E.g.,
///   LDBG_OS([&] (raw_ostream &Os) {
///     Os << "Pass Manager contains: ";
///     pm.printAsTextual(Os);
///   });
///
/// Just like LDBG(), it optionally accepts a `level` and `type` arguments.
/// E.g.,
///   LDBG_OS(2, [&] (raw_ostream &Os) { ... });
///   LDBG_OS("debug_type", [&] (raw_ostream &Os) { ... });
///   LDBG_OS("debug_type", 2, [&] (raw_ostream &Os) { ... });
///
#define LDBG_OS(...) _GET_LDBG_OS_MACRO(__VA_ARGS__)(__VA_ARGS__)

// We want the filename without the full path. We are using the __FILE__ macro
// and a constexpr function to strip the path prefix. We can avoid the frontend
// repeated evaluation of __FILE__ by using the __FILE_NAME__ when defined
// (gcc and clang do) which contains the file name already.
#if defined(__FILE_NAME__)
#define __LLVM_FILE_NAME__ __FILE_NAME__
#else
#define __LLVM_FILE_NAME__ ::llvm::impl::getShortFileName(__FILE__)
#endif

// Everything below are implementation details of the macros above.
namespace impl {

/// This macro expands to the stream to use for output, we use a macro to allow
/// unit-testing to override.
#define LDBG_STREAM ::llvm::dbgs()

// ----------------------------------------------------------------------------
// LDBG() implementation
// ----------------------------------------------------------------------------

// Helper macros to choose the correct LDBG() macro based on the number of
// arguments.
#define LDBG_FUNC_CHOOSER(_f1, _f2, _f3, ...) _f3
#define LDBG_FUNC_RECOMPOSER(argsWithParentheses)                              \
  LDBG_FUNC_CHOOSER argsWithParentheses
#define LDBG_CHOOSE_FROM_ARG_COUNT(...)                                        \
  LDBG_FUNC_RECOMPOSER((__VA_ARGS__, LDBG_TYPE_AND_LEVEL, LDBG_LEVEL_OR_TYPE, ))
#define LDBG_NO_ARG_EXPANDER() , , LDBG_NO_ARG
#define _GET_LDBG_MACRO(...)                                                   \
  LDBG_CHOOSE_FROM_ARG_COUNT(LDBG_NO_ARG_EXPANDER __VA_ARGS__())

/// This macro is the core of the LDBG() implementation. It is used to print the
/// debug output with the given stream, level, type, file, and line number.
#define LDBG_STREAM_LEVEL_TYPE_FILE_AND_LINE(STREAM, LEVEL_OR_TYPE,            \
                                             TYPE_OR_LEVEL, FILE, LINE)        \
  for (bool _c = ::llvm::DebugFlag && ::llvm::impl::ldbgIsCurrentDebugType(    \
                                          TYPE_OR_LEVEL, LEVEL_OR_TYPE);       \
       _c; _c = false)                                                         \
  ::llvm::impl::raw_ldbg_ostream{                                              \
      ::llvm::impl::computePrefix(TYPE_OR_LEVEL, FILE, LINE, LEVEL_OR_TYPE),   \
      (STREAM), /*ShouldPrefixNextString=*/true,                               \
      /*ShouldEmitNewLineOnDestruction=*/true}                                 \
      .asLvalue()

/// These macros are helpers to implement LDBG() with an increasing amount of
/// optional arguments made explicit.
#define LDBG_STREAM_LEVEL_TYPE_AND_FILE(STREAM, LEVEL_OR_TYPE, TYPE_OR_LEVEL,  \
                                        FILE)                                  \
  LDBG_STREAM_LEVEL_TYPE_FILE_AND_LINE(STREAM, LEVEL_OR_TYPE, TYPE_OR_LEVEL,   \
                                       FILE, __LINE__)
#define LDGB_STREAM_LEVEL_AND_TYPE(STREAM, LEVEL_OR_TYPE, TYPE_OR_LEVEL)       \
  LDBG_STREAM_LEVEL_TYPE_AND_FILE(STREAM, LEVEL_OR_TYPE, TYPE_OR_LEVEL,        \
                                  __LLVM_FILE_NAME__)
/// This macro is a helper when LDBG() is called with 2 arguments.
/// In this case we want to force the first argument to be the type for
/// consistency in the codebase.
/// We trick this by casting the first argument to a (const char *) which
/// won't compile with an int.
#define LDBG_TYPE_AND_LEVEL(TYPE, LEVEL)                                       \
  LDGB_STREAM_LEVEL_AND_TYPE(LDBG_STREAM, static_cast<const char *>(TYPE),     \
                             (LEVEL))

/// When a single argument is provided. This can be either a level or the debug
/// type. If a level is provided, we default the debug type to DEBUG_TYPE, if a
/// string is provided, we default the level to 1.
#define LDBG_LEVEL_OR_TYPE(LEVEL_OR_TYPE)                                      \
  LDGB_STREAM_LEVEL_AND_TYPE(LDBG_STREAM, (LEVEL_OR_TYPE),                     \
                             LDBG_GET_DEFAULT_TYPE_OR_LEVEL(LEVEL_OR_TYPE))
#define LDBG_NO_ARG() LDBG_LEVEL_OR_TYPE(1)

// ----------------------------------------------------------------------------
// LDBG_OS() implementation
// ----------------------------------------------------------------------------

// Helper macros to choose the correct LDBG_OS() macro based on the number of
// arguments.
#define LDBG_OS_FUNC_CHOOSER(_f1, _f2, _f3, _f4, ...) _f4
#define LDBG_OS_FUNC_RECOMPOSER(argsWithParentheses)                           \
  LDBG_OS_FUNC_CHOOSER argsWithParentheses
#define LDBG_OS_CHOOSE_FROM_ARG_COUNT(...)                                     \
  LDBG_OS_FUNC_RECOMPOSER((__VA_ARGS__, LDBG_OS_TYPE_AND_LEVEL_AND_CALLBACK,   \
                           LDBG_OS_LEVEL_OR_TYPE_AND_CALLBACK,                 \
                           LDBG_OS_CALLBACK, ))
#define LDBG_OS_NO_ARG_EXPANDER() , , , LDBG_OS_CALLBACK
#define _GET_LDBG_OS_MACRO(...)                                                \
  LDBG_OS_CHOOSE_FROM_ARG_COUNT(LDBG_OS_NO_ARG_EXPANDER __VA_ARGS__())

/// This macro is the core of the LDBG_OS() macros. It is used to print the
/// debug output with the given stream, level, type, file, and line number.
#define LDBG_OS_IMPL(TYPE_OR_LEVEL, LEVEL_OR_TYPE, CALLBACK, STREAM, FILE,     \
                     LINE)                                                     \
  if (::llvm::DebugFlag &&                                                     \
      ::llvm::impl::ldbgIsCurrentDebugType(TYPE_OR_LEVEL, LEVEL_OR_TYPE)) {    \
    ::llvm::impl::raw_ldbg_ostream LdbgOS{                                     \
        ::llvm::impl::computePrefix(TYPE_OR_LEVEL, FILE, LINE, LEVEL_OR_TYPE), \
        (STREAM), /*ShouldPrefixNextString=*/true,                             \
        /*ShouldEmitNewLineOnDestruction=*/true};                              \
    CALLBACK(LdbgOS);                                                          \
  }

#define LDBG_OS_TYPE_AND_LEVEL_AND_CALLBACK(TYPE, LEVEL, CALLBACK)             \
  LDBG_OS_IMPL(static_cast<const char *>(TYPE), LEVEL, CALLBACK, LDBG_STREAM,  \
               __LLVM_FILE_NAME__, __LINE__)
#define LDBG_OS_LEVEL_OR_TYPE_AND_CALLBACK(LEVEL_OR_TYPE, CALLBACK)            \
  LDBG_OS_IMPL(LDBG_GET_DEFAULT_TYPE_OR_LEVEL(LEVEL_OR_TYPE), LEVEL_OR_TYPE,   \
               CALLBACK, LDBG_STREAM, __LLVM_FILE_NAME__, __LINE__)
#define LDBG_OS_CALLBACK(CALLBACK)                                             \
  LDBG_OS_LEVEL_OR_TYPE_AND_CALLBACK(1, CALLBACK)

// ----------------------------------------------------------------------------
// General Helpers for the implementation above
// ----------------------------------------------------------------------------

/// Return the stringified macro as a StringRef.
/// Also, strip out potential surrounding quotes: this comes from an artifact of
/// the macro stringification, if DEBUG_TYPE is undefined we get the string
/// "DEBUG_TYPE", however if it is defined we get the string with the quotes.
/// For example if DEBUG_TYPE is "foo", we get "\"foo\"" but we want to return
/// "foo" here.
constexpr ::llvm::StringRef strip_quotes(const char *Str) {
  ::llvm::StringRef S(Str);
  if (Str[0] == '"' && Str[S.size() - 1] == '"')
    return StringRef(Str + 1, S.size() - 2);
  return S;
}

/// Helper to provide the default level (=1) or type (=DEBUG_TYPE). This is used
/// when a single argument is passed to LDBG() (or LDBG_OS()), if it is an
/// integer we return DEBUG_TYPE and if it is a string we return 1. This fails
/// with a static_assert if we pass an integer and DEBUG_TYPE is not defined.
#define LDBG_GET_DEFAULT_TYPE_OR_LEVEL(LEVEL_OR_TYPE)                          \
  [](auto LevelOrType) {                                                       \
    if constexpr (std::is_integral_v<decltype(LevelOrType)>) {                 \
      constexpr const char *DebugType = LDBG_GET_DEBUG_TYPE_STR();             \
      if constexpr (DebugType[0] == '"')                                       \
        return ::llvm::impl::strip_quotes(DebugType);                          \
      else                                                                     \
        static_assert(false, "DEBUG_TYPE is not defined");                     \
    } else {                                                                   \
      return 1;                                                                \
    }                                                                          \
  }(LEVEL_OR_TYPE)

/// Helpers to get DEBUG_TYPE as a StringRef, even when DEBUG_TYPE is not
/// defined (in which case it expands to "DEBUG_TYPE")
#define LDBG_GET_DEBUG_TYPE_STR__(X) #X
#define LDBG_GET_DEBUG_TYPE_STR_(X) LDBG_GET_DEBUG_TYPE_STR__(X)
#define LDBG_GET_DEBUG_TYPE_STR() LDBG_GET_DEBUG_TYPE_STR_(DEBUG_TYPE)

/// Helper to call isCurrentDebugType with a StringRef.
static LLVM_ATTRIBUTE_UNUSED bool ldbgIsCurrentDebugType(StringRef Type,
                                                         int Level) {
  return ::llvm::isCurrentDebugType(Type.str().c_str(), Level);
}
static LLVM_ATTRIBUTE_UNUSED bool ldbgIsCurrentDebugType(int Level,
                                                         StringRef Type) {
  return ::llvm::isCurrentDebugType(Type.str().c_str(), Level);
}

/// A raw_ostream that tracks `\n` and print the prefix after each
/// newline.
class LLVM_ABI raw_ldbg_ostream final : public raw_ostream {
  std::string Prefix;
  raw_ostream &Os;
  bool ShouldPrefixNextString;
  bool ShouldEmitNewLineOnDestruction;

  /// Split the line on newlines and insert the prefix before each
  /// newline. Forward everything to the underlying stream.
  void write_impl(const char *Ptr, size_t Size) final {
    auto Str = StringRef(Ptr, Size);
    auto Eol = Str.find('\n');
    // Handle `\n` occurring in the string, ensure to print the prefix at the
    // beginning of each line.
    while (Eol != StringRef::npos) {
      // Take the line up to the newline (including the newline).
      StringRef Line = Str.take_front(Eol + 1);
      if (!Line.empty())
        writeWithPrefix(Line);
      // We printed a newline, record here to print a prefix.
      ShouldPrefixNextString = true;
      Str = Str.drop_front(Eol + 1);
      Eol = Str.find('\n');
    }
    if (!Str.empty())
      writeWithPrefix(Str);
  }
  void emitPrefix() { Os.write(Prefix.c_str(), Prefix.size()); }
  void writeWithPrefix(StringRef Str) {
    if (ShouldPrefixNextString) {
      emitPrefix();
      ShouldPrefixNextString = false;
    }
    Os.write(Str.data(), Str.size());
  }

public:
  explicit raw_ldbg_ostream(std::string Prefix, raw_ostream &Os,
                            bool ShouldPrefixNextString = true,
                            bool ShouldEmitNewLineOnDestruction = false)
      : Prefix(std::move(Prefix)), Os(Os),
        ShouldPrefixNextString(ShouldPrefixNextString),
        ShouldEmitNewLineOnDestruction(ShouldEmitNewLineOnDestruction) {
    SetUnbuffered();
  }
  ~raw_ldbg_ostream() final {
    if (ShouldEmitNewLineOnDestruction)
      Os << '\n';
  }

  /// Forward the current_pos method to the underlying stream.
  uint64_t current_pos() const final { return Os.tell(); }

  /// Some of the `<<` operators expect an lvalue, so we trick the type
  /// system.
  raw_ldbg_ostream &asLvalue() { return *this; }
};

/// A raw_ostream that prints a newline on destruction, useful for LDBG()
class RAIINewLineStream final : public raw_ostream {
  raw_ostream &Os;

public:
  RAIINewLineStream(raw_ostream &Os) : Os(Os) { SetUnbuffered(); }
  ~RAIINewLineStream() { Os << '\n'; }
  void write_impl(const char *Ptr, size_t Size) final { Os.write(Ptr, Size); }
  uint64_t current_pos() const final { return Os.tell(); }
  RAIINewLineStream &asLvalue() { return *this; }
};

/// Remove the path prefix from the file name.
static LLVM_ATTRIBUTE_UNUSED constexpr const char *
getShortFileName(const char *path) {
  const char *filename = path;
  for (const char *p = path; *p != '\0'; ++p) {
    if (*p == '/' || *p == '\\')
      filename = p + 1;
  }
  return filename;
}

/// Compute the prefix for the debug log in the form of:
/// "[DebugType] File:Line "
/// Where the File is the file name without the path prefix.
static LLVM_ATTRIBUTE_UNUSED std::string
computePrefix(StringRef DebugType, const char *File, int Line, int Level) {
  std::string Prefix;
  raw_string_ostream OsPrefix(Prefix);
  if (!DebugType.empty())
    OsPrefix << "[" << DebugType << ":" << Level << "] ";
  OsPrefix << File << ":" << Line << " ";
  return OsPrefix.str();
}
/// Overload allowing to swap the order of the DebugType and Level arguments.
static LLVM_ATTRIBUTE_UNUSED std::string
computePrefix(int Level, const char *File, int Line, StringRef DebugType) {
  return computePrefix(DebugType, File, Line, Level);
}

} // end namespace impl
#else
// As others in Debug, When compiling without assertions, the -debug-* options
// and all inputs too LDBG() are ignored.
#define LDBG(...)                                                              \
  for (bool _c = false; _c; _c = false)                                        \
  ::llvm::nulls()
#define LDBG_OS(...)
#endif
} // end namespace llvm

#endif // LLVM_SUPPORT_DEBUGLOG_H

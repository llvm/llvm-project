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

// LDBG() is a macro that can be used as a raw_ostream for debugging.
// It will stream the output to the dbgs() stream, with a prefix of the
// debug type and the file and line number. A trailing newline is added to the
// output automatically. If the streamed content contains a newline, the prefix
// is added to each beginning of a new line. Nothing is printed if the debug
// output is not enabled or the debug type does not match.
//
// E.g.,
//   LDBG() << "Bitset contains: " << Bitset;
// is somehow equivalent to
//   LLVM_DEBUG(dbgs() << "[" << DEBUG_TYPE << "] " << __FILE__ << ":" <<
//   __LINE__ << " "
//              << "Bitset contains: " << Bitset << "\n");
//
// An optional `level` argument can be provided to control the verbosity of the
// output. The default level is 1, and is in increasing level of verbosity.
//
// The `level` argument can be a literal integer, or a macro that evaluates to
// an integer.
//
#define LDBG(...) _GET_LDBG_MACRO(__VA_ARGS__)(__VA_ARGS__)

// Helper macros to choose the correct macro based on the number of arguments.
#define LDBG_FUNC_CHOOSER(_f1, _f2, ...) _f2
#define LDBG_FUNC_RECOMPOSER(argsWithParentheses)                              \
  LDBG_FUNC_CHOOSER argsWithParentheses
#define LDBG_CHOOSE_FROM_ARG_COUNT(...)                                        \
  LDBG_FUNC_RECOMPOSER((__VA_ARGS__, LDBG_LOG_LEVEL, ))
#define LDBG_NO_ARG_EXPANDER() , LDBG_LOG_LEVEL_1
#define _GET_LDBG_MACRO(...)                                                   \
  LDBG_CHOOSE_FROM_ARG_COUNT(LDBG_NO_ARG_EXPANDER __VA_ARGS__())

// Dispatch macros to support the `level` argument or none (default to 1)
#define LDBG_LOG_LEVEL(LEVEL)                                                  \
  DEBUGLOG_WITH_STREAM_AND_TYPE(llvm::dbgs(), LEVEL, DEBUG_TYPE)
#define LDBG_LOG_LEVEL_1() LDBG_LOG_LEVEL(1)

// We want the filename without the full path. We are using the __FILE__ macro
// and a constexpr function to strip the path prefix. We can avoid the frontend
// repeated evaluation of __FILE__ by using the __FILE_NAME__ when defined
// (gcc and clang do) which contains the file name already.
#if defined(__FILE_NAME__)
#define __LLVM_FILE_NAME__ __FILE_NAME__
#else
#define __LLVM_FILE_NAME__ ::llvm::impl::getShortFileName(__FILE__)
#endif

#define DEBUGLOG_WITH_STREAM_TYPE_FILE_AND_LINE(STREAM, LEVEL, TYPE, FILE,     \
                                                LINE)                          \
  for (bool _c =                                                               \
           (::llvm::DebugFlag && ::llvm::isCurrentDebugType(TYPE, LEVEL));     \
       _c; _c = false)                                                         \
    for (::llvm::impl::RAIINewLineStream NewLineStream{(STREAM)}; _c;          \
         _c = false)                                                           \
  ::llvm::impl::raw_ldbg_ostream{                                              \
      ::llvm::impl::computePrefix(TYPE, FILE, LINE, LEVEL), NewLineStream}     \
      .asLvalue()

#define DEBUGLOG_WITH_STREAM_TYPE_AND_FILE(STREAM, LEVEL, TYPE, FILE)          \
  DEBUGLOG_WITH_STREAM_TYPE_FILE_AND_LINE(STREAM, LEVEL, TYPE, FILE, __LINE__)
#define DEBUGLOG_WITH_STREAM_AND_TYPE(STREAM, LEVEL, TYPE)                     \
  DEBUGLOG_WITH_STREAM_TYPE_AND_FILE(STREAM, LEVEL, TYPE, __LLVM_FILE_NAME__)

namespace impl {

/// A raw_ostream that tracks `\n` and print the prefix after each
/// newline.
class LLVM_ABI raw_ldbg_ostream final : public raw_ostream {
  std::string Prefix;
  raw_ostream &Os;
  bool HasPendingNewline;

  /// Split the line on newlines and insert the prefix before each
  /// newline. Forward everything to the underlying stream.
  void write_impl(const char *Ptr, size_t Size) final {
    auto Str = StringRef(Ptr, Size);
    // Handle the initial prefix.
    if (!Str.empty())
      writeWithPrefix(StringRef());

    auto Eol = Str.find('\n');
    while (Eol != StringRef::npos) {
      StringRef Line = Str.take_front(Eol + 1);
      if (!Line.empty())
        writeWithPrefix(Line);
      HasPendingNewline = true;
      Str = Str.drop_front(Eol + 1);
      Eol = Str.find('\n');
    }
    if (!Str.empty())
      writeWithPrefix(Str);
  }
  void emitPrefix() { Os.write(Prefix.c_str(), Prefix.size()); }
  void writeWithPrefix(StringRef Str) {
    flushEol();
    Os.write(Str.data(), Str.size());
  }

public:
  explicit raw_ldbg_ostream(std::string Prefix, raw_ostream &Os,
                            bool HasPendingNewline = true)
      : Prefix(std::move(Prefix)), Os(Os),
        HasPendingNewline(HasPendingNewline) {
    SetUnbuffered();
  }
  ~raw_ldbg_ostream() final { flushEol(); }
  void flushEol() {
    if (HasPendingNewline) {
      emitPrefix();
      HasPendingNewline = false;
    }
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
computePrefix(const char *DebugType, const char *File, int Line, int Level) {
  std::string Prefix;
  raw_string_ostream OsPrefix(Prefix);
  if (DebugType)
    OsPrefix << "[" << DebugType << ":" << Level << "] ";
  OsPrefix << File << ":" << Line << " ";
  return OsPrefix.str();
}
} // end namespace impl
#else
// As others in Debug, When compiling without assertions, the -debug-* options
// and all inputs too LDBG() are ignored.
#define LDBG(...)                                                              \
  for (bool _c = false; _c; _c = false)                                        \
  ::llvm::nulls()
#endif
} // end namespace llvm

#endif // LLVM_SUPPORT_DEBUGLOG_H

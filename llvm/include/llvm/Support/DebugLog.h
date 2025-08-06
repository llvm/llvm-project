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

#define DEBUGLOG_WITH_STREAM_AND_TYPE(STREAM, TYPE)                            \
  for (bool _c = (::llvm::DebugFlag && ::llvm::isCurrentDebugType(TYPE)); _c;  \
       _c = false)                                                             \
  ::llvm::impl::LogWithNewline(TYPE, __FILE__, __LINE__, (STREAM))

namespace impl {
class LogWithNewline {
public:
  LogWithNewline(const char *debug_type, const char *file, int line,
                 raw_ostream &os)
      : os(os) {
    if (debug_type)
      os << "[" << debug_type << "] ";
    os << file << ":" << line << " ";
  }
  ~LogWithNewline() { os << '\n'; }
  template <typename T> raw_ostream &operator<<(const T &t) && {
    return os << t;
  }

  // Prevent copying, as this class manages newline responsibility and is
  // intended for use as a temporary.
  LogWithNewline(const LogWithNewline &) = delete;
  LogWithNewline &operator=(const LogWithNewline &) = delete;
  LogWithNewline &operator=(LogWithNewline &&) = delete;

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

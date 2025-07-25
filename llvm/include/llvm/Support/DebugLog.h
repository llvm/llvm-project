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

// Output with given inputs and trailing newline. E.g.,
//   LDBG() << "Bitset contains: " << Bitset;
// is equivalent to
//   LLVM_DEBUG(dbgs() << DEBUG_TYPE << " [" << __FILE__ << ":" << __LINE__
//              << "] " << "Bitset contains: " << Bitset << "\n");
#define LDBG() DEBUGLOG_WITH_STREAM_AND_TYPE(llvm::dbgs(), DEBUG_TYPE)

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

private:
  raw_ostream &os;
};
} // end namespace impl
#else
// As others in Debug, When compiling without assertions, the -debug-* options
// and all inputs too LDBG() are ignored.
#define LDBG()                                                                 \
  for (bool _c = false; _c; _c = false)                                        \
  ::llvm::nulls()
#endif
} // end namespace llvm

#endif // LLVM_SUPPORT_DEBUGLOG_H

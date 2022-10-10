//===-------- Debug.h ---- Debug utilities ------------------------ C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_DEVICERTL_DEBUG_H
#define OMPTARGET_DEVICERTL_DEBUG_H

#include "Configuration.h"
#include "LibC.h"

/// Assertion
///
/// {
extern "C" {
void __assert_assume(bool condition);
void __assert_fail(const char *assertion, const char *file, unsigned line,
                   const char *function);
}

#define ASSERT(expr)                                                           \
  {                                                                            \
    if (config::isDebugMode(config::DebugKind::Assertion) && !(expr))          \
      __assert_fail(#expr, __FILE__, __LINE__, __PRETTY_FUNCTION__);           \
    else                                                                       \
      __assert_assume(expr);                                                   \
  }

///}

#define PRINTF(fmt, ...) (void)printf(fmt, ##__VA_ARGS__);
#define PRINT(str) PRINTF("%s", str)

///}

/// Enter a debugging scope for performing function traces. Enabled with
/// FunctionTracting set in the debug kind.
#define FunctionTracingRAII()                                                  \
  DebugEntryRAII Entry(__FILE__, __LINE__, __PRETTY_FUNCTION__);

/// An RAII class for handling entries to debug locations. The current location
/// and function will be printed on entry. Nested levels increase the
/// indentation shown in the debugging output.
struct DebugEntryRAII {
  DebugEntryRAII(const char *File, const unsigned Line, const char *Function);
  ~DebugEntryRAII();

  static void init();
};

#endif

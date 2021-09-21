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

/// Assertion
///
/// {
extern "C" {
void __assert_assume(bool cond, const char *exp, const char *file, int line);
void __assert_fail(const char *assertion, const char *file, unsigned line,
                   const char *function);
}

#define ASSERT(e) __assert_assume(e, #e, __FILE__, __LINE__)

///}

// TODO: We need to allow actual printf.
#define PRINTF(fmt, ...) (void)fmt;
#define PRINT(str) PRINTF("%s", str)

///}

/// Enter a debugging scope for performing function traces. Enabled with
/// FunctionTracting set in the debug kind.
#define FunctionTracingRAII()                                                  \
  DebugEntryRAII Entry(__LINE__, __PRETTY_FUNCTION__);

/// An RAII class for handling entries to debug locations. The current location
/// and function will be printed on entry. Nested levels increase the
/// indentation shown in the debugging output.
struct DebugEntryRAII {
  DebugEntryRAII(const unsigned Line, const char *Function);
  ~DebugEntryRAII();
};

#endif

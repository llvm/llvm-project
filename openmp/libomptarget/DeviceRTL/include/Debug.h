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

/// Print
/// TODO: For now we have to use macros to guard the code because Clang lowers
/// `printf` to different function calls on NVPTX and AMDGCN platforms, and it
/// doesn't work for AMDGCN. After it can work on AMDGCN, we will remove the
/// macro.
/// {

#ifndef __AMDGCN__
extern "C" {
int printf(const char *format, ...);
}

#define PRINTF(fmt, ...) (void)printf(fmt, __VA_ARGS__);
#define PRINT(str) PRINTF("%s", str)
#else
#define PRINTF(fmt, ...)
#define PRINT(str)
#endif

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

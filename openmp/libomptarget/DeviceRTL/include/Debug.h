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
}

#define ASSERT(e) __assert_assume(e, #e, __FILE__, __LINE__)

///}

// TODO: We need to allow actual printf.
#define PRINTF(fmt, ...) (void)fmt;
#define PRINT(str) PRINTF("%s", str)

#endif

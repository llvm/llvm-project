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
void __assert_fail(const char *expr, const char *file, unsigned line,
                   const char *function);
void __assert_fail_internal(const char *expr, const char *msg, const char *file,
                            unsigned line, const char *function);
}

#define ASSERT(expr, msg)                                                      \
  {                                                                            \
    if (config::isDebugMode(DeviceDebugKind::Assertion) && !(expr))            \
      __assert_fail_internal(#expr, msg, __FILE__, __LINE__,                   \
                             __PRETTY_FUNCTION__);                             \
    else                                                                       \
      __assert_assume(expr);                                                   \
  }
#define UNREACHABLE(msg)                                                       \
  printf(msg);                                                                 \
  __builtin_trap();                                                            \
  __builtin_unreachable();

///}

#endif

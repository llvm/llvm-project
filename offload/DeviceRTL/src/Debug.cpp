//===--- Debug.cpp -------- Debug utilities ----------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains debug utilities
//
//===----------------------------------------------------------------------===//

#include "Shared/Environment.h"

#include "Configuration.h"
#include "Debug.h"
#include "DeviceTypes.h"
#include "Interface.h"
#include "Mapping.h"
#include "State.h"

using namespace ompx;

#pragma omp begin declare target device_type(nohost)

extern "C" {
void __assert_assume(bool condition) { __builtin_assume(condition); }

#ifndef OMPTARGET_HAS_LIBC
[[gnu::weak]] void __assert_fail(const char *expr, const char *file,
                                 unsigned line, const char *function) {
  __assert_fail_internal(expr, nullptr, file, line, function);
}
#endif

void __assert_fail_internal(const char *expr, const char *msg, const char *file,
                            unsigned line, const char *function) {
  if (msg) {
    PRINTF("%s:%u: %s: Assertion %s (`%s`) failed.\n", file, line, function,
           msg, expr);
  } else {
    PRINTF("%s:%u: %s: Assertion `%s` failed.\n", file, line, function, expr);
  }
  __builtin_trap();
}
}

#pragma omp end declare target

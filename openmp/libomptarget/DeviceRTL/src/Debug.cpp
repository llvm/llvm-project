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

#include "Debug.h"
#include "Configuration.h"

using namespace _OMP;

#pragma omp declare target

extern "C" {
void __assert_assume(bool cond, const char *exp, const char *file, int line) {
  if (!cond && config::isDebugMode(config::DebugLevel::Assertion)) {
    PRINTF("ASSERTION failed: %s at %s, line %d\n", exp, file, line);
    __builtin_trap();
  }

  __builtin_assume(cond);
}

void __assert_fail(const char *assertion, const char *file, unsigned line,
                   const char *function) {
  PRINTF("%s:%u: %s: Assertion `%s' failed.\n", file, line, function,
         assertion);
  __builtin_trap();
}
}

#pragma omp end declare target

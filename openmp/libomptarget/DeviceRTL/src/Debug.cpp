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
#include "Environment.h"
#include "Interface.h"
#include "Mapping.h"
#include "State.h"
#include "Types.h"

using namespace ompx;

#pragma omp begin declare target device_type(nohost)

extern "C" {

/// AMDGCN Implementation
///
///{

namespace impl {
void omp_assert_assume(bool condition) {}
}

#pragma omp begin declare variant match(device = {arch(amdgcn)})
namespace impl {
void omp_assert_assume(bool condition) {}
}
#pragma omp end declare variant
///}

/// NVPTX Implementation
///
///{

#pragma omp begin declare variant match(                                       \
    device = {arch(nvptx, nvptx64)}, implementation = {extension(match_any)})
namespace impl {
void omp_assert_assume(bool condition) { __builtin_assume(condition); }
}
#pragma omp end declare variant
///}

void __assert_assume(bool condition) { impl::omp_assert_assume(condition); }

void __assert_fail(const char *expr, const char *msg, const char *file,
                   unsigned line, const char *function) {
  if (msg) {
    PRINTF("%s:%u: %s: Assertion %s (`%s') failed.\n", file, line, function,
           msg, expr);
  } else {
    PRINTF("%s:%u: %s: Assertion `%s' failed.\n", file, line, function, expr);
  }
  __builtin_trap();
}
}

#pragma omp end declare target

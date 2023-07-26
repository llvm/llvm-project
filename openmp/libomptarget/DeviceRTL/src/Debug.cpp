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
void __assert_assume(bool condition) { __builtin_assume(condition); }

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

DebugEntryRAII::DebugEntryRAII(const char *File, const unsigned Line,
                               const char *Function) {
  if (config::isDebugMode(config::DebugKind::FunctionTracing) &&
      mapping::getThreadIdInBlock() == 0 &&
      mapping::getBlockIdInKernel() == 0) {

    uint16_t &Level =
        state::getKernelEnvironment().DynamicEnv->DebugIndentionLevel;

    for (int I = 0; I < Level; ++I)
      PRINTF("%s", "  ");

    PRINTF("%s:%u: Thread %u Entering %s\n", File, Line,
           mapping::getThreadIdInBlock(), Function);
    Level++;
  }
}

DebugEntryRAII::~DebugEntryRAII() {
  if (config::isDebugMode(config::DebugKind::FunctionTracing) &&
      mapping::getThreadIdInBlock() == 0 &&
      mapping::getBlockIdInKernel() == 0) {
    uint16_t &Level =
        state::getKernelEnvironment().DynamicEnv->DebugIndentionLevel;
    Level--;
  }
}

#pragma omp end declare target

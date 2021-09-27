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
#include "Mapping.h"
#include "Types.h"

using namespace _OMP;

#pragma omp declare target

extern "C" {
void __assert_assume(bool cond, const char *exp, const char *file, int line) {
  if (!cond && config::isDebugMode(config::DebugKind::Assertion)) {
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

/// Current indentation level for the function trace. Only accessed by thread 0.
static uint32_t Level = 0;
#pragma omp allocate(Level) allocator(omp_pteam_mem_alloc)

DebugEntryRAII::DebugEntryRAII(const unsigned Line, const char *Function) {
  if (config::isDebugMode(config::DebugKind::FunctionTracing) &&
      mapping::getThreadIdInBlock() == 0) {

    for (int I = 0; I < Level; ++I)
      PRINTF("%s", "  ");

    PRINTF("Line %u: Thread %u Entering %s:%u\n", Line,
           mapping::getThreadIdInBlock(), Function);
    Level++;
  }
}

DebugEntryRAII::~DebugEntryRAII() {
  if (config::isDebugMode(config::DebugKind::FunctionTracing) &&
      mapping::getThreadIdInBlock() == 0)
    Level--;
}

#pragma omp end declare target

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
#include "Interface.h"
#include "Mapping.h"
#include "Types.h"

using namespace _OMP;

#pragma omp begin declare target device_type(nohost)

extern "C" {
void __assert_assume(bool condition) { __builtin_assume(condition); }

void __assert_fail(const char *assertion, const char *file, unsigned line,
                   const char *function) {
  PRINTF("%s:%u: %s: Assertion `%s' failed.\n", file, line, function,
         assertion);
  __builtin_trap();
}

namespace impl {
int32_t omp_vprintf(const char *Format, void *Arguments, uint32_t);
}

#pragma omp begin declare variant match(                                       \
    device = {arch(nvptx, nvptx64)}, implementation = {extension(match_any)})
int32_t vprintf(const char *, void *);
namespace impl {
static int32_t omp_vprintf(const char *Format, void *Arguments, uint32_t) {
  return vprintf(Format, Arguments);
}
} // namespace impl
#pragma omp end declare variant

// We do not have a vprintf implementation for AMD GPU yet so we use a stub.
#pragma omp begin declare variant match(device = {arch(amdgcn)})
namespace impl {
static int32_t omp_vprintf(const char *Format, void *Arguments, uint32_t) {
  return -1;
}
} // namespace impl
#pragma omp end declare variant

int32_t __llvm_omp_vprintf(const char *Format, void *Arguments, uint32_t Size) {
  return impl::omp_vprintf(Format, Arguments, Size);
}
}

/// Current indentation level for the function trace. Only accessed by thread 0.
__attribute__((loader_uninitialized)) static uint32_t Level;
#pragma omp allocate(Level) allocator(omp_pteam_mem_alloc)

DebugEntryRAII::DebugEntryRAII(const char *File, const unsigned Line,
                               const char *Function) {
  if (config::isDebugMode(config::DebugKind::FunctionTracing) &&
      mapping::getThreadIdInBlock() == 0 && mapping::getBlockId() == 0) {

    for (int I = 0; I < Level; ++I)
      PRINTF("%s", "  ");

    PRINTF("%s:%u: Thread %u Entering %s\n", File, Line,
           mapping::getThreadIdInBlock(), Function);
    Level++;
  }
}

DebugEntryRAII::~DebugEntryRAII() {
  if (config::isDebugMode(config::DebugKind::FunctionTracing) &&
      mapping::getThreadIdInBlock() == 0 && mapping::getBlockId() == 0)
    Level--;
}

void DebugEntryRAII::init() { Level = 0; }

#pragma omp end declare target

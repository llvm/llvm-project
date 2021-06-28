//===- CRunnerUtils.cpp - Utils for MLIR execution ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements basic functions to manipulate structured MLIR types at
// runtime. Entities in this file are meant to be retargetable, including on
// targets without a C++ runtime, and must be kept C compatible.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/CRunnerUtils.h"

#ifndef _WIN32
#include <sys/time.h>
#endif // _WIN32

#include <cinttypes>
#include <cstdio>

#ifdef MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS

// Small runtime support "lib" for vector.print lowering.
// By providing elementary printing methods only, this
// library can remain fully unaware of low-level implementation
// details of our vectors. Also useful for direct LLVM IR output.
extern "C" void printI64(int64_t i) { fprintf(stdout, "%" PRId64, i); }
extern "C" void printU64(uint64_t u) { fprintf(stdout, "%" PRIu64, u); }
extern "C" void printF32(float f) { fprintf(stdout, "%g", f); }
extern "C" void printF64(double d) { fprintf(stdout, "%lg", d); }
extern "C" void printOpen() { fputs("( ", stdout); }
extern "C" void printClose() { fputs(" )", stdout); }
extern "C" void printComma() { fputs(", ", stdout); }
extern "C" void printNewline() { fputc('\n', stdout); }

/// Prints GFLOPS rating.
extern "C" void print_flops(double flops) {
  fprintf(stderr, "%lf GFLOPS\n", flops / 1.0E9);
}

/// Returns the number of seconds since Epoch 1970-01-01 00:00:00 +0000 (UTC).
extern "C" double rtclock() {
#ifndef _WIN32
  struct timeval tp;
  int stat = gettimeofday(&tp, NULL);
  if (stat != 0)
    fprintf(stderr, "Error returning time from gettimeofday: %d\n", stat);
  return (tp.tv_sec + tp.tv_usec * 1.0e-6);
#else
  fprintf(stderr, "Timing utility not implemented on Windows\n");
  return 0.0;
#endif // _WIN32
}

#endif // MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS

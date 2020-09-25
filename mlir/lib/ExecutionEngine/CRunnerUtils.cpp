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

#include <cinttypes>
#include <cstdio>

#ifdef MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS

// Small runtime support "lib" for vector.print lowering.
// By providing elementary printing methods only, this
// library can remain fully unaware of low-level implementation
// details of our vectors. Also useful for direct LLVM IR output.
extern "C" void print_i32(int32_t i) { fprintf(stdout, "%" PRId32, i); }
extern "C" void print_i64(int64_t l) { fprintf(stdout, "%" PRId64, l); }
extern "C" void printU32(uint32_t i) { fprintf(stdout, "%" PRIu32, i); }
extern "C" void printU64(uint64_t l) { fprintf(stdout, "%" PRIu64, l); }
extern "C" void print_f32(float f) { fprintf(stdout, "%g", f); }
extern "C" void print_f64(double d) { fprintf(stdout, "%lg", d); }
extern "C" void print_open() { fputs("( ", stdout); }
extern "C" void print_close() { fputs(" )", stdout); }
extern "C" void print_comma() { fputs(", ", stdout); }
extern "C" void print_newline() { fputc('\n', stdout); }

#endif

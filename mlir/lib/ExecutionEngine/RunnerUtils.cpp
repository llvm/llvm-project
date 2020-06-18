//===- RunnerUtils.cpp - Utils for MLIR exec on targets with a C++ runtime ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements basic functions to debug structured MLIR types at
// runtime. Entities in this file may not be compatible with targets without a
// C++ runtime. These may be progressively migrated to CRunnerUtils.cpp over
// time.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/RunnerUtils.h"

#ifndef _WIN32
#include <sys/time.h>
#endif // _WIN32

extern "C" void _mlir_ciface_print_memref_vector_4x4xf32(
    StridedMemRefType<Vector2D<4, 4, float>, 2> *M) {
  impl::printMemRef(*M);
}

extern "C" void _mlir_ciface_print_memref_i8(UnrankedMemRefType<int8_t> *M) {
  impl::printMemRef(*M);
}

extern "C" void _mlir_ciface_print_memref_i32(UnrankedMemRefType<int32_t> *M) {
  impl::printMemRef(*M);
}

extern "C" void _mlir_ciface_print_memref_f32(UnrankedMemRefType<float> *M) {
  impl::printMemRef(*M);
}

extern "C" void print_memref_i32(int64_t rank, void *ptr) {
  UnrankedMemRefType<int32_t> descriptor = {rank, ptr};
  _mlir_ciface_print_memref_i32(&descriptor);
}

extern "C" void print_memref_f32(int64_t rank, void *ptr) {
  UnrankedMemRefType<float> descriptor = {rank, ptr};
  _mlir_ciface_print_memref_f32(&descriptor);
}

extern "C" void
_mlir_ciface_print_memref_0d_f32(StridedMemRefType<float, 0> *M) {
  impl::printMemRef(*M);
}
extern "C" void
_mlir_ciface_print_memref_1d_f32(StridedMemRefType<float, 1> *M) {
  impl::printMemRef(*M);
}
extern "C" void
_mlir_ciface_print_memref_2d_f32(StridedMemRefType<float, 2> *M) {
  impl::printMemRef(*M);
}
extern "C" void
_mlir_ciface_print_memref_3d_f32(StridedMemRefType<float, 3> *M) {
  impl::printMemRef(*M);
}
extern "C" void
_mlir_ciface_print_memref_4d_f32(StridedMemRefType<float, 4> *M) {
  impl::printMemRef(*M);
}

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

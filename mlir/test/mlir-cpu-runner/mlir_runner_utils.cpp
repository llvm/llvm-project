//===- mlir_runner_utils.cpp - Utils for MLIR CPU execution ---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for interfacing MLIR types with C code as well as printing,
// debugging etc.
//
//===----------------------------------------------------------------------===//

#include "include/mlir_runner_utils.h"

#include <cstdio>

extern "C" void
print_memref_vector_4x4xf32(StridedMemRefType<Vector2D<4, 4, float>, 2> *M) {
  impl::printMemRef(*M);
}

#define MEMREF_CASE(TYPE, RANK)                                                \
  case RANK:                                                                   \
    impl::printMemRef(*(static_cast<StridedMemRefType<TYPE, RANK> *>(ptr)));   \
    break

extern "C" void print_memref_i8(UnrankedMemRefType<int8_t> *M) {
  printUnrankedMemRefMetaData(std::cout, *M);
  int rank = M->rank;
  void *ptr = M->descriptor;

  switch (rank) {
    MEMREF_CASE(int8_t, 0);
    MEMREF_CASE(int8_t, 1);
    MEMREF_CASE(int8_t, 2);
    MEMREF_CASE(int8_t, 3);
    MEMREF_CASE(int8_t, 4);
  default:
    assert(0 && "Unsupported rank to print");
  }
}

extern "C" void print_memref_f32(UnrankedMemRefType<float> *M) {
  printUnrankedMemRefMetaData(std::cout, *M);
  int rank = M->rank;
  void *ptr = M->descriptor;

  switch (rank) {
    MEMREF_CASE(float, 0);
    MEMREF_CASE(float, 1);
    MEMREF_CASE(float, 2);
    MEMREF_CASE(float, 3);
    MEMREF_CASE(float, 4);
  default:
    assert(0 && "Unsupported rank to print");
  }
}

extern "C" void print_memref_0d_f32(StridedMemRefType<float, 0> *M) {
  impl::printMemRef(*M);
}
extern "C" void print_memref_1d_f32(StridedMemRefType<float, 1> *M) {
  impl::printMemRef(*M);
}
extern "C" void print_memref_2d_f32(StridedMemRefType<float, 2> *M) {
  impl::printMemRef(*M);
}
extern "C" void print_memref_3d_f32(StridedMemRefType<float, 3> *M) {
  impl::printMemRef(*M);
}
extern "C" void print_memref_4d_f32(StridedMemRefType<float, 4> *M) {
  impl::printMemRef(*M);
}

// Small runtime support "lib" for vector.print lowering.
// By providing elementary printing methods only, this
// library can remain fully unaware of low-level implementation
// details of our vectors.
extern "C" void print_f32(float f) { fprintf(stdout, "%g", f); }
extern "C" void print_f64(double d) { fprintf(stdout, "%lg", d); }
extern "C" void print_open() { fputs("( ", stdout); }
extern "C" void print_close() { fputs(" )", stdout); }
extern "C" void print_comma() { fputs(", ", stdout); }
extern "C" void print_newline() { fputc('\n', stdout); }

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
#include <chrono>

// NOLINTBEGIN(*-identifier-naming)

extern "C" void
_mlir_ciface_print_memref_shape_i8(UnrankedMemRefType<int8_t> *M) {
  std::cout << "Unranked Memref ";
  printMemRefMetaData(std::cout, DynamicMemRefType<int8_t>(*M));
  std::cout << "\n";
}

extern "C" void
_mlir_ciface_print_memref_shape_i32(UnrankedMemRefType<int32_t> *M) {
  std::cout << "Unranked Memref ";
  printMemRefMetaData(std::cout, DynamicMemRefType<int32_t>(*M));
  std::cout << "\n";
}

extern "C" void
_mlir_ciface_print_memref_shape_i64(UnrankedMemRefType<int64_t> *M) {
  std::cout << "Unranked Memref ";
  printMemRefMetaData(std::cout, DynamicMemRefType<int64_t>(*M));
  std::cout << "\n";
}

extern "C" void
_mlir_ciface_print_memref_shape_f32(UnrankedMemRefType<float> *M) {
  std::cout << "Unranked Memref ";
  printMemRefMetaData(std::cout, DynamicMemRefType<float>(*M));
  std::cout << "\n";
}

extern "C" void
_mlir_ciface_print_memref_shape_f64(UnrankedMemRefType<double> *M) {
  std::cout << "Unranked Memref ";
  printMemRefMetaData(std::cout, DynamicMemRefType<double>(*M));
  std::cout << "\n";
}

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

extern "C" void _mlir_ciface_print_memref_i64(UnrankedMemRefType<int64_t> *M) {
  impl::printMemRef(*M);
}

extern "C" void _mlir_ciface_print_memref_f32(UnrankedMemRefType<float> *M) {
  impl::printMemRef(*M);
}

extern "C" void _mlir_ciface_print_memref_f64(UnrankedMemRefType<double> *M) {
  impl::printMemRef(*M);
}

extern "C" int64_t _mlir_ciface_nano_time() {
  auto now = std::chrono::high_resolution_clock::now();
  auto duration = now.time_since_epoch();
  auto nanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
  return nanoseconds.count();
}

extern "C" void print_memref_i32(int64_t rank, void *ptr) {
  UnrankedMemRefType<int32_t> descriptor = {rank, ptr};
  _mlir_ciface_print_memref_i32(&descriptor);
}

extern "C" void print_memref_i64(int64_t rank, void *ptr) {
  UnrankedMemRefType<int64_t> descriptor = {rank, ptr};
  _mlir_ciface_print_memref_i64(&descriptor);
}

extern "C" void print_memref_f32(int64_t rank, void *ptr) {
  UnrankedMemRefType<float> descriptor = {rank, ptr};
  _mlir_ciface_print_memref_f32(&descriptor);
}

extern "C" void print_memref_f64(int64_t rank, void *ptr) {
  UnrankedMemRefType<double> descriptor = {rank, ptr};
  _mlir_ciface_print_memref_f64(&descriptor);
}

extern "C" void print_c_string(char *str) { printf("%s", str); }

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

extern "C" int64_t
_mlir_ciface_verifyMemRefI32(UnrankedMemRefType<int32_t> *actual,
                             UnrankedMemRefType<int32_t> *expected) {
  return impl::verifyMemRef(*actual, *expected);
}

extern "C" int64_t
_mlir_ciface_verifyMemRefF32(UnrankedMemRefType<float> *actual,
                             UnrankedMemRefType<float> *expected) {
  return impl::verifyMemRef(*actual, *expected);
}

extern "C" int64_t
_mlir_ciface_verifyMemRefF64(UnrankedMemRefType<double> *actual,
                             UnrankedMemRefType<double> *expected) {
  return impl::verifyMemRef(*actual, *expected);
}

extern "C" int64_t verifyMemRefI32(int64_t rank, void *actualPtr,
                                   void *expectedPtr) {
  UnrankedMemRefType<int32_t> actualDesc = {rank, actualPtr};
  UnrankedMemRefType<int32_t> expectedDesc = {rank, expectedPtr};
  return _mlir_ciface_verifyMemRefI32(&actualDesc, &expectedDesc);
}

extern "C" int64_t verifyMemRefF32(int64_t rank, void *actualPtr,
                                   void *expectedPtr) {
  UnrankedMemRefType<float> actualDesc = {rank, actualPtr};
  UnrankedMemRefType<float> expectedDesc = {rank, expectedPtr};
  return _mlir_ciface_verifyMemRefF32(&actualDesc, &expectedDesc);
}

extern "C" int64_t verifyMemRefF64(int64_t rank, void *actualPtr,
                                   void *expectedPtr) {
  UnrankedMemRefType<double> actualDesc = {rank, actualPtr};
  UnrankedMemRefType<double> expectedDesc = {rank, expectedPtr};
  return _mlir_ciface_verifyMemRefF64(&actualDesc, &expectedDesc);
}

// NOLINTEND(*-identifier-naming)

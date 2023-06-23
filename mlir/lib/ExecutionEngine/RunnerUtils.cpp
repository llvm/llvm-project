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

extern "C" void _mlir_ciface_printMemrefShapeI8(UnrankedMemRefType<int8_t> *M) {
  std::cout << "Unranked Memref ";
  printMemRefMetaData(std::cout, DynamicMemRefType<int8_t>(*M));
  std::cout << "\n";
}

extern "C" void
_mlir_ciface_printMemrefShapeI32(UnrankedMemRefType<int32_t> *M) {
  std::cout << "Unranked Memref ";
  printMemRefMetaData(std::cout, DynamicMemRefType<int32_t>(*M));
  std::cout << "\n";
}

extern "C" void
_mlir_ciface_printMemrefShapeI64(UnrankedMemRefType<int64_t> *M) {
  std::cout << "Unranked Memref ";
  printMemRefMetaData(std::cout, DynamicMemRefType<int64_t>(*M));
  std::cout << "\n";
}

extern "C" void _mlir_ciface_printMemrefShapeF32(UnrankedMemRefType<float> *M) {
  std::cout << "Unranked Memref ";
  printMemRefMetaData(std::cout, DynamicMemRefType<float>(*M));
  std::cout << "\n";
}

extern "C" void
_mlir_ciface_printMemrefShapeF64(UnrankedMemRefType<double> *M) {
  std::cout << "Unranked Memref ";
  printMemRefMetaData(std::cout, DynamicMemRefType<double>(*M));
  std::cout << "\n";
}

extern "C" void
_mlir_ciface_printMemrefShapeInd(UnrankedMemRefType<impl::index_type> *M) {
  std::cout << "Unranked Memref ";
  printMemRefMetaData(std::cout, DynamicMemRefType<impl::index_type>(*M));
  std::cout << "\n";
}

extern "C" void
_mlir_ciface_printMemrefShapeC32(UnrankedMemRefType<impl::complex32> *M) {
  std::cout << "Unranked Memref ";
  printMemRefMetaData(std::cout, DynamicMemRefType<impl::complex32>(*M));
  std::cout << "\n";
}

extern "C" void
_mlir_ciface_printMemrefShapeC64(UnrankedMemRefType<impl::complex64> *M) {
  std::cout << "Unranked Memref ";
  printMemRefMetaData(std::cout, DynamicMemRefType<impl::complex64>(*M));
  std::cout << "\n";
}

extern "C" void _mlir_ciface_printMemrefVector4x4xf32(
    StridedMemRefType<Vector2D<4, 4, float>, 2> *M) {
  impl::printMemRef(*M);
}

extern "C" void _mlir_ciface_printMemrefI8(UnrankedMemRefType<int8_t> *M) {
  impl::printMemRef(*M);
}

extern "C" void _mlir_ciface_printMemrefI32(UnrankedMemRefType<int32_t> *M) {
  impl::printMemRef(*M);
}

extern "C" void _mlir_ciface_printMemrefI64(UnrankedMemRefType<int64_t> *M) {
  impl::printMemRef(*M);
}

extern "C" void _mlir_ciface_printMemrefF32(UnrankedMemRefType<float> *M) {
  impl::printMemRef(*M);
}

extern "C" void _mlir_ciface_printMemrefF64(UnrankedMemRefType<double> *M) {
  impl::printMemRef(*M);
}

extern "C" void
_mlir_ciface_printMemrefInd(UnrankedMemRefType<impl::index_type> *M) {
  impl::printMemRef(*M);
}

extern "C" void
_mlir_ciface_printMemrefC32(UnrankedMemRefType<impl::complex32> *M) {
  impl::printMemRef(*M);
}

extern "C" void
_mlir_ciface_printMemrefC64(UnrankedMemRefType<impl::complex64> *M) {
  impl::printMemRef(*M);
}

extern "C" int64_t _mlir_ciface_nanoTime() {
  auto now = std::chrono::high_resolution_clock::now();
  auto duration = now.time_since_epoch();
  auto nanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
  return nanoseconds.count();
}

extern "C" void printMemrefI32(int64_t rank, void *ptr) {
  UnrankedMemRefType<int32_t> descriptor = {rank, ptr};
  _mlir_ciface_printMemrefI32(&descriptor);
}

extern "C" void printMemrefI64(int64_t rank, void *ptr) {
  UnrankedMemRefType<int64_t> descriptor = {rank, ptr};
  _mlir_ciface_printMemrefI64(&descriptor);
}

extern "C" void printMemrefF32(int64_t rank, void *ptr) {
  UnrankedMemRefType<float> descriptor = {rank, ptr};
  _mlir_ciface_printMemrefF32(&descriptor);
}

extern "C" void printMemrefF64(int64_t rank, void *ptr) {
  UnrankedMemRefType<double> descriptor = {rank, ptr};
  _mlir_ciface_printMemrefF64(&descriptor);
}

// Assume index_type is in fact uint64_t.
static_assert(std::is_same<impl::index_type, uint64_t>::value,
              "Expected index_type == uint64_t");
extern "C" void printMemrefInd(int64_t rank, void *ptr) {
  UnrankedMemRefType<impl::index_type> descriptor = {rank, ptr};
  _mlir_ciface_printMemrefInd(&descriptor);
}

extern "C" void printMemrefC32(int64_t rank, void *ptr) {
  UnrankedMemRefType<impl::complex32> descriptor = {rank, ptr};
  _mlir_ciface_printMemrefC32(&descriptor);
}

extern "C" void printMemrefC64(int64_t rank, void *ptr) {
  UnrankedMemRefType<impl::complex64> descriptor = {rank, ptr};
  _mlir_ciface_printMemrefC64(&descriptor);
}

extern "C" void printCString(char *str) { printf("%s", str); }

extern "C" void _mlir_ciface_printMemref0dF32(StridedMemRefType<float, 0> *M) {
  impl::printMemRef(*M);
}
extern "C" void _mlir_ciface_printMemref1dF32(StridedMemRefType<float, 1> *M) {
  impl::printMemRef(*M);
}
extern "C" void _mlir_ciface_printMemref2dF32(StridedMemRefType<float, 2> *M) {
  impl::printMemRef(*M);
}
extern "C" void _mlir_ciface_printMemref3dF32(StridedMemRefType<float, 3> *M) {
  impl::printMemRef(*M);
}
extern "C" void _mlir_ciface_printMemref4dF32(StridedMemRefType<float, 4> *M) {
  impl::printMemRef(*M);
}

extern "C" void _mlir_ciface_printMemref1dI8(StridedMemRefType<int8_t, 1> *M) {
  impl::printMemRef(*M);
}

extern "C" void
_mlir_ciface_printMemref1dI32(StridedMemRefType<int32_t, 1> *M) {
  impl::printMemRef(*M);
}

extern "C" void
_mlir_ciface_printMemref1dI64(StridedMemRefType<int64_t, 1> *M) {
  impl::printMemRef(*M);
}

extern "C" void _mlir_ciface_printMemref1dF64(StridedMemRefType<double, 1> *M) {
  impl::printMemRef(*M);
}

extern "C" void
_mlir_ciface_printMemref1dInd(StridedMemRefType<impl::index_type, 1> *M) {
  impl::printMemRef(*M);
}

extern "C" void
_mlir_ciface_printMemref1dC32(StridedMemRefType<impl::complex32, 1> *M) {
  impl::printMemRef(*M);
}

extern "C" void
_mlir_ciface_printMemref1dC64(StridedMemRefType<impl::complex64, 1> *M) {
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

extern "C" int64_t
_mlir_ciface_verifyMemRefInd(UnrankedMemRefType<impl::index_type> *actual,
                             UnrankedMemRefType<impl::index_type> *expected) {
  return impl::verifyMemRef(*actual, *expected);
}

extern "C" int64_t
_mlir_ciface_verifyMemRefC32(UnrankedMemRefType<impl::complex32> *actual,
                             UnrankedMemRefType<impl::complex32> *expected) {
  return impl::verifyMemRef(*actual, *expected);
}

extern "C" int64_t
_mlir_ciface_verifyMemRefC64(UnrankedMemRefType<impl::complex64> *actual,
                             UnrankedMemRefType<impl::complex64> *expected) {
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

extern "C" int64_t verifyMemRefInd(int64_t rank, void *actualPtr,
                                   void *expectedPtr) {
  UnrankedMemRefType<impl::index_type> actualDesc = {rank, actualPtr};
  UnrankedMemRefType<impl::index_type> expectedDesc = {rank, expectedPtr};
  return _mlir_ciface_verifyMemRefInd(&actualDesc, &expectedDesc);
}

extern "C" int64_t verifyMemRefC32(int64_t rank, void *actualPtr,
                                   void *expectedPtr) {
  UnrankedMemRefType<impl::complex32> actualDesc = {rank, actualPtr};
  UnrankedMemRefType<impl::complex32> expectedDesc = {rank, expectedPtr};
  return _mlir_ciface_verifyMemRefC32(&actualDesc, &expectedDesc);
}

extern "C" int64_t verifyMemRefC64(int64_t rank, void *actualPtr,
                                   void *expectedPtr) {
  UnrankedMemRefType<impl::complex64> actualDesc = {rank, actualPtr};
  UnrankedMemRefType<impl::complex64> expectedDesc = {rank, expectedPtr};
  return _mlir_ciface_verifyMemRefC64(&actualDesc, &expectedDesc);
}

// NOLINTEND(*-identifier-naming)

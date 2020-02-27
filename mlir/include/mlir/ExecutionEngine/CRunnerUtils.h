//===- CRunnerUtils.h - Utils for debugging MLIR execution ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares basic classes and functions to manipulate structured MLIR
// types at runtime. Entities in this file are must be retargetable, including
// on targets without a C++ runtime.
//
//===----------------------------------------------------------------------===//

#ifndef EXECUTIONENGINE_CRUNNERUTILS_H_
#define EXECUTIONENGINE_CRUNNERUTILS_H_

#ifdef _WIN32
#ifndef MLIR_CRUNNERUTILS_EXPORT
#ifdef mlir_c_runner_utils_EXPORTS
/* We are building this library */
#define MLIR_CRUNNERUTILS_EXPORT __declspec(dllexport)
#else
/* We are using this library */
#define MLIR_CRUNNERUTILS_EXPORT __declspec(dllimport)
#endif // mlir_c_runner_utils_EXPORTS
#endif // MLIR_CRUNNERUTILS_EXPORT
#else
#define MLIR_CRUNNERUTILS_EXPORT
#endif // _WIN32

#include <cstdint>

template <int N> void dropFront(int64_t arr[N], int64_t *res) {
  for (unsigned i = 1; i < N; ++i)
    *(res + i - 1) = arr[i];
}

/// StridedMemRef descriptor type with static rank.
template <typename T, int N> struct StridedMemRefType {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
  // This operator[] is extremely slow and only for sugaring purposes.
  StridedMemRefType<T, N - 1> operator[](int64_t idx) {
    StridedMemRefType<T, N - 1> res;
    res.basePtr = basePtr;
    res.data = data;
    res.offset = offset + idx * strides[0];
    dropFront<N>(sizes, res.sizes);
    dropFront<N>(strides, res.strides);
    return res;
  }
};

/// StridedMemRef descriptor type specialized for rank 1.
template <typename T> struct StridedMemRefType<T, 1> {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
  T &operator[](int64_t idx) { return *(data + offset + idx * strides[0]); }
};

/// StridedMemRef descriptor type specialized for rank 0.
template <typename T> struct StridedMemRefType<T, 0> {
  T *basePtr;
  T *data;
  int64_t offset;
};

// Unranked MemRef
template <typename T> struct UnrankedMemRefType {
  int64_t rank;
  void *descriptor;
};

// Small runtime support "lib" for vector.print lowering.
extern "C" MLIR_CRUNNERUTILS_EXPORT void print_f32(float f);
extern "C" MLIR_CRUNNERUTILS_EXPORT void print_f64(double d);
extern "C" MLIR_CRUNNERUTILS_EXPORT void print_open();
extern "C" MLIR_CRUNNERUTILS_EXPORT void print_close();
extern "C" MLIR_CRUNNERUTILS_EXPORT void print_comma();
extern "C" MLIR_CRUNNERUTILS_EXPORT void print_newline();

#endif // EXECUTIONENGINE_CRUNNERUTILS_H_

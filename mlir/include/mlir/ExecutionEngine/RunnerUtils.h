//===- RunnerUtils.h - Utils for debugging MLIR execution -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares basic classes and functions to debug structured MLIR
// types at runtime. Entities in this file may not be compatible with targets
// without a C++ runtime. These may be progressively migrated to CRunnerUtils.h
// over time.
//
//===----------------------------------------------------------------------===//

#ifndef EXECUTIONENGINE_RUNNERUTILS_H_
#define EXECUTIONENGINE_RUNNERUTILS_H_

#ifdef _WIN32
#ifndef MLIR_RUNNERUTILS_EXPORT
#ifdef mlir_runner_utils_EXPORTS
/* We are building this library */
#define MLIR_RUNNERUTILS_EXPORT __declspec(dllexport)
#else
/* We are using this library */
#define MLIR_RUNNERUTILS_EXPORT __declspec(dllimport)
#endif // mlir_runner_utils_EXPORTS
#endif // MLIR_RUNNERUTILS_EXPORT
#else
#define MLIR_RUNNERUTILS_EXPORT
#endif // _WIN32

#include <assert.h>
#include <iostream>

#include "mlir/ExecutionEngine/CRunnerUtils.h"

template <typename T, typename StreamType>
void printMemRefMetaData(StreamType &os, const DynamicMemRefType<T> &V) {
  os << "base@ = " << reinterpret_cast<void *>(V.data) << " rank = " << V.rank
     << " offset = " << V.offset;
  auto print = [&](const int64_t *ptr) {
    if (V.rank == 0)
      return;
    os << ptr[0];
    for (int64_t i = 1; i < V.rank; ++i)
      os << ", " << ptr[i];
  };
  os << " sizes = [";
  print(V.sizes);
  os << "] strides = [";
  print(V.strides);
  os << "]";
}

template <typename StreamType, typename T, int N>
void printMemRefMetaData(StreamType &os, StridedMemRefType<T, N> &V) {
  static_assert(N >= 0, "Expected N > 0");
  os << "MemRef ";
  printMemRefMetaData(os, DynamicMemRefType<T>(V));
}

template <typename StreamType, typename T>
void printUnrankedMemRefMetaData(StreamType &os, UnrankedMemRefType<T> &V) {
  os << "Unranked MemRef ";
  printMemRefMetaData(os, DynamicMemRefType<T>(V));
}

////////////////////////////////////////////////////////////////////////////////
// Templated instantiation follows.
////////////////////////////////////////////////////////////////////////////////
namespace impl {
template <typename T, int M, int... Dims>
std::ostream &operator<<(std::ostream &os, const Vector<T, M, Dims...> &v);

template <int... Dims> struct StaticSizeMult {
  static constexpr int value = 1;
};

template <int N, int... Dims> struct StaticSizeMult<N, Dims...> {
  static constexpr int value = N * StaticSizeMult<Dims...>::value;
};

static inline void printSpace(std::ostream &os, int count) {
  for (int i = 0; i < count; ++i) {
    os << ' ';
  }
}

template <typename T, int M, int... Dims> struct VectorDataPrinter {
  static void print(std::ostream &os, const Vector<T, M, Dims...> &val);
};

template <typename T, int M, int... Dims>
void VectorDataPrinter<T, M, Dims...>::print(std::ostream &os,
                                             const Vector<T, M, Dims...> &val) {
  static_assert(M > 0, "0 dimensioned tensor");
  static_assert(sizeof(val) == M * StaticSizeMult<Dims...>::value * sizeof(T),
                "Incorrect vector size!");
  // First
  os << "(" << val[0];
  if (M > 1)
    os << ", ";
  if (sizeof...(Dims) > 1)
    os << "\n";
  // Kernel
  for (unsigned i = 1; i + 1 < M; ++i) {
    printSpace(os, 2 * sizeof...(Dims));
    os << val[i] << ", ";
    if (sizeof...(Dims) > 1)
      os << "\n";
  }
  // Last
  if (M > 1) {
    printSpace(os, sizeof...(Dims));
    os << val[M - 1];
  }
  os << ")";
}

template <typename T, int M, int... Dims>
std::ostream &operator<<(std::ostream &os, const Vector<T, M, Dims...> &v) {
  VectorDataPrinter<T, M, Dims...>::print(os, v);
  return os;
}

template <typename T>
struct MemRefDataPrinter {
  static void print(std::ostream &os, T *base, int64_t dim, int64_t rank,
                    int64_t offset, const int64_t *sizes,
                    const int64_t *strides);
  static void printFirst(std::ostream &os, T *base, int64_t dim, int64_t rank,
                         int64_t offset, const int64_t *sizes,
                         const int64_t *strides);
  static void printLast(std::ostream &os, T *base, int64_t dim, int64_t rank,
                        int64_t offset, const int64_t *sizes,
                        const int64_t *strides);
};

template <typename T>
void MemRefDataPrinter<T>::printFirst(std::ostream &os, T *base, int64_t dim,
                                      int64_t rank, int64_t offset,
                                      const int64_t *sizes,
                                      const int64_t *strides) {
  os << "[";
  print(os, base, dim - 1, rank, offset, sizes + 1, strides + 1);
  // If single element, close square bracket and return early.
  if (sizes[0] <= 1) {
    os << "]";
    return;
  }
  os << ", ";
  if (dim > 1)
    os << "\n";
}

template <typename T>
void MemRefDataPrinter<T>::print(std::ostream &os, T *base, int64_t dim,
                                 int64_t rank, int64_t offset,
                                 const int64_t *sizes, const int64_t *strides) {
  if (dim == 0) {
    os << base[offset];
    return;
  }
  printFirst(os, base, dim, rank, offset, sizes, strides);
  for (unsigned i = 1; i + 1 < sizes[0]; ++i) {
    printSpace(os, rank - dim + 1);
    print(os, base, dim - 1, rank, offset + i * strides[0], sizes + 1,
          strides + 1);
    os << ", ";
    if (dim > 1)
      os << "\n";
  }
  if (sizes[0] <= 1)
    return;
  printLast(os, base, dim, rank, offset, sizes, strides);
}

template <typename T>
void MemRefDataPrinter<T>::printLast(std::ostream &os, T *base, int64_t dim,
                                     int64_t rank, int64_t offset,
                                     const int64_t *sizes,
                                     const int64_t *strides) {
  printSpace(os, rank - dim + 1);
  print(os, base, dim - 1, rank, offset + (sizes[0] - 1) * (*strides),
        sizes + 1, strides + 1);
  os << "]";
}

template <typename T>
void printMemRef(const DynamicMemRefType<T> &M) {
  printMemRefMetaData(std::cout, M);
  std::cout << " data = " << std::endl;
  if (M.rank == 0)
    std::cout << "[";
  MemRefDataPrinter<T>::print(std::cout, M.data, M.rank, M.rank, M.offset,
                              M.sizes, M.strides);
  if (M.rank == 0)
    std::cout << "]";
  std::cout << std::endl;
}

template <typename T, int N>
void printMemRef(StridedMemRefType<T, N> &M) {
  std::cout << "Memref ";
  printMemRef(DynamicMemRefType<T>(M));
}

template <typename T>
void printMemRef(UnrankedMemRefType<T> &M) {
  std::cout << "Unranked Memref ";
  printMemRef(DynamicMemRefType<T>(M));
}
} // namespace impl

////////////////////////////////////////////////////////////////////////////////
// Currently exposed C API.
////////////////////////////////////////////////////////////////////////////////
extern "C" MLIR_RUNNERUTILS_EXPORT void
_mlir_ciface_print_memref_i8(UnrankedMemRefType<int8_t> *M);
extern "C" MLIR_RUNNERUTILS_EXPORT void
_mlir_ciface_print_memref_f32(UnrankedMemRefType<float> *M);
extern "C" MLIR_RUNNERUTILS_EXPORT void
_mlir_ciface_print_memref_f64(UnrankedMemRefType<double> *M);

extern "C" MLIR_RUNNERUTILS_EXPORT void print_memref_i32(int64_t rank,
                                                         void *ptr);
extern "C" MLIR_RUNNERUTILS_EXPORT void print_memref_f32(int64_t rank,
                                                         void *ptr);
extern "C" MLIR_RUNNERUTILS_EXPORT void print_memref_f64(int64_t rank,
                                                         void *ptr);

extern "C" MLIR_RUNNERUTILS_EXPORT void
_mlir_ciface_print_memref_0d_f32(StridedMemRefType<float, 0> *M);
extern "C" MLIR_RUNNERUTILS_EXPORT void
_mlir_ciface_print_memref_1d_f32(StridedMemRefType<float, 1> *M);
extern "C" MLIR_RUNNERUTILS_EXPORT void
_mlir_ciface_print_memref_2d_f32(StridedMemRefType<float, 2> *M);
extern "C" MLIR_RUNNERUTILS_EXPORT void
_mlir_ciface_print_memref_3d_f32(StridedMemRefType<float, 3> *M);
extern "C" MLIR_RUNNERUTILS_EXPORT void
_mlir_ciface_print_memref_4d_f32(StridedMemRefType<float, 4> *M);

extern "C" MLIR_RUNNERUTILS_EXPORT void
_mlir_ciface_print_memref_vector_4x4xf32(
    StridedMemRefType<Vector2D<4, 4, float>, 2> *M);

#endif // EXECUTIONENGINE_RUNNERUTILS_H_

//===- mlir_runner_utils.h - Utils for debugging MLIR CPU execution -------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CPU_RUNNER_MLIRUTILS_H_
#define MLIR_CPU_RUNNER_MLIRUTILS_H_

#include <assert.h>
#include <cstdint>
#include <iostream>

#ifdef _WIN32
#ifndef MLIR_RUNNER_UTILS_EXPORT
#ifdef mlir_runner_utils_EXPORTS
/* We are building this library */
#define MLIR_RUNNER_UTILS_EXPORT __declspec(dllexport)
#else
/* We are using this library */
#define MLIR_RUNNER_UTILS_EXPORT __declspec(dllimport)
#endif
#endif
#else
#define MLIR_RUNNER_UTILS_EXPORT
#endif

template <typename T, int N> struct StridedMemRefType;
template <typename StreamType, typename T, int N>
void printMemRefMetaData(StreamType &os, StridedMemRefType<T, N> &V);

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

template <typename StreamType, typename T, int N>
void printMemRefMetaData(StreamType &os, StridedMemRefType<T, N> &V) {
  static_assert(N > 0, "Expected N > 0");
  os << "Memref base@ = " << reinterpret_cast<void *>(V.data) << " rank = " << N
     << " offset = " << V.offset << " sizes = [" << V.sizes[0];
  for (unsigned i = 1; i < N; ++i)
    os << ", " << V.sizes[i];
  os << "] strides = [" << V.strides[0];
  for (unsigned i = 1; i < N; ++i)
    os << ", " << V.strides[i];
  os << "]";
}

template <typename StreamType, typename T>
void printMemRefMetaData(StreamType &os, StridedMemRefType<T, 0> &V) {
  os << "Memref base@ = " << reinterpret_cast<void *>(V.data) << " rank = 0"
     << " offset = " << V.offset;
}

template <typename T, typename StreamType>
void printUnrankedMemRefMetaData(StreamType &os, UnrankedMemRefType<T> &V) {
  os << "Unranked Memref rank = " << V.rank << " "
     << "descriptor@ = " << reinterpret_cast<void *>(V.descriptor) << "\n";
}

template <typename T, int Dim, int... Dims> struct Vector {
  Vector<T, Dims...> vector[Dim];
};
template <typename T, int Dim> struct Vector<T, Dim> { T vector[Dim]; };

template <int D1, typename T> using Vector1D = Vector<T, D1>;
template <int D1, int D2, typename T> using Vector2D = Vector<T, D1, D2>;
template <int D1, int D2, int D3, typename T>
using Vector3D = Vector<T, D1, D2, D3>;
template <int D1, int D2, int D3, int D4, typename T>
using Vector4D = Vector<T, D1, D2, D3, D4>;

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
  os << "(" << val.vector[0];
  if (M > 1)
    os << ", ";
  if (sizeof...(Dims) > 1)
    os << "\n";
  // Kernel
  for (unsigned i = 1; i + 1 < M; ++i) {
    printSpace(os, 2 * sizeof...(Dims));
    os << val.vector[i] << ", ";
    if (sizeof...(Dims) > 1)
      os << "\n";
  }
  // Last
  if (M > 1) {
    printSpace(os, sizeof...(Dims));
    os << val.vector[M - 1];
  }
  os << ")";
}

template <typename T, int M, int... Dims>
std::ostream &operator<<(std::ostream &os, const Vector<T, M, Dims...> &v) {
  VectorDataPrinter<T, M, Dims...>::print(os, v);
  return os;
}

template <typename T, int N> struct MemRefDataPrinter {
  static void print(std::ostream &os, T *base, int64_t rank, int64_t offset,
                    int64_t *sizes, int64_t *strides);
  static void printFirst(std::ostream &os, T *base, int64_t rank,
                         int64_t offset, int64_t *sizes, int64_t *strides);
  static void printLast(std::ostream &os, T *base, int64_t rank, int64_t offset,
                        int64_t *sizes, int64_t *strides);
};

template <typename T> struct MemRefDataPrinter<T, 0> {
  static void print(std::ostream &os, T *base, int64_t rank, int64_t offset,
                    int64_t *sizes = nullptr, int64_t *strides = nullptr);
};

template <typename T, int N>
void MemRefDataPrinter<T, N>::printFirst(std::ostream &os, T *base,
                                         int64_t rank, int64_t offset,
                                         int64_t *sizes, int64_t *strides) {
  os << "[";
  MemRefDataPrinter<T, N - 1>::print(os, base, rank, offset, sizes + 1,
                                     strides + 1);
  // If single element, close square bracket and return early.
  if (sizes[0] <= 1) {
    os << "]";
    return;
  }
  os << ", ";
  if (N > 1)
    os << "\n";
}

template <typename T, int N>
void MemRefDataPrinter<T, N>::print(std::ostream &os, T *base, int64_t rank,
                                    int64_t offset, int64_t *sizes,
                                    int64_t *strides) {
  printFirst(os, base, rank, offset, sizes, strides);
  for (unsigned i = 1; i + 1 < sizes[0]; ++i) {
    printSpace(os, rank - N + 1);
    MemRefDataPrinter<T, N - 1>::print(os, base, rank, offset + i * strides[0],
                                       sizes + 1, strides + 1);
    os << ", ";
    if (N > 1)
      os << "\n";
  }
  if (sizes[0] <= 1)
    return;
  printLast(os, base, rank, offset, sizes, strides);
}

template <typename T, int N>
void MemRefDataPrinter<T, N>::printLast(std::ostream &os, T *base, int64_t rank,
                                        int64_t offset, int64_t *sizes,
                                        int64_t *strides) {
  printSpace(os, rank - N + 1);
  MemRefDataPrinter<T, N - 1>::print(os, base, rank,
                                     offset + (sizes[0] - 1) * (*strides),
                                     sizes + 1, strides + 1);
  os << "]";
}

template <typename T>
void MemRefDataPrinter<T, 0>::print(std::ostream &os, T *base, int64_t rank,
                                    int64_t offset, int64_t *sizes,
                                    int64_t *strides) {
  os << base[offset];
}

template <typename T, int N> void printMemRef(StridedMemRefType<T, N> &M) {
  static_assert(N > 0, "Expected N > 0");
  printMemRefMetaData(std::cout, M);
  std::cout << " data = " << std::endl;
  MemRefDataPrinter<T, N>::print(std::cout, M.data, N, M.offset, M.sizes,
                                 M.strides);
  std::cout << std::endl;
}

template <typename T> void printMemRef(StridedMemRefType<T, 0> &M) {
  printMemRefMetaData(std::cout, M);
  std::cout << " data = " << std::endl;
  std::cout << "[";
  MemRefDataPrinter<T, 0>::print(std::cout, M.data, 0, M.offset);
  std::cout << "]" << std::endl;
}
} // namespace impl

////////////////////////////////////////////////////////////////////////////////
// Currently exposed C API.
////////////////////////////////////////////////////////////////////////////////
extern "C" MLIR_RUNNER_UTILS_EXPORT void
print_memref_i8(UnrankedMemRefType<int8_t> *M);
extern "C" MLIR_RUNNER_UTILS_EXPORT void
print_memref_f32(UnrankedMemRefType<float> *M);

extern "C" MLIR_RUNNER_UTILS_EXPORT void
print_memref_0d_f32(StridedMemRefType<float, 0> *M);
extern "C" MLIR_RUNNER_UTILS_EXPORT void
print_memref_1d_f32(StridedMemRefType<float, 1> *M);
extern "C" MLIR_RUNNER_UTILS_EXPORT void
print_memref_2d_f32(StridedMemRefType<float, 2> *M);
extern "C" MLIR_RUNNER_UTILS_EXPORT void
print_memref_3d_f32(StridedMemRefType<float, 3> *M);
extern "C" MLIR_RUNNER_UTILS_EXPORT void
print_memref_4d_f32(StridedMemRefType<float, 4> *M);

extern "C" MLIR_RUNNER_UTILS_EXPORT void
print_memref_vector_4x4xf32(StridedMemRefType<Vector2D<4, 4, float>, 2> *M);

// Small runtime support "lib" for vector.print lowering.
extern "C" MLIR_RUNNER_UTILS_EXPORT void print_f32(float f);
extern "C" MLIR_RUNNER_UTILS_EXPORT void print_f64(double d);
extern "C" MLIR_RUNNER_UTILS_EXPORT void print_open();
extern "C" MLIR_RUNNER_UTILS_EXPORT void print_close();
extern "C" MLIR_RUNNER_UTILS_EXPORT void print_comma();
extern "C" MLIR_RUNNER_UTILS_EXPORT void print_newline();

#endif // MLIR_CPU_RUNNER_MLIRUTILS_H_

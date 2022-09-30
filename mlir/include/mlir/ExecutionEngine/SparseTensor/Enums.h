//===- Enums.h - Enums shared with the runtime ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Typedefs and enums for the lightweight runtime support library for
// sparse tensor manipulations.  These are required to be public so that
// they can be shared with `Transforms/SparseTensorConversion.cpp`, since
// they define the arguments to the public functions declared later on.
//
// This file also defines x-macros <https://en.wikipedia.org/wiki/X_Macro>
// so that we can generate variations of the public functions for each
// supported primary- and/or overhead-type.
//
// This file is part of the lightweight runtime support library for sparse
// tensor manipulations.  The functionality of the support library is meant
// to simplify benchmarking, testing, and debugging MLIR code operating on
// sparse tensors.  However, the provided functionality is **not** part of
// core MLIR itself.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_SPARSETENSOR_ENUMS_H
#define MLIR_EXECUTIONENGINE_SPARSETENSOR_ENUMS_H

#include "mlir/ExecutionEngine/Float16bits.h"

#include <cinttypes>
#include <complex>

#ifdef _WIN32
#ifdef mlir_sparse_tensor_utils_EXPORTS // We are building this library
#define MLIR_SPARSETENSOR_EXPORT __declspec(dllexport)
#define MLIR_SPARSETENSOR_DEFINE_FUNCTIONS
#else // We are using this library
#define MLIR_SPARSETENSOR_EXPORT __declspec(dllimport)
#endif // mlir_sparse_tensor_utils_EXPORTS
#else  // Non-windows: use visibility attributes.
#define MLIR_SPARSETENSOR_EXPORT __attribute__((visibility("default")))
#define MLIR_SPARSETENSOR_DEFINE_FUNCTIONS
#endif // _WIN32

namespace mlir {
namespace sparse_tensor {

/// This type is used in the public API at all places where MLIR expects
/// values with the built-in type "index".  For now, we simply assume that
/// type is 64-bit, but targets with different "index" bit widths should
/// link with an alternatively built runtime support library.
// TODO: support such targets?
using index_type = uint64_t;

/// Encoding of overhead types (both pointer overhead and indices
/// overhead), for "overloading" @newSparseTensor.
enum class MLIR_SPARSETENSOR_EXPORT OverheadType : uint32_t {
  kIndex = 0,
  kU64 = 1,
  kU32 = 2,
  kU16 = 3,
  kU8 = 4
};

// This x-macro calls its argument on every overhead type which has
// fixed-width.  It excludes `index_type` because that type is often
// handled specially (e.g., by translating it into the architecture-dependent
// equivalent fixed-width overhead type).
#define FOREVERY_FIXED_O(DO)                                                   \
  DO(64, uint64_t)                                                             \
  DO(32, uint32_t)                                                             \
  DO(16, uint16_t)                                                             \
  DO(8, uint8_t)

// This x-macro calls its argument on every overhead type, including
// `index_type`.
#define FOREVERY_O(DO)                                                         \
  FOREVERY_FIXED_O(DO)                                                         \
  DO(0, index_type)

// These are not just shorthands but indicate the particular
// implementation used (e.g., as opposed to C99's `complex double`,
// or MLIR's `ComplexType`).
using complex64 = std::complex<double>;
using complex32 = std::complex<float>;

/// Encoding of the elemental type, for "overloading" @newSparseTensor.
enum class MLIR_SPARSETENSOR_EXPORT PrimaryType : uint32_t {
  kF64 = 1,
  kF32 = 2,
  kF16 = 3,
  kBF16 = 4,
  kI64 = 5,
  kI32 = 6,
  kI16 = 7,
  kI8 = 8,
  kC64 = 9,
  kC32 = 10
};

// This x-macro includes all `V` types.
#define FOREVERY_V(DO)                                                         \
  DO(F64, double)                                                              \
  DO(F32, float)                                                               \
  DO(F16, f16)                                                                 \
  DO(BF16, bf16)                                                               \
  DO(I64, int64_t)                                                             \
  DO(I32, int32_t)                                                             \
  DO(I16, int16_t)                                                             \
  DO(I8, int8_t)                                                               \
  DO(C64, complex64)                                                           \
  DO(C32, complex32)

constexpr MLIR_SPARSETENSOR_EXPORT bool
isFloatingPrimaryType(PrimaryType valTy) {
  return PrimaryType::kF64 <= valTy && valTy <= PrimaryType::kBF16;
}

constexpr MLIR_SPARSETENSOR_EXPORT bool
isIntegralPrimaryType(PrimaryType valTy) {
  return PrimaryType::kI64 <= valTy && valTy <= PrimaryType::kI8;
}

constexpr MLIR_SPARSETENSOR_EXPORT bool isRealPrimaryType(PrimaryType valTy) {
  return PrimaryType::kF64 <= valTy && valTy <= PrimaryType::kI8;
}

constexpr MLIR_SPARSETENSOR_EXPORT bool
isComplexPrimaryType(PrimaryType valTy) {
  return PrimaryType::kC64 <= valTy && valTy <= PrimaryType::kC32;
}

/// The actions performed by @newSparseTensor.
enum class MLIR_SPARSETENSOR_EXPORT Action : uint32_t {
  kEmpty = 0,
  kFromFile = 1,
  kFromCOO = 2,
  kSparseToSparse = 3,
  kEmptyCOO = 4,
  kToCOO = 5,
  kToIterator = 6,
};

/// This enum mimics `SparseTensorEncodingAttr::DimLevelType` for
/// breaking dependency cycles.  `SparseTensorEncodingAttr::DimLevelType`
/// is the source of truth and this enum should be kept consistent with it.
enum class MLIR_SPARSETENSOR_EXPORT DimLevelType : uint8_t {
  kDense = 0,
  kCompressed = 1,
  kCompressedNu = 2,
  kCompressedNo = 3,
  kCompressedNuNo = 4,
  kSingleton = 5,
  kSingletonNu = 6,
  kSingletonNo = 7,
  kSingletonNuNo = 8,
};

/// Check if the `DimLevelType` is dense.
constexpr MLIR_SPARSETENSOR_EXPORT bool isDenseDLT(DimLevelType dlt) {
  return dlt == DimLevelType::kDense;
}

/// Check if the `DimLevelType` is compressed (regardless of properties).
constexpr MLIR_SPARSETENSOR_EXPORT bool isCompressedDLT(DimLevelType dlt) {
  switch (dlt) {
  case DimLevelType::kCompressed:
  case DimLevelType::kCompressedNu:
  case DimLevelType::kCompressedNo:
  case DimLevelType::kCompressedNuNo:
    return true;
  default:
    return false;
  }
}

/// Check if the `DimLevelType` is singleton (regardless of properties).
constexpr MLIR_SPARSETENSOR_EXPORT bool isSingletonDLT(DimLevelType dlt) {
  switch (dlt) {
  case DimLevelType::kSingleton:
  case DimLevelType::kSingletonNu:
  case DimLevelType::kSingletonNo:
  case DimLevelType::kSingletonNuNo:
    return true;
  default:
    return false;
  }
}

/// Check if the `DimLevelType` is ordered (regardless of storage format).
constexpr MLIR_SPARSETENSOR_EXPORT bool isOrderedDLT(DimLevelType dlt) {
  switch (dlt) {
  case DimLevelType::kCompressedNo:
  case DimLevelType::kCompressedNuNo:
  case DimLevelType::kSingletonNo:
  case DimLevelType::kSingletonNuNo:
    return false;
  default:
    return true;
  }
}

/// Check if the `DimLevelType` is unique (regardless of storage format).
constexpr MLIR_SPARSETENSOR_EXPORT bool isUniqueDLT(DimLevelType dlt) {
  switch (dlt) {
  case DimLevelType::kCompressedNu:
  case DimLevelType::kCompressedNuNo:
  case DimLevelType::kSingletonNu:
  case DimLevelType::kSingletonNuNo:
    return false;
  default:
    return true;
  }
}

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_SPARSETENSOR_ENUMS_H

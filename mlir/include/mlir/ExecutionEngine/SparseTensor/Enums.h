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
enum class OverheadType : uint32_t {
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
#define MLIR_SPARSETENSOR_FOREVERY_FIXED_O(DO)                                 \
  DO(64, uint64_t)                                                             \
  DO(32, uint32_t)                                                             \
  DO(16, uint16_t)                                                             \
  DO(8, uint8_t)

// This x-macro calls its argument on every overhead type, including
// `index_type`.
#define MLIR_SPARSETENSOR_FOREVERY_O(DO)                                       \
  MLIR_SPARSETENSOR_FOREVERY_FIXED_O(DO)                                       \
  DO(0, index_type)

// These are not just shorthands but indicate the particular
// implementation used (e.g., as opposed to C99's `complex double`,
// or MLIR's `ComplexType`).
using complex64 = std::complex<double>;
using complex32 = std::complex<float>;

/// Encoding of the elemental type, for "overloading" @newSparseTensor.
enum class PrimaryType : uint32_t {
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
#define MLIR_SPARSETENSOR_FOREVERY_V(DO)                                       \
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

constexpr bool isFloatingPrimaryType(PrimaryType valTy) {
  return PrimaryType::kF64 <= valTy && valTy <= PrimaryType::kBF16;
}

constexpr bool isIntegralPrimaryType(PrimaryType valTy) {
  return PrimaryType::kI64 <= valTy && valTy <= PrimaryType::kI8;
}

constexpr bool isRealPrimaryType(PrimaryType valTy) {
  return PrimaryType::kF64 <= valTy && valTy <= PrimaryType::kI8;
}

constexpr bool isComplexPrimaryType(PrimaryType valTy) {
  return PrimaryType::kC64 <= valTy && valTy <= PrimaryType::kC32;
}

/// The actions performed by @newSparseTensor.
enum class Action : uint32_t {
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
enum class DimLevelType : uint8_t {
  kDense = 4,           // 0b001_00
  kCompressed = 8,      // 0b010_00
  kCompressedNu = 9,    // 0b010_01
  kCompressedNo = 10,   // 0b010_10
  kCompressedNuNo = 11, // 0b010_11
  kSingleton = 16,      // 0b100_00
  kSingletonNu = 17,    // 0b100_01
  kSingletonNo = 18,    // 0b100_10
  kSingletonNuNo = 19,  // 0b100_11
};

/// Check if the `DimLevelType` is dense.
constexpr bool isDenseDLT(DimLevelType dlt) {
  return dlt == DimLevelType::kDense;
}

/// Check if the `DimLevelType` is compressed (regardless of properties).
constexpr bool isCompressedDLT(DimLevelType dlt) {
  return static_cast<uint8_t>(dlt) &
         static_cast<uint8_t>(DimLevelType::kCompressed);
}

/// Check if the `DimLevelType` is singleton (regardless of properties).
constexpr bool isSingletonDLT(DimLevelType dlt) {
  return static_cast<uint8_t>(dlt) &
         static_cast<uint8_t>(DimLevelType::kSingleton);
}

/// Check if the `DimLevelType` is ordered (regardless of storage format).
constexpr bool isOrderedDLT(DimLevelType dlt) {
  return !(static_cast<uint8_t>(dlt) & 2);
}

/// Check if the `DimLevelType` is unique (regardless of storage format).
constexpr bool isUniqueDLT(DimLevelType dlt) {
  return !(static_cast<uint8_t>(dlt) & 1);
}

// Ensure the above predicates work as intended.
static_assert((!isCompressedDLT(DimLevelType::kDense) &&
               isCompressedDLT(DimLevelType::kCompressed) &&
               isCompressedDLT(DimLevelType::kCompressedNu) &&
               isCompressedDLT(DimLevelType::kCompressedNo) &&
               isCompressedDLT(DimLevelType::kCompressedNuNo) &&
               !isCompressedDLT(DimLevelType::kSingleton) &&
               !isCompressedDLT(DimLevelType::kSingletonNu) &&
               !isCompressedDLT(DimLevelType::kSingletonNo) &&
               !isCompressedDLT(DimLevelType::kSingletonNuNo)),
              "isCompressedDLT definition is broken");

static_assert((!isSingletonDLT(DimLevelType::kDense) &&
               !isSingletonDLT(DimLevelType::kCompressed) &&
               !isSingletonDLT(DimLevelType::kCompressedNu) &&
               !isSingletonDLT(DimLevelType::kCompressedNo) &&
               !isSingletonDLT(DimLevelType::kCompressedNuNo) &&
               isSingletonDLT(DimLevelType::kSingleton) &&
               isSingletonDLT(DimLevelType::kSingletonNu) &&
               isSingletonDLT(DimLevelType::kSingletonNo) &&
               isSingletonDLT(DimLevelType::kSingletonNuNo)),
              "isSingletonDLT definition is broken");

static_assert((isOrderedDLT(DimLevelType::kDense) &&
               isOrderedDLT(DimLevelType::kCompressed) &&
               isOrderedDLT(DimLevelType::kCompressedNu) &&
               !isOrderedDLT(DimLevelType::kCompressedNo) &&
               !isOrderedDLT(DimLevelType::kCompressedNuNo) &&
               isOrderedDLT(DimLevelType::kSingleton) &&
               isOrderedDLT(DimLevelType::kSingletonNu) &&
               !isOrderedDLT(DimLevelType::kSingletonNo) &&
               !isOrderedDLT(DimLevelType::kSingletonNuNo)),
              "isOrderedDLT definition is broken");

static_assert((isUniqueDLT(DimLevelType::kDense) &&
               isUniqueDLT(DimLevelType::kCompressed) &&
               !isUniqueDLT(DimLevelType::kCompressedNu) &&
               isUniqueDLT(DimLevelType::kCompressedNo) &&
               !isUniqueDLT(DimLevelType::kCompressedNuNo) &&
               isUniqueDLT(DimLevelType::kSingleton) &&
               !isUniqueDLT(DimLevelType::kSingletonNu) &&
               isUniqueDLT(DimLevelType::kSingletonNo) &&
               !isUniqueDLT(DimLevelType::kSingletonNuNo)),
              "isUniqueDLT definition is broken");

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_SPARSETENSOR_ENUMS_H

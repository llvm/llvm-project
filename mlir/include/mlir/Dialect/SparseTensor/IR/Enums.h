//===- Enums.h - Enums for the SparseTensor dialect -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Typedefs and enums shared between MLIR code for manipulating the
// IR, and the lightweight runtime support library for sparse tensor
// manipulations.  That is, all the enums are used to define the API
// of the runtime library and hence are also needed when generating
// calls into the runtime library.  Moveover, the `DimLevelType` enum
// is also used as the internal IR encoding of dimension level types,
// to avoid code duplication (e.g., for the predicates).
//
// This file also defines x-macros <https://en.wikipedia.org/wiki/X_Macro>
// so that we can generate variations of the public functions for each
// supported primary- and/or overhead-type.
//
// Because this file defines a library which is a dependency of the
// runtime library itself, this file must not depend on any MLIR internals
// (e.g., operators, attributes, ArrayRefs, etc) lest the runtime library
// inherit those dependencies.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_IR_ENUMS_H
#define MLIR_DIALECT_SPARSETENSOR_IR_ENUMS_H

// NOTE: Client code will need to include "mlir/ExecutionEngine/Float16bits.h"
// if they want to use the `MLIR_SPARSETENSOR_FOREVERY_V` macro.

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
  // newSparseTensor no longer handles `kFromFile=1`, so we leave this
  // number reserved to help catch any code that still needs updating.
  kFromCOO = 2,
  kSparseToSparse = 3,
  kEmptyCOO = 4,
  kToCOO = 5,
  kToIterator = 6,
};

/// This enum defines all the sparse representations supportable by
/// the SparseTensor dialect.  We use a lightweight encoding to encode
/// both the "format" per se (dense, compressed, singleton) as well as
/// the "properties" (ordered, unique).  The encoding is chosen for
/// performance of the runtime library, and thus may change in future
/// versions; consequently, client code should use the predicate functions
/// defined below, rather than relying on knowledge about the particular
/// binary encoding.
///
/// The `Undef` "format" is a special value used internally for cases
/// where we need to store an undefined or indeterminate `DimLevelType`.
/// It should not be used externally, since it does not indicate an
/// actual/representable format.
enum class DimLevelType : uint8_t {
  Undef = 0,           // 0b000_00
  Dense = 4,           // 0b001_00
  Compressed = 8,      // 0b010_00
  CompressedNu = 9,    // 0b010_01
  CompressedNo = 10,   // 0b010_10
  CompressedNuNo = 11, // 0b010_11
  Singleton = 16,      // 0b100_00
  SingletonNu = 17,    // 0b100_01
  SingletonNo = 18,    // 0b100_10
  SingletonNuNo = 19,  // 0b100_11
};

/// Check that the `DimLevelType` contains a valid (possibly undefined) value.
constexpr bool isValidDLT(DimLevelType dlt) {
  const uint8_t formatBits = static_cast<uint8_t>(dlt) >> 2;
  const uint8_t propertyBits = static_cast<uint8_t>(dlt) & 3;
  // If undefined or dense, then must be unique and ordered.
  // Otherwise, the format must be one of the known ones.
  return (formatBits <= 1) ? (propertyBits == 0)
                           : (formatBits == 2 || formatBits == 4);
}

/// Check if the `DimLevelType` is the special undefined value.
constexpr bool isUndefDLT(DimLevelType dlt) {
  return dlt == DimLevelType::Undef;
}

/// Check if the `DimLevelType` is dense.
constexpr bool isDenseDLT(DimLevelType dlt) {
  return dlt == DimLevelType::Dense;
}

// We use the idiom `(dlt & ~3) == format` in order to only return true
// for valid DLTs.  Whereas the `dlt & format` idiom is a bit faster but
// can return false-positives on invalid DLTs.

/// Check if the `DimLevelType` is compressed (regardless of properties).
constexpr bool isCompressedDLT(DimLevelType dlt) {
  return (static_cast<uint8_t>(dlt) & ~3) ==
         static_cast<uint8_t>(DimLevelType::Compressed);
}

/// Check if the `DimLevelType` is singleton (regardless of properties).
constexpr bool isSingletonDLT(DimLevelType dlt) {
  return (static_cast<uint8_t>(dlt) & ~3) ==
         static_cast<uint8_t>(DimLevelType::Singleton);
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
static_assert((isValidDLT(DimLevelType::Undef) &&
               isValidDLT(DimLevelType::Dense) &&
               isValidDLT(DimLevelType::Compressed) &&
               isValidDLT(DimLevelType::CompressedNu) &&
               isValidDLT(DimLevelType::CompressedNo) &&
               isValidDLT(DimLevelType::CompressedNuNo) &&
               isValidDLT(DimLevelType::Singleton) &&
               isValidDLT(DimLevelType::SingletonNu) &&
               isValidDLT(DimLevelType::SingletonNo) &&
               isValidDLT(DimLevelType::SingletonNuNo)),
              "isValidDLT definition is broken");

static_assert((!isCompressedDLT(DimLevelType::Dense) &&
               isCompressedDLT(DimLevelType::Compressed) &&
               isCompressedDLT(DimLevelType::CompressedNu) &&
               isCompressedDLT(DimLevelType::CompressedNo) &&
               isCompressedDLT(DimLevelType::CompressedNuNo) &&
               !isCompressedDLT(DimLevelType::Singleton) &&
               !isCompressedDLT(DimLevelType::SingletonNu) &&
               !isCompressedDLT(DimLevelType::SingletonNo) &&
               !isCompressedDLT(DimLevelType::SingletonNuNo)),
              "isCompressedDLT definition is broken");

static_assert((!isSingletonDLT(DimLevelType::Dense) &&
               !isSingletonDLT(DimLevelType::Compressed) &&
               !isSingletonDLT(DimLevelType::CompressedNu) &&
               !isSingletonDLT(DimLevelType::CompressedNo) &&
               !isSingletonDLT(DimLevelType::CompressedNuNo) &&
               isSingletonDLT(DimLevelType::Singleton) &&
               isSingletonDLT(DimLevelType::SingletonNu) &&
               isSingletonDLT(DimLevelType::SingletonNo) &&
               isSingletonDLT(DimLevelType::SingletonNuNo)),
              "isSingletonDLT definition is broken");

static_assert((isOrderedDLT(DimLevelType::Dense) &&
               isOrderedDLT(DimLevelType::Compressed) &&
               isOrderedDLT(DimLevelType::CompressedNu) &&
               !isOrderedDLT(DimLevelType::CompressedNo) &&
               !isOrderedDLT(DimLevelType::CompressedNuNo) &&
               isOrderedDLT(DimLevelType::Singleton) &&
               isOrderedDLT(DimLevelType::SingletonNu) &&
               !isOrderedDLT(DimLevelType::SingletonNo) &&
               !isOrderedDLT(DimLevelType::SingletonNuNo)),
              "isOrderedDLT definition is broken");

static_assert((isUniqueDLT(DimLevelType::Dense) &&
               isUniqueDLT(DimLevelType::Compressed) &&
               !isUniqueDLT(DimLevelType::CompressedNu) &&
               isUniqueDLT(DimLevelType::CompressedNo) &&
               !isUniqueDLT(DimLevelType::CompressedNuNo) &&
               isUniqueDLT(DimLevelType::Singleton) &&
               !isUniqueDLT(DimLevelType::SingletonNu) &&
               isUniqueDLT(DimLevelType::SingletonNo) &&
               !isUniqueDLT(DimLevelType::SingletonNuNo)),
              "isUniqueDLT definition is broken");

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_IR_ENUMS_H

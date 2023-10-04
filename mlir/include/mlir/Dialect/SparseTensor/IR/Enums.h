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
#include <optional>

namespace mlir {
namespace sparse_tensor {

/// This type is used in the public API at all places where MLIR expects
/// values with the built-in type "index".  For now, we simply assume that
/// type is 64-bit, but targets with different "index" bitwidths should
/// link with an alternatively built runtime support library.
using index_type = uint64_t;

/// Encoding of overhead types (both position overhead and coordinate
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

// This x-macro includes all `V` types and supports variadic arguments.
#define MLIR_SPARSETENSOR_FOREVERY_V_VAR(DO, ...)                              \
  DO(F64, double, __VA_ARGS__)                                                 \
  DO(F32, float, __VA_ARGS__)                                                  \
  DO(F16, f16, __VA_ARGS__)                                                    \
  DO(BF16, bf16, __VA_ARGS__)                                                  \
  DO(I64, int64_t, __VA_ARGS__)                                                \
  DO(I32, int32_t, __VA_ARGS__)                                                \
  DO(I16, int16_t, __VA_ARGS__)                                                \
  DO(I8, int8_t, __VA_ARGS__)                                                  \
  DO(C64, complex64, __VA_ARGS__)                                              \
  DO(C32, complex32, __VA_ARGS__)

// This x-macro calls its argument on every pair of overhead and `V` types.
#define MLIR_SPARSETENSOR_FOREVERY_V_O(DO)                                     \
  MLIR_SPARSETENSOR_FOREVERY_V_VAR(DO, 64, uint64_t)                           \
  MLIR_SPARSETENSOR_FOREVERY_V_VAR(DO, 32, uint32_t)                           \
  MLIR_SPARSETENSOR_FOREVERY_V_VAR(DO, 16, uint16_t)                           \
  MLIR_SPARSETENSOR_FOREVERY_V_VAR(DO, 8, uint8_t)                             \
  MLIR_SPARSETENSOR_FOREVERY_V_VAR(DO, 0, index_type)

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
  kPack = 7,
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
///
// TODO: We should generalize TwoOutOfFour to N out of M and use property to
// encode the value of N and M.
// TODO: Update DimLevelType to use lower 8 bits for storage formats and the
// higher 4 bits to store level properties. Consider LooseCompressed and
// TwoOutOfFour as properties instead of formats.
enum class DimLevelType : uint8_t {
  Undef = 0,                // 0b00000_00
  Dense = 4,                // 0b00001_00
  Compressed = 8,           // 0b00010_00
  CompressedNu = 9,         // 0b00010_01
  CompressedNo = 10,        // 0b00010_10
  CompressedNuNo = 11,      // 0b00010_11
  Singleton = 16,           // 0b00100_00
  SingletonNu = 17,         // 0b00100_01
  SingletonNo = 18,         // 0b00100_10
  SingletonNuNo = 19,       // 0b00100_11
  LooseCompressed = 32,     // 0b01000_00
  LooseCompressedNu = 33,   // 0b01000_01
  LooseCompressedNo = 34,   // 0b01000_10
  LooseCompressedNuNo = 35, // 0b01000_11
  TwoOutOfFour = 64,        // 0b10000_00
};

/// This enum defines all supported storage format without the level properties.
enum class LevelFormat : uint8_t {
  Dense = 4,            // 0b00001_00
  Compressed = 8,       // 0b00010_00
  Singleton = 16,       // 0b00100_00
  LooseCompressed = 32, // 0b01000_00
  TwoOutOfFour = 64,    // 0b10000_00
};

/// This enum defines all the nondefault properties for storage formats.
enum class LevelNondefaultProperty : uint8_t {
  Nonunique = 1,  // 0b00000_01
  Nonordered = 2, // 0b00000_10
};

/// Returns string representation of the given dimension level type.
constexpr const char *toMLIRString(DimLevelType dlt) {
  switch (dlt) {
  case DimLevelType::Undef:
    return "undef";
  case DimLevelType::Dense:
    return "dense";
  case DimLevelType::Compressed:
    return "compressed";
  case DimLevelType::CompressedNu:
    return "compressed(nonunique)";
  case DimLevelType::CompressedNo:
    return "compressed(nonordered)";
  case DimLevelType::CompressedNuNo:
    return "compressed(nonunique, nonordered)";
  case DimLevelType::Singleton:
    return "singleton";
  case DimLevelType::SingletonNu:
    return "singleton(nonunique)";
  case DimLevelType::SingletonNo:
    return "singleton(nonordered)";
  case DimLevelType::SingletonNuNo:
    return "singleton(nonunique, nonordered)";
  case DimLevelType::LooseCompressed:
    return "loose_compressed";
  case DimLevelType::LooseCompressedNu:
    return "loose_compressed(nonunique)";
  case DimLevelType::LooseCompressedNo:
    return "loose_compressed(nonordered)";
  case DimLevelType::LooseCompressedNuNo:
    return "loose_compressed(nonunique, nonordered)";
  case DimLevelType::TwoOutOfFour:
    return "block2_4";
  }
  return "";
}

/// Check that the `DimLevelType` contains a valid (possibly undefined) value.
constexpr bool isValidDLT(DimLevelType dlt) {
  const uint8_t formatBits = static_cast<uint8_t>(dlt) >> 2;
  const uint8_t propertyBits = static_cast<uint8_t>(dlt) & 3;
  // If undefined or dense, then must be unique and ordered.
  // Otherwise, the format must be one of the known ones.
  return (formatBits <= 1 || formatBits == 16)
             ? (propertyBits == 0)
             : (formatBits == 2 || formatBits == 4 || formatBits == 8);
}

/// Check if the `DimLevelType` is the special undefined value.
constexpr bool isUndefDLT(DimLevelType dlt) {
  return dlt == DimLevelType::Undef;
}

/// Check if the `DimLevelType` is dense.
constexpr bool isDenseDLT(DimLevelType dlt) {
  return dlt == DimLevelType::Dense;
}

/// Check if the `DimLevelType` is 2:4
constexpr bool isTwoOutOfFourDLT(DimLevelType dlt) {
  return dlt == DimLevelType::TwoOutOfFour;
}

// We use the idiom `(dlt & ~3) == format` in order to only return true
// for valid DLTs.  Whereas the `dlt & format` idiom is a bit faster but
// can return false-positives on invalid DLTs.

/// Check if the `DimLevelType` is compressed (regardless of properties).
constexpr bool isCompressedDLT(DimLevelType dlt) {
  return (static_cast<uint8_t>(dlt) & ~3) ==
         static_cast<uint8_t>(DimLevelType::Compressed);
}

/// Check if the `DimLevelType` is compressed (regardless of properties).
constexpr bool isLooseCompressedDLT(DimLevelType dlt) {
  return (static_cast<uint8_t>(dlt) & ~3) ==
         static_cast<uint8_t>(DimLevelType::LooseCompressed);
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

/// Convert a DimLevelType to its corresponding LevelFormat.
/// Returns std::nullopt when input dlt is Undef.
constexpr std::optional<LevelFormat> getLevelFormat(DimLevelType dlt) {
  if (dlt == DimLevelType::Undef)
    return std::nullopt;
  return static_cast<LevelFormat>(static_cast<uint8_t>(dlt) & ~3);
}

/// Convert a LevelFormat to its corresponding DimLevelType with the given
/// properties. Returns std::nullopt when the properties are not applicable for
/// the input level format.
/// TODO: factor out a new LevelProperties type so we can add new properties
/// without changing this function's signature
constexpr std::optional<DimLevelType>
buildLevelType(LevelFormat lf, bool ordered, bool unique) {
  auto dlt = static_cast<DimLevelType>(static_cast<uint8_t>(lf) |
                                       (ordered ? 0 : 2) | (unique ? 0 : 1));
  return isValidDLT(dlt) ? std::optional(dlt) : std::nullopt;
}

/// Ensure the above conversion works as intended.
static_assert(
    (getLevelFormat(DimLevelType::Undef) == std::nullopt &&
     *getLevelFormat(DimLevelType::Dense) == LevelFormat::Dense &&
     *getLevelFormat(DimLevelType::Compressed) == LevelFormat::Compressed &&
     *getLevelFormat(DimLevelType::CompressedNu) == LevelFormat::Compressed &&
     *getLevelFormat(DimLevelType::CompressedNo) == LevelFormat::Compressed &&
     *getLevelFormat(DimLevelType::CompressedNuNo) == LevelFormat::Compressed &&
     *getLevelFormat(DimLevelType::Singleton) == LevelFormat::Singleton &&
     *getLevelFormat(DimLevelType::SingletonNu) == LevelFormat::Singleton &&
     *getLevelFormat(DimLevelType::SingletonNo) == LevelFormat::Singleton &&
     *getLevelFormat(DimLevelType::SingletonNuNo) == LevelFormat::Singleton),
    "getLevelFormat conversion is broken");

static_assert(
    (buildLevelType(LevelFormat::Dense, false, true) == std::nullopt &&
     buildLevelType(LevelFormat::Dense, true, false) == std::nullopt &&
     buildLevelType(LevelFormat::Dense, false, false) == std::nullopt &&
     *buildLevelType(LevelFormat::Dense, true, true) == DimLevelType::Dense &&
     buildLevelType(LevelFormat::TwoOutOfFour, false, true) == std::nullopt &&
     buildLevelType(LevelFormat::TwoOutOfFour, true, false) == std::nullopt &&
     buildLevelType(LevelFormat::TwoOutOfFour, false, false) == std::nullopt &&
     *buildLevelType(LevelFormat::TwoOutOfFour, true, true) ==
         DimLevelType::TwoOutOfFour &&
     *buildLevelType(LevelFormat::Compressed, true, true) ==
         DimLevelType::Compressed &&
     *buildLevelType(LevelFormat::Compressed, true, false) ==
         DimLevelType::CompressedNu &&
     *buildLevelType(LevelFormat::Compressed, false, true) ==
         DimLevelType::CompressedNo &&
     *buildLevelType(LevelFormat::Compressed, false, false) ==
         DimLevelType::CompressedNuNo &&
     *buildLevelType(LevelFormat::Singleton, true, true) ==
         DimLevelType::Singleton &&
     *buildLevelType(LevelFormat::Singleton, true, false) ==
         DimLevelType::SingletonNu &&
     *buildLevelType(LevelFormat::Singleton, false, true) ==
         DimLevelType::SingletonNo &&
     *buildLevelType(LevelFormat::Singleton, false, false) ==
         DimLevelType::SingletonNuNo),
    "buildLevelType conversion is broken");

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
               isValidDLT(DimLevelType::SingletonNuNo) &&
               isValidDLT(DimLevelType::LooseCompressed) &&
               isValidDLT(DimLevelType::LooseCompressedNu) &&
               isValidDLT(DimLevelType::LooseCompressedNo) &&
               isValidDLT(DimLevelType::LooseCompressedNuNo) &&
               isValidDLT(DimLevelType::TwoOutOfFour)),
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

static_assert((!isLooseCompressedDLT(DimLevelType::Dense) &&
               isLooseCompressedDLT(DimLevelType::LooseCompressed) &&
               isLooseCompressedDLT(DimLevelType::LooseCompressedNu) &&
               isLooseCompressedDLT(DimLevelType::LooseCompressedNo) &&
               isLooseCompressedDLT(DimLevelType::LooseCompressedNuNo) &&
               !isLooseCompressedDLT(DimLevelType::Singleton) &&
               !isLooseCompressedDLT(DimLevelType::SingletonNu) &&
               !isLooseCompressedDLT(DimLevelType::SingletonNo) &&
               !isLooseCompressedDLT(DimLevelType::SingletonNuNo)),
              "isLooseCompressedDLT definition is broken");

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
               isOrderedDLT(DimLevelType::TwoOutOfFour) &&
               isOrderedDLT(DimLevelType::Compressed) &&
               isOrderedDLT(DimLevelType::CompressedNu) &&
               !isOrderedDLT(DimLevelType::CompressedNo) &&
               !isOrderedDLT(DimLevelType::CompressedNuNo) &&
               isOrderedDLT(DimLevelType::Singleton) &&
               isOrderedDLT(DimLevelType::SingletonNu) &&
               !isOrderedDLT(DimLevelType::SingletonNo) &&
               !isOrderedDLT(DimLevelType::SingletonNuNo) &&
               isOrderedDLT(DimLevelType::LooseCompressed) &&
               isOrderedDLT(DimLevelType::LooseCompressedNu) &&
               !isOrderedDLT(DimLevelType::LooseCompressedNo) &&
               !isOrderedDLT(DimLevelType::LooseCompressedNuNo)),
              "isOrderedDLT definition is broken");

static_assert((isUniqueDLT(DimLevelType::Dense) &&
               isUniqueDLT(DimLevelType::TwoOutOfFour) &&
               isUniqueDLT(DimLevelType::Compressed) &&
               !isUniqueDLT(DimLevelType::CompressedNu) &&
               isUniqueDLT(DimLevelType::CompressedNo) &&
               !isUniqueDLT(DimLevelType::CompressedNuNo) &&
               isUniqueDLT(DimLevelType::Singleton) &&
               !isUniqueDLT(DimLevelType::SingletonNu) &&
               isUniqueDLT(DimLevelType::SingletonNo) &&
               !isUniqueDLT(DimLevelType::SingletonNuNo) &&
               isUniqueDLT(DimLevelType::LooseCompressed) &&
               !isUniqueDLT(DimLevelType::LooseCompressedNu) &&
               isUniqueDLT(DimLevelType::LooseCompressedNo) &&
               !isUniqueDLT(DimLevelType::LooseCompressedNuNo)),
              "isUniqueDLT definition is broken");

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_IR_ENUMS_H

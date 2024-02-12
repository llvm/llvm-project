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
// calls into the runtime library.  Moveover, the `LevelType` enum
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

#include <cassert>
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
  kEmptyForward = 1,
  kFromCOO = 2,
  kFromReader = 4,
  kToCOO = 5,
  kPack = 7,
  kSortCOOInPlace = 8,
};

/// This enum defines all the sparse representations supportable by
/// the SparseTensor dialect. We use a lightweight encoding to encode
/// the "format" per se (dense, compressed, singleton, loose_compressed,
/// n-out-of-m), the "properties" (ordered, unique) as well as n and m when
/// the format is NOutOfM.
/// The encoding is chosen for performance of the runtime library, and thus may
/// change in future versions; consequently, client code should use the
/// predicate functions defined below, rather than relying on knowledge
/// about the particular binary encoding.
///
/// The `Undef` "format" is a special value used internally for cases
/// where we need to store an undefined or indeterminate `LevelType`.
/// It should not be used externally, since it does not indicate an
/// actual/representable format.
///
/// Bit manipulations for LevelType:
///
/// | 8-bit n | 8-bit m | 16-bit LevelFormat | 16-bit LevelProperty |
///
enum class LevelType : uint64_t {
  Undef = 0x000000000000,
  Dense = 0x000000010000,
  Compressed = 0x000000020000,
  CompressedNu = 0x000000020001,
  CompressedNo = 0x000000020002,
  CompressedNuNo = 0x000000020003,
  Singleton = 0x000000040000,
  SingletonNu = 0x000000040001,
  SingletonNo = 0x000000040002,
  SingletonNuNo = 0x000000040003,
  LooseCompressed = 0x000000080000,
  LooseCompressedNu = 0x000000080001,
  LooseCompressedNo = 0x000000080002,
  LooseCompressedNuNo = 0x000000080003,
  NOutOfM = 0x000000100000,
};

/// This enum defines all supported storage format without the level properties.
enum class LevelFormat : uint64_t {
  Dense = 0x00010000,
  Compressed = 0x00020000,
  Singleton = 0x00040000,
  LooseCompressed = 0x00080000,
  NOutOfM = 0x00100000,
};

/// This enum defines all the nondefault properties for storage formats.
enum class LevelPropertyNondefault : uint64_t {
  Nonunique = 0x0001,
  Nonordered = 0x0002,
};

/// Get N of NOutOfM level type.
constexpr uint64_t getN(LevelType lt) {
  return (static_cast<uint64_t>(lt) >> 32) & 0xff;
}

/// Get M of NOutOfM level type.
constexpr uint64_t getM(LevelType lt) {
  return (static_cast<uint64_t>(lt) >> 40) & 0xff;
}

/// Convert N of NOutOfM level type to the stored bits.
constexpr uint64_t nToBits(uint64_t n) { return n << 32; }

/// Convert M of NOutOfM level type to the stored bits.
constexpr uint64_t mToBits(uint64_t m) { return m << 40; }

/// Check if the `LevelType` is NOutOfM (regardless of
/// properties and block sizes).
constexpr bool isNOutOfMLT(LevelType lt) {
  return ((static_cast<uint64_t>(lt) & 0x100000) ==
          static_cast<uint64_t>(LevelType::NOutOfM));
}

/// Check if the `LevelType` is NOutOfM with the correct block sizes.
constexpr bool isValidNOutOfMLT(LevelType lt, uint64_t n, uint64_t m) {
  return isNOutOfMLT(lt) && getN(lt) == n && getM(lt) == m;
}

/// Returns string representation of the given dimension level type.
constexpr const char *toMLIRString(LevelType lvlType) {
  auto lt = static_cast<LevelType>(static_cast<uint64_t>(lvlType) & 0xffffffff);
  switch (lt) {
  case LevelType::Undef:
    return "undef";
  case LevelType::Dense:
    return "dense";
  case LevelType::Compressed:
    return "compressed";
  case LevelType::CompressedNu:
    return "compressed(nonunique)";
  case LevelType::CompressedNo:
    return "compressed(nonordered)";
  case LevelType::CompressedNuNo:
    return "compressed(nonunique, nonordered)";
  case LevelType::Singleton:
    return "singleton";
  case LevelType::SingletonNu:
    return "singleton(nonunique)";
  case LevelType::SingletonNo:
    return "singleton(nonordered)";
  case LevelType::SingletonNuNo:
    return "singleton(nonunique, nonordered)";
  case LevelType::LooseCompressed:
    return "loose_compressed";
  case LevelType::LooseCompressedNu:
    return "loose_compressed(nonunique)";
  case LevelType::LooseCompressedNo:
    return "loose_compressed(nonordered)";
  case LevelType::LooseCompressedNuNo:
    return "loose_compressed(nonunique, nonordered)";
  case LevelType::NOutOfM:
    return "structured";
  }
  return "";
}

/// Check that the `LevelType` contains a valid (possibly undefined) value.
constexpr bool isValidLT(LevelType lt) {
  const uint64_t formatBits = static_cast<uint64_t>(lt) & 0xffff0000;
  const uint64_t propertyBits = static_cast<uint64_t>(lt) & 0xffff;
  // If undefined/dense/NOutOfM, then must be unique and ordered.
  // Otherwise, the format must be one of the known ones.
  return (formatBits <= 0x10000 || formatBits == 0x100000)
             ? (propertyBits == 0)
             : (formatBits == 0x20000 || formatBits == 0x40000 ||
                formatBits == 0x80000);
}

/// Check if the `LevelType` is the special undefined value.
constexpr bool isUndefLT(LevelType lt) { return lt == LevelType::Undef; }

/// Check if the `LevelType` is dense (regardless of properties).
constexpr bool isDenseLT(LevelType lt) {
  return (static_cast<uint64_t>(lt) & ~0xffff) ==
         static_cast<uint64_t>(LevelType::Dense);
}

/// Check if the `LevelType` is compressed (regardless of properties).
constexpr bool isCompressedLT(LevelType lt) {
  return (static_cast<uint64_t>(lt) & ~0xffff) ==
         static_cast<uint64_t>(LevelType::Compressed);
}

/// Check if the `LevelType` is singleton (regardless of properties).
constexpr bool isSingletonLT(LevelType lt) {
  return (static_cast<uint64_t>(lt) & ~0xffff) ==
         static_cast<uint64_t>(LevelType::Singleton);
}

/// Check if the `LevelType` is loose compressed (regardless of properties).
constexpr bool isLooseCompressedLT(LevelType lt) {
  return (static_cast<uint64_t>(lt) & ~0xffff) ==
         static_cast<uint64_t>(LevelType::LooseCompressed);
}

/// Check if the `LevelType` needs positions array.
constexpr bool isWithPosLT(LevelType lt) {
  return isCompressedLT(lt) || isLooseCompressedLT(lt);
}

/// Check if the `LevelType` needs coordinates array.
constexpr bool isWithCrdLT(LevelType lt) {
  return isCompressedLT(lt) || isSingletonLT(lt) || isLooseCompressedLT(lt) ||
         isNOutOfMLT(lt);
}

/// Check if the `LevelType` is ordered (regardless of storage format).
constexpr bool isOrderedLT(LevelType lt) {
  return !(static_cast<uint64_t>(lt) & 2);
  return !(static_cast<uint64_t>(lt) & 2);
}

/// Check if the `LevelType` is unique (regardless of storage format).
constexpr bool isUniqueLT(LevelType lt) {
  return !(static_cast<uint64_t>(lt) & 1);
  return !(static_cast<uint64_t>(lt) & 1);
}

/// Convert a LevelType to its corresponding LevelFormat.
/// Returns std::nullopt when input lt is Undef.
constexpr std::optional<LevelFormat> getLevelFormat(LevelType lt) {
  if (lt == LevelType::Undef)
    return std::nullopt;
  return static_cast<LevelFormat>(static_cast<uint64_t>(lt) & 0xffff0000);
}

/// Convert a LevelFormat to its corresponding LevelType with the given
/// properties. Returns std::nullopt when the properties are not applicable
/// for the input level format.
constexpr std::optional<LevelType> buildLevelType(LevelFormat lf, bool ordered,
                                                  bool unique, uint64_t n = 0,
                                                  uint64_t m = 0) {
  uint64_t newN = n << 32;
  uint64_t newM = m << 40;
  auto lt =
      static_cast<LevelType>(static_cast<uint64_t>(lf) | (ordered ? 0 : 2) |
                             (unique ? 0 : 1) | newN | newM);
  return isValidLT(lt) ? std::optional(lt) : std::nullopt;
}

//
// Ensure the above methods work as intended.
//

static_assert(
    (getLevelFormat(LevelType::Undef) == std::nullopt &&
     *getLevelFormat(LevelType::Dense) == LevelFormat::Dense &&
     *getLevelFormat(LevelType::Compressed) == LevelFormat::Compressed &&
     *getLevelFormat(LevelType::CompressedNu) == LevelFormat::Compressed &&
     *getLevelFormat(LevelType::CompressedNo) == LevelFormat::Compressed &&
     *getLevelFormat(LevelType::CompressedNuNo) == LevelFormat::Compressed &&
     *getLevelFormat(LevelType::Singleton) == LevelFormat::Singleton &&
     *getLevelFormat(LevelType::SingletonNu) == LevelFormat::Singleton &&
     *getLevelFormat(LevelType::SingletonNo) == LevelFormat::Singleton &&
     *getLevelFormat(LevelType::SingletonNuNo) == LevelFormat::Singleton &&
     *getLevelFormat(LevelType::LooseCompressed) ==
         LevelFormat::LooseCompressed &&
     *getLevelFormat(LevelType::LooseCompressedNu) ==
         LevelFormat::LooseCompressed &&
     *getLevelFormat(LevelType::LooseCompressedNo) ==
         LevelFormat::LooseCompressed &&
     *getLevelFormat(LevelType::LooseCompressedNuNo) ==
         LevelFormat::LooseCompressed &&
     *getLevelFormat(LevelType::NOutOfM) == LevelFormat::NOutOfM),
    "getLevelFormat conversion is broken");

static_assert(
    (buildLevelType(LevelFormat::Dense, false, true) == std::nullopt &&
     buildLevelType(LevelFormat::Dense, true, false) == std::nullopt &&
     buildLevelType(LevelFormat::Dense, false, false) == std::nullopt &&
     *buildLevelType(LevelFormat::Dense, true, true) == LevelType::Dense &&
     *buildLevelType(LevelFormat::Compressed, true, true) ==
         LevelType::Compressed &&
     *buildLevelType(LevelFormat::Compressed, true, false) ==
         LevelType::CompressedNu &&
     *buildLevelType(LevelFormat::Compressed, false, true) ==
         LevelType::CompressedNo &&
     *buildLevelType(LevelFormat::Compressed, false, false) ==
         LevelType::CompressedNuNo &&
     *buildLevelType(LevelFormat::Singleton, true, true) ==
         LevelType::Singleton &&
     *buildLevelType(LevelFormat::Singleton, true, false) ==
         LevelType::SingletonNu &&
     *buildLevelType(LevelFormat::Singleton, false, true) ==
         LevelType::SingletonNo &&
     *buildLevelType(LevelFormat::Singleton, false, false) ==
         LevelType::SingletonNuNo &&
     *buildLevelType(LevelFormat::LooseCompressed, true, true) ==
         LevelType::LooseCompressed &&
     *buildLevelType(LevelFormat::LooseCompressed, true, false) ==
         LevelType::LooseCompressedNu &&
     *buildLevelType(LevelFormat::LooseCompressed, false, true) ==
         LevelType::LooseCompressedNo &&
     *buildLevelType(LevelFormat::LooseCompressed, false, false) ==
         LevelType::LooseCompressedNuNo &&
     buildLevelType(LevelFormat::NOutOfM, false, true) == std::nullopt &&
     buildLevelType(LevelFormat::NOutOfM, true, false) == std::nullopt &&
     buildLevelType(LevelFormat::NOutOfM, false, false) == std::nullopt &&
     *buildLevelType(LevelFormat::NOutOfM, true, true) == LevelType::NOutOfM),
    "buildLevelType conversion is broken");

static_assert(
    (getN(*buildLevelType(LevelFormat::NOutOfM, true, true, 2, 4)) == 2 &&
     getM(*buildLevelType(LevelFormat::NOutOfM, true, true, 2, 4)) == 4 &&
     getN(*buildLevelType(LevelFormat::NOutOfM, true, true, 8, 10)) == 8 &&
     getM(*buildLevelType(LevelFormat::NOutOfM, true, true, 8, 10)) == 10),
    "getN/M conversion is broken");

static_assert(
    (isValidNOutOfMLT(*buildLevelType(LevelFormat::NOutOfM, true, true, 2, 4),
                      2, 4) &&
     isValidNOutOfMLT(*buildLevelType(LevelFormat::NOutOfM, true, true, 8, 10),
                      8, 10) &&
     !isValidNOutOfMLT(*buildLevelType(LevelFormat::NOutOfM, true, true, 3, 4),
                       2, 4)),
    "isValidNOutOfMLT definition is broken");

static_assert(
    (isValidLT(LevelType::Undef) && isValidLT(LevelType::Dense) &&
     isValidLT(LevelType::Compressed) && isValidLT(LevelType::CompressedNu) &&
     isValidLT(LevelType::CompressedNo) &&
     isValidLT(LevelType::CompressedNuNo) && isValidLT(LevelType::Singleton) &&
     isValidLT(LevelType::SingletonNu) && isValidLT(LevelType::SingletonNo) &&
     isValidLT(LevelType::SingletonNuNo) &&
     isValidLT(LevelType::LooseCompressed) &&
     isValidLT(LevelType::LooseCompressedNu) &&
     isValidLT(LevelType::LooseCompressedNo) &&
     isValidLT(LevelType::LooseCompressedNuNo) &&
     isValidLT(LevelType::NOutOfM)),
    "isValidLT definition is broken");

static_assert((isDenseLT(LevelType::Dense) &&
               !isDenseLT(LevelType::Compressed) &&
               !isDenseLT(LevelType::CompressedNu) &&
               !isDenseLT(LevelType::CompressedNo) &&
               !isDenseLT(LevelType::CompressedNuNo) &&
               !isDenseLT(LevelType::Singleton) &&
               !isDenseLT(LevelType::SingletonNu) &&
               !isDenseLT(LevelType::SingletonNo) &&
               !isDenseLT(LevelType::SingletonNuNo) &&
               !isDenseLT(LevelType::LooseCompressed) &&
               !isDenseLT(LevelType::LooseCompressedNu) &&
               !isDenseLT(LevelType::LooseCompressedNo) &&
               !isDenseLT(LevelType::LooseCompressedNuNo) &&
               !isDenseLT(LevelType::NOutOfM)),
              "isDenseLT definition is broken");

static_assert((!isCompressedLT(LevelType::Dense) &&
               isCompressedLT(LevelType::Compressed) &&
               isCompressedLT(LevelType::CompressedNu) &&
               isCompressedLT(LevelType::CompressedNo) &&
               isCompressedLT(LevelType::CompressedNuNo) &&
               !isCompressedLT(LevelType::Singleton) &&
               !isCompressedLT(LevelType::SingletonNu) &&
               !isCompressedLT(LevelType::SingletonNo) &&
               !isCompressedLT(LevelType::SingletonNuNo) &&
               !isCompressedLT(LevelType::LooseCompressed) &&
               !isCompressedLT(LevelType::LooseCompressedNu) &&
               !isCompressedLT(LevelType::LooseCompressedNo) &&
               !isCompressedLT(LevelType::LooseCompressedNuNo) &&
               !isCompressedLT(LevelType::NOutOfM)),
              "isCompressedLT definition is broken");

static_assert((!isSingletonLT(LevelType::Dense) &&
               !isSingletonLT(LevelType::Compressed) &&
               !isSingletonLT(LevelType::CompressedNu) &&
               !isSingletonLT(LevelType::CompressedNo) &&
               !isSingletonLT(LevelType::CompressedNuNo) &&
               isSingletonLT(LevelType::Singleton) &&
               isSingletonLT(LevelType::SingletonNu) &&
               isSingletonLT(LevelType::SingletonNo) &&
               isSingletonLT(LevelType::SingletonNuNo) &&
               !isSingletonLT(LevelType::LooseCompressed) &&
               !isSingletonLT(LevelType::LooseCompressedNu) &&
               !isSingletonLT(LevelType::LooseCompressedNo) &&
               !isSingletonLT(LevelType::LooseCompressedNuNo) &&
               !isSingletonLT(LevelType::NOutOfM)),
              "isSingletonLT definition is broken");

static_assert((!isLooseCompressedLT(LevelType::Dense) &&
               !isLooseCompressedLT(LevelType::Compressed) &&
               !isLooseCompressedLT(LevelType::CompressedNu) &&
               !isLooseCompressedLT(LevelType::CompressedNo) &&
               !isLooseCompressedLT(LevelType::CompressedNuNo) &&
               !isLooseCompressedLT(LevelType::Singleton) &&
               !isLooseCompressedLT(LevelType::SingletonNu) &&
               !isLooseCompressedLT(LevelType::SingletonNo) &&
               !isLooseCompressedLT(LevelType::SingletonNuNo) &&
               isLooseCompressedLT(LevelType::LooseCompressed) &&
               isLooseCompressedLT(LevelType::LooseCompressedNu) &&
               isLooseCompressedLT(LevelType::LooseCompressedNo) &&
               isLooseCompressedLT(LevelType::LooseCompressedNuNo) &&
               !isLooseCompressedLT(LevelType::NOutOfM)),
              "isLooseCompressedLT definition is broken");

static_assert((!isNOutOfMLT(LevelType::Dense) &&
               !isNOutOfMLT(LevelType::Compressed) &&
               !isNOutOfMLT(LevelType::CompressedNu) &&
               !isNOutOfMLT(LevelType::CompressedNo) &&
               !isNOutOfMLT(LevelType::CompressedNuNo) &&
               !isNOutOfMLT(LevelType::Singleton) &&
               !isNOutOfMLT(LevelType::SingletonNu) &&
               !isNOutOfMLT(LevelType::SingletonNo) &&
               !isNOutOfMLT(LevelType::SingletonNuNo) &&
               !isNOutOfMLT(LevelType::LooseCompressed) &&
               !isNOutOfMLT(LevelType::LooseCompressedNu) &&
               !isNOutOfMLT(LevelType::LooseCompressedNo) &&
               !isNOutOfMLT(LevelType::LooseCompressedNuNo) &&
               isNOutOfMLT(LevelType::NOutOfM)),
              "isNOutOfMLT definition is broken");

static_assert((isOrderedLT(LevelType::Dense) &&
               isOrderedLT(LevelType::Compressed) &&
               isOrderedLT(LevelType::CompressedNu) &&
               !isOrderedLT(LevelType::CompressedNo) &&
               !isOrderedLT(LevelType::CompressedNuNo) &&
               isOrderedLT(LevelType::Singleton) &&
               isOrderedLT(LevelType::SingletonNu) &&
               !isOrderedLT(LevelType::SingletonNo) &&
               !isOrderedLT(LevelType::SingletonNuNo) &&
               isOrderedLT(LevelType::LooseCompressed) &&
               isOrderedLT(LevelType::LooseCompressedNu) &&
               !isOrderedLT(LevelType::LooseCompressedNo) &&
               !isOrderedLT(LevelType::LooseCompressedNuNo) &&
               isOrderedLT(LevelType::NOutOfM)),
              "isOrderedLT definition is broken");

static_assert((isUniqueLT(LevelType::Dense) &&
               isUniqueLT(LevelType::Compressed) &&
               !isUniqueLT(LevelType::CompressedNu) &&
               isUniqueLT(LevelType::CompressedNo) &&
               !isUniqueLT(LevelType::CompressedNuNo) &&
               isUniqueLT(LevelType::Singleton) &&
               !isUniqueLT(LevelType::SingletonNu) &&
               isUniqueLT(LevelType::SingletonNo) &&
               !isUniqueLT(LevelType::SingletonNuNo) &&
               isUniqueLT(LevelType::LooseCompressed) &&
               !isUniqueLT(LevelType::LooseCompressedNu) &&
               isUniqueLT(LevelType::LooseCompressedNo) &&
               !isUniqueLT(LevelType::LooseCompressedNuNo) &&
               isUniqueLT(LevelType::NOutOfM)),
              "isUniqueLT definition is broken");

/// Bit manipulations for affine encoding.
///
/// Note that because the indices in the mappings refer to dimensions
/// and levels (and *not* the sizes of these dimensions and levels), the
/// 64-bit encoding gives ample room for a compact encoding of affine
/// operations in the higher bits. Pure permutations still allow for
/// 60-bit indices. But non-permutations reserve 20-bits for the
/// potential three components (index i, constant, index ii).
///
/// The compact encoding is as follows:
///
///  0xffffffffffffffff
/// |0000      |                        60-bit idx| e.g. i
/// |0001 floor|           20-bit const|20-bit idx| e.g. i floor c
/// |0010 mod  |           20-bit const|20-bit idx| e.g. i mod c
/// |0011 mul  |20-bit idx|20-bit const|20-bit idx| e.g. i + c * ii
///
/// This encoding provides sufficient generality for currently supported
/// sparse tensor types. To generalize this more, we will need to provide
/// a broader encoding scheme for affine functions. Also, the library
/// encoding may be replaced with pure "direct-IR" code in the future.
///
constexpr uint64_t encodeDim(uint64_t i, uint64_t cf, uint64_t cm) {
  if (cf != 0) {
    assert(cf <= 0xfffffu && cm == 0 && i <= 0xfffffu);
    return (static_cast<uint64_t>(0x01u) << 60) | (cf << 20) | i;
  }
  if (cm != 0) {
    assert(cm <= 0xfffffu && i <= 0xfffffu);
    return (static_cast<uint64_t>(0x02u) << 60) | (cm << 20) | i;
  }
  assert(i <= 0x0fffffffffffffffu);
  return i;
}
constexpr uint64_t encodeLvl(uint64_t i, uint64_t c, uint64_t ii) {
  if (c != 0) {
    assert(c <= 0xfffffu && ii <= 0xfffffu && i <= 0xfffffu);
    return (static_cast<uint64_t>(0x03u) << 60) | (c << 20) | (ii << 40) | i;
  }
  assert(i <= 0x0fffffffffffffffu);
  return i;
}
constexpr bool isEncodedFloor(uint64_t v) { return (v >> 60) == 0x01u; }
constexpr bool isEncodedMod(uint64_t v) { return (v >> 60) == 0x02u; }
constexpr bool isEncodedMul(uint64_t v) { return (v >> 60) == 0x03u; }
constexpr uint64_t decodeIndex(uint64_t v) { return v & 0xfffffu; }
constexpr uint64_t decodeConst(uint64_t v) { return (v >> 20) & 0xfffffu; }
constexpr uint64_t decodeMulc(uint64_t v) { return (v >> 20) & 0xfffffu; }
constexpr uint64_t decodeMuli(uint64_t v) { return (v >> 40) & 0xfffffu; }

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_IR_ENUMS_H

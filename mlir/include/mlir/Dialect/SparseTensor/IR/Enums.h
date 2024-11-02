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
/// both the "format" per se (dense, compressed, singleton, loose_compressed,
/// two-out-of-four) as well as the "properties" (ordered, unique). The
/// encoding is chosen for performance of the runtime library, and thus may
/// change in future versions; consequently, client code should use the
/// predicate functions defined below, rather than relying on knowledge
/// about the particular binary encoding.
///
/// The `Undef` "format" is a special value used internally for cases
/// where we need to store an undefined or indeterminate `LevelType`.
/// It should not be used externally, since it does not indicate an
/// actual/representable format.
enum class LevelType : uint64_t {
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
enum class LevelFormat : uint64_t {
  Dense = 4,            // 0b00001_00
  Compressed = 8,       // 0b00010_00
  Singleton = 16,       // 0b00100_00
  LooseCompressed = 32, // 0b01000_00
  TwoOutOfFour = 64,    // 0b10000_00
};

/// This enum defines all the nondefault properties for storage formats.
enum class LevelPropertyNondefault : uint64_t {
  Nonunique = 1,  // 0b00000_01
  Nonordered = 2, // 0b00000_10
};

/// Returns string representation of the given dimension level type.
constexpr const char *toMLIRString(LevelType lt) {
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
  case LevelType::TwoOutOfFour:
    return "block2_4";
  }
  return "";
}

/// Check that the `LevelType` contains a valid (possibly undefined) value.
constexpr bool isValidLT(LevelType lt) {
  const uint64_t formatBits = static_cast<uint64_t>(lt) >> 2;
  const uint64_t propertyBits = static_cast<uint64_t>(lt) & 3;
  // If undefined or dense, then must be unique and ordered.
  // Otherwise, the format must be one of the known ones.
  return (formatBits <= 1 || formatBits == 16)
             ? (propertyBits == 0)
             : (formatBits == 2 || formatBits == 4 || formatBits == 8);
}

/// Check if the `LevelType` is the special undefined value.
constexpr bool isUndefLT(LevelType lt) { return lt == LevelType::Undef; }

/// Check if the `LevelType` is dense (regardless of properties).
constexpr bool isDenseLT(LevelType lt) {
  return (static_cast<uint64_t>(lt) & ~3) ==
         static_cast<uint64_t>(LevelType::Dense);
}

/// Check if the `LevelType` is compressed (regardless of properties).
constexpr bool isCompressedLT(LevelType lt) {
  return (static_cast<uint64_t>(lt) & ~3) ==
         static_cast<uint64_t>(LevelType::Compressed);
}

/// Check if the `LevelType` is singleton (regardless of properties).
constexpr bool isSingletonLT(LevelType lt) {
  return (static_cast<uint64_t>(lt) & ~3) ==
         static_cast<uint64_t>(LevelType::Singleton);
}

/// Check if the `LevelType` is loose compressed (regardless of properties).
constexpr bool isLooseCompressedLT(LevelType lt) {
  return (static_cast<uint64_t>(lt) & ~3) ==
         static_cast<uint64_t>(LevelType::LooseCompressed);
}

/// Check if the `LevelType` is 2OutOf4 (regardless of properties).
constexpr bool is2OutOf4LT(LevelType lt) {
  return (static_cast<uint64_t>(lt) & ~3) ==
         static_cast<uint64_t>(LevelType::TwoOutOfFour);
}

/// Check if the `LevelType` needs positions array.
constexpr bool isWithPosLT(LevelType lt) {
  return isCompressedLT(lt) || isLooseCompressedLT(lt);
}

/// Check if the `LevelType` needs coordinates array.
constexpr bool isWithCrdLT(LevelType lt) {
  return isCompressedLT(lt) || isSingletonLT(lt) || isLooseCompressedLT(lt) ||
         is2OutOf4LT(lt);
}

/// Check if the `LevelType` is ordered (regardless of storage format).
constexpr bool isOrderedLT(LevelType lt) {
  return !(static_cast<uint64_t>(lt) & 2);
}

/// Check if the `LevelType` is unique (regardless of storage format).
constexpr bool isUniqueLT(LevelType lt) {
  return !(static_cast<uint64_t>(lt) & 1);
}

/// Convert a LevelType to its corresponding LevelFormat.
/// Returns std::nullopt when input lt is Undef.
constexpr std::optional<LevelFormat> getLevelFormat(LevelType lt) {
  if (lt == LevelType::Undef)
    return std::nullopt;
  return static_cast<LevelFormat>(static_cast<uint64_t>(lt) & ~3);
}

/// Convert a LevelFormat to its corresponding LevelType with the given
/// properties. Returns std::nullopt when the properties are not applicable
/// for the input level format.
constexpr std::optional<LevelType> buildLevelType(LevelFormat lf, bool ordered,
                                                  bool unique) {
  auto lt = static_cast<LevelType>(static_cast<uint64_t>(lf) |
                                   (ordered ? 0 : 2) | (unique ? 0 : 1));
  return isValidLT(lt) ? std::optional(lt) : std::nullopt;
}

//
// Ensure the above methods work as indended.
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
     *getLevelFormat(LevelType::TwoOutOfFour) == LevelFormat::TwoOutOfFour),
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
     buildLevelType(LevelFormat::TwoOutOfFour, false, true) == std::nullopt &&
     buildLevelType(LevelFormat::TwoOutOfFour, true, false) == std::nullopt &&
     buildLevelType(LevelFormat::TwoOutOfFour, false, false) == std::nullopt &&
     *buildLevelType(LevelFormat::TwoOutOfFour, true, true) ==
         LevelType::TwoOutOfFour),
    "buildLevelType conversion is broken");

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
     isValidLT(LevelType::TwoOutOfFour)),
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
               !isDenseLT(LevelType::TwoOutOfFour)),
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
               !isCompressedLT(LevelType::TwoOutOfFour)),
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
               !isSingletonLT(LevelType::TwoOutOfFour)),
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
               !isLooseCompressedLT(LevelType::TwoOutOfFour)),
              "isLooseCompressedLT definition is broken");

static_assert((!is2OutOf4LT(LevelType::Dense) &&
               !is2OutOf4LT(LevelType::Compressed) &&
               !is2OutOf4LT(LevelType::CompressedNu) &&
               !is2OutOf4LT(LevelType::CompressedNo) &&
               !is2OutOf4LT(LevelType::CompressedNuNo) &&
               !is2OutOf4LT(LevelType::Singleton) &&
               !is2OutOf4LT(LevelType::SingletonNu) &&
               !is2OutOf4LT(LevelType::SingletonNo) &&
               !is2OutOf4LT(LevelType::SingletonNuNo) &&
               !is2OutOf4LT(LevelType::LooseCompressed) &&
               !is2OutOf4LT(LevelType::LooseCompressedNu) &&
               !is2OutOf4LT(LevelType::LooseCompressedNo) &&
               !is2OutOf4LT(LevelType::LooseCompressedNuNo) &&
               is2OutOf4LT(LevelType::TwoOutOfFour)),
              "is2OutOf4LT definition is broken");

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
               isOrderedLT(LevelType::TwoOutOfFour)),
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
               isUniqueLT(LevelType::TwoOutOfFour)),
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

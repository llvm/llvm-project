//===- MachineValueType.h - Machine-Level types -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the set of machine-level target independent types which
// legal values in the code generator use.
//
// Constants and properties are defined in ValueTypes.td.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TMP_MACHINEVALUETYPE_H
#define LLVM_TMP_MACHINEVALUETYPE_H

#include "llvm/ADT/Sequence.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TypeSize.h"
#include <cassert>
#include <cstdint>

namespace llvm::tmp {

  class Type;
  class raw_ostream;

  /// Machine Value Type. Every type that is supported natively by some
  /// processor targeted by LLVM occurs here. This means that any legal value
  /// type can be represented by an MVT.
  class MVT {
  public:
    enum SimpleValueType : uint8_t {
      // Simple value types that aren't explicitly part of this enumeration
      // are considered extended value types.
      INVALID_SIMPLE_VALUE_TYPE = 0,

#define GET_VT_ATTR(Ty, n, sz) Ty = n,
#define GET_VT_RANGES
#include "GenVT.inc"
#undef GET_VT_ATTR
#undef GET_VT_RANGES

      FIRST_INTEGER_VALUETYPE = i1,
      LAST_INTEGER_VALUETYPE  = i128,

      FIRST_FP_VALUETYPE = bf16,
      LAST_FP_VALUETYPE  = ppcf128,

      FIRST_INTEGER_FIXEDLEN_VECTOR_VALUETYPE = v1i1,
      LAST_INTEGER_FIXEDLEN_VECTOR_VALUETYPE = v1i128,

      FIRST_FP_FIXEDLEN_VECTOR_VALUETYPE = v1f16,
      LAST_FP_FIXEDLEN_VECTOR_VALUETYPE = v256f64,

      FIRST_FIXEDLEN_VECTOR_VALUETYPE = v1i1,
      LAST_FIXEDLEN_VECTOR_VALUETYPE = v256f64,

      FIRST_INTEGER_SCALABLE_VECTOR_VALUETYPE = nxv1i1,
      LAST_INTEGER_SCALABLE_VECTOR_VALUETYPE = nxv32i64,

      FIRST_FP_SCALABLE_VECTOR_VALUETYPE = nxv1f16,
      LAST_FP_SCALABLE_VECTOR_VALUETYPE = nxv8f64,

      FIRST_SCALABLE_VECTOR_VALUETYPE = nxv1i1,
      LAST_SCALABLE_VECTOR_VALUETYPE = nxv8f64,

      FIRST_VECTOR_VALUETYPE = v1i1,
      LAST_VECTOR_VALUETYPE  = nxv8f64,

      VALUETYPE_SIZE = LAST_VALUETYPE + 1,

      // This is the current maximum for LAST_VALUETYPE.
      // MVT::MAX_ALLOWED_VALUETYPE is used for asserts and to size bit vectors
      // This value must be a multiple of 32.
      MAX_ALLOWED_VALUETYPE = 224,
    };

    static_assert(FIRST_VALUETYPE > 0);
    static_assert(LAST_VALUETYPE < MAX_ALLOWED_VALUETYPE);

    SimpleValueType SimpleTy = INVALID_SIMPLE_VALUE_TYPE;

    constexpr MVT() = default;
    constexpr MVT(SimpleValueType SVT) : SimpleTy(SVT) {}

    bool operator>(const MVT& S)  const { return SimpleTy >  S.SimpleTy; }
    bool operator<(const MVT& S)  const { return SimpleTy <  S.SimpleTy; }
    bool operator==(const MVT& S) const { return SimpleTy == S.SimpleTy; }
    bool operator!=(const MVT& S) const { return SimpleTy != S.SimpleTy; }
    bool operator>=(const MVT& S) const { return SimpleTy >= S.SimpleTy; }
    bool operator<=(const MVT& S) const { return SimpleTy <= S.SimpleTy; }

    /// Support for debugging, callable in GDB: VT.dump()
    void dump() const;

    /// Implement operator<<.
    void print(raw_ostream &OS) const;

    /// Return true if this is a valid simple valuetype.
    bool isValid() const {
      return (SimpleTy >= MVT::FIRST_VALUETYPE &&
              SimpleTy <= MVT::LAST_VALUETYPE);
    }

    /// Return true if this is a FP or a vector FP type.
    bool isFloatingPoint() const {
      return ((SimpleTy >= MVT::FIRST_FP_VALUETYPE &&
               SimpleTy <= MVT::LAST_FP_VALUETYPE) ||
              (SimpleTy >= MVT::FIRST_FP_FIXEDLEN_VECTOR_VALUETYPE &&
               SimpleTy <= MVT::LAST_FP_FIXEDLEN_VECTOR_VALUETYPE) ||
              (SimpleTy >= MVT::FIRST_FP_SCALABLE_VECTOR_VALUETYPE &&
               SimpleTy <= MVT::LAST_FP_SCALABLE_VECTOR_VALUETYPE));
    }

    /// Return true if this is an integer or a vector integer type.
    bool isInteger() const {
      return ((SimpleTy >= MVT::FIRST_INTEGER_VALUETYPE &&
               SimpleTy <= MVT::LAST_INTEGER_VALUETYPE) ||
              (SimpleTy >= MVT::FIRST_INTEGER_FIXEDLEN_VECTOR_VALUETYPE &&
               SimpleTy <= MVT::LAST_INTEGER_FIXEDLEN_VECTOR_VALUETYPE) ||
              (SimpleTy >= MVT::FIRST_INTEGER_SCALABLE_VECTOR_VALUETYPE &&
               SimpleTy <= MVT::LAST_INTEGER_SCALABLE_VECTOR_VALUETYPE));
    }

    /// Return true if this is an integer, not including vectors.
    bool isScalarInteger() const {
      return (SimpleTy >= MVT::FIRST_INTEGER_VALUETYPE &&
              SimpleTy <= MVT::LAST_INTEGER_VALUETYPE);
    }

    /// Return true if this is a vector value type.
    bool isVector() const {
      return (SimpleTy >= MVT::FIRST_VECTOR_VALUETYPE &&
              SimpleTy <= MVT::LAST_VECTOR_VALUETYPE);
    }

    /// Return true if this is a vector value type where the
    /// runtime length is machine dependent
    bool isScalableVector() const {
      return (SimpleTy >= MVT::FIRST_SCALABLE_VECTOR_VALUETYPE &&
              SimpleTy <= MVT::LAST_SCALABLE_VECTOR_VALUETYPE);
    }

    /// Return true if this is a custom target type that has a scalable size.
    bool isScalableTargetExtVT() const {
      return SimpleTy == MVT::aarch64svcount;
    }

    /// Return true if the type is a scalable type.
    bool isScalableVT() const {
      return isScalableVector() || isScalableTargetExtVT();
    }

    bool isFixedLengthVector() const {
      return (SimpleTy >= MVT::FIRST_FIXEDLEN_VECTOR_VALUETYPE &&
              SimpleTy <= MVT::LAST_FIXEDLEN_VECTOR_VALUETYPE);
    }

    /// Return true if this is a 16-bit vector type.
    bool is16BitVector() const {
      return (isFixedLengthVector() && getFixedSizeInBits() == 16);
    }

    /// Return true if this is a 32-bit vector type.
    bool is32BitVector() const {
      return (isFixedLengthVector() && getFixedSizeInBits() == 32);
    }

    /// Return true if this is a 64-bit vector type.
    bool is64BitVector() const {
      return (isFixedLengthVector() && getFixedSizeInBits() == 64);
    }

    /// Return true if this is a 128-bit vector type.
    bool is128BitVector() const {
      return (isFixedLengthVector() && getFixedSizeInBits() == 128);
    }

    /// Return true if this is a 256-bit vector type.
    bool is256BitVector() const {
      return (isFixedLengthVector() && getFixedSizeInBits() == 256);
    }

    /// Return true if this is a 512-bit vector type.
    bool is512BitVector() const {
      return (isFixedLengthVector() && getFixedSizeInBits() == 512);
    }

    /// Return true if this is a 1024-bit vector type.
    bool is1024BitVector() const {
      return (isFixedLengthVector() && getFixedSizeInBits() == 1024);
    }

    /// Return true if this is a 2048-bit vector type.
    bool is2048BitVector() const {
      return (isFixedLengthVector() && getFixedSizeInBits() == 2048);
    }

    /// Return true if this is an overloaded type for TableGen.
    bool isOverloaded() const {
      return (SimpleTy == MVT::Any || SimpleTy == MVT::iAny ||
              SimpleTy == MVT::fAny || SimpleTy == MVT::vAny ||
              SimpleTy == MVT::iPTRAny);
    }

    /// Return a vector with the same number of elements as this vector, but
    /// with the element type converted to an integer type with the same
    /// bitwidth.
    MVT changeVectorElementTypeToInteger() const {
      MVT EltTy = getVectorElementType();
      MVT IntTy = MVT::getIntegerVT(EltTy.getSizeInBits());
      MVT VecTy = MVT::getVectorVT(IntTy, getVectorElementCount());
      assert(VecTy.SimpleTy != MVT::INVALID_SIMPLE_VALUE_TYPE &&
             "Simple vector VT not representable by simple integer vector VT!");
      return VecTy;
    }

    /// Return a VT for a vector type whose attributes match ourselves
    /// with the exception of the element type that is chosen by the caller.
    MVT changeVectorElementType(MVT EltVT) const {
      MVT VecTy = MVT::getVectorVT(EltVT, getVectorElementCount());
      assert(VecTy.SimpleTy != MVT::INVALID_SIMPLE_VALUE_TYPE &&
             "Simple vector VT not representable by simple integer vector VT!");
      return VecTy;
    }

    /// Return the type converted to an equivalently sized integer or vector
    /// with integer element type. Similar to changeVectorElementTypeToInteger,
    /// but also handles scalars.
    MVT changeTypeToInteger() {
      if (isVector())
        return changeVectorElementTypeToInteger();
      return MVT::getIntegerVT(getSizeInBits());
    }

    /// Return a VT for a vector type with the same element type but
    /// half the number of elements.
    MVT getHalfNumVectorElementsVT() const {
      MVT EltVT = getVectorElementType();
      auto EltCnt = getVectorElementCount();
      assert(EltCnt.isKnownEven() && "Splitting vector, but not in half!");
      return getVectorVT(EltVT, EltCnt.divideCoefficientBy(2));
    }

    /// Returns true if the given vector is a power of 2.
    bool isPow2VectorType() const {
      unsigned NElts = getVectorMinNumElements();
      return !(NElts & (NElts - 1));
    }

    /// Widens the length of the given vector MVT up to the nearest power of 2
    /// and returns that type.
    MVT getPow2VectorType() const {
      if (isPow2VectorType())
        return *this;

      ElementCount NElts = getVectorElementCount();
      unsigned NewMinCount = 1 << Log2_32_Ceil(NElts.getKnownMinValue());
      NElts = ElementCount::get(NewMinCount, NElts.isScalable());
      return MVT::getVectorVT(getVectorElementType(), NElts);
    }

    /// If this is a vector, return the element type, otherwise return this.
    MVT getScalarType() const {
      return isVector() ? getVectorElementType() : *this;
    }

    MVT getVectorElementType() const {
      // clang-format off
      switch (SimpleTy) {
      default:
        llvm_unreachable("Not a vector MVT!");
      case v1i1:
      case v2i1:
      case v4i1:
      case v8i1:
      case v16i1:
      case v32i1:
      case v64i1:
      case v128i1:
      case v256i1:
      case v512i1:
      case v1024i1:
      case v2048i1:
      case nxv1i1:
      case nxv2i1:
      case nxv4i1:
      case nxv8i1:
      case nxv16i1:
      case nxv32i1:
      case nxv64i1: return i1;
      case v128i2:
      case v256i2: return i2;
      case v64i4:
      case v128i4: return i4;
      case v1i8:
      case v2i8:
      case v4i8:
      case v8i8:
      case v16i8:
      case v32i8:
      case v64i8:
      case v128i8:
      case v256i8:
      case v512i8:
      case v1024i8:
      case nxv1i8:
      case nxv2i8:
      case nxv4i8:
      case nxv8i8:
      case nxv16i8:
      case nxv32i8:
      case nxv64i8: return i8;
      case v1i16:
      case v2i16:
      case v3i16:
      case v4i16:
      case v8i16:
      case v16i16:
      case v32i16:
      case v64i16:
      case v128i16:
      case v256i16:
      case v512i16:
      case nxv1i16:
      case nxv2i16:
      case nxv4i16:
      case nxv8i16:
      case nxv16i16:
      case nxv32i16: return i16;
      case v1i32:
      case v2i32:
      case v3i32:
      case v4i32:
      case v5i32:
      case v6i32:
      case v7i32:
      case v8i32:
      case v9i32:
      case v10i32:
      case v11i32:
      case v12i32:
      case v16i32:
      case v32i32:
      case v64i32:
      case v128i32:
      case v256i32:
      case v512i32:
      case v1024i32:
      case v2048i32:
      case nxv1i32:
      case nxv2i32:
      case nxv4i32:
      case nxv8i32:
      case nxv16i32:
      case nxv32i32: return i32;
      case v1i64:
      case v2i64:
      case v3i64:
      case v4i64:
      case v8i64:
      case v16i64:
      case v32i64:
      case v64i64:
      case v128i64:
      case v256i64:
      case nxv1i64:
      case nxv2i64:
      case nxv4i64:
      case nxv8i64:
      case nxv16i64:
      case nxv32i64: return i64;
      case v1i128: return i128;
      case v1f16:
      case v2f16:
      case v3f16:
      case v4f16:
      case v8f16:
      case v16f16:
      case v32f16:
      case v64f16:
      case v128f16:
      case v256f16:
      case v512f16:
      case nxv1f16:
      case nxv2f16:
      case nxv4f16:
      case nxv8f16:
      case nxv16f16:
      case nxv32f16: return f16;
      case v2bf16:
      case v3bf16:
      case v4bf16:
      case v8bf16:
      case v16bf16:
      case v32bf16:
      case v64bf16:
      case v128bf16:
      case nxv1bf16:
      case nxv2bf16:
      case nxv4bf16:
      case nxv8bf16:
      case nxv16bf16:
      case nxv32bf16: return bf16;
      case v1f32:
      case v2f32:
      case v3f32:
      case v4f32:
      case v5f32:
      case v6f32:
      case v7f32:
      case v8f32:
      case v9f32:
      case v10f32:
      case v11f32:
      case v12f32:
      case v16f32:
      case v32f32:
      case v64f32:
      case v128f32:
      case v256f32:
      case v512f32:
      case v1024f32:
      case v2048f32:
      case nxv1f32:
      case nxv2f32:
      case nxv4f32:
      case nxv8f32:
      case nxv16f32: return f32;
      case v1f64:
      case v2f64:
      case v3f64:
      case v4f64:
      case v8f64:
      case v16f64:
      case v32f64:
      case v64f64:
      case v128f64:
      case v256f64:
      case nxv1f64:
      case nxv2f64:
      case nxv4f64:
      case nxv8f64: return f64;
      }
      // clang-format on
    }

    /// Given a vector type, return the minimum number of elements it contains.
    unsigned getVectorMinNumElements() const {
      switch (SimpleTy) {
      default:
        llvm_unreachable("Not a vector MVT!");
      case v2048i1:
      case v2048i32:
      case v2048f32: return 2048;
      case v1024i1:
      case v1024i8:
      case v1024i32:
      case v1024f32: return 1024;
      case v512i1:
      case v512i8:
      case v512i16:
      case v512i32:
      case v512f16:
      case v512f32: return 512;
      case v256i1:
      case v256i2:
      case v256i8:
      case v256i16:
      case v256f16:
      case v256i32:
      case v256i64:
      case v256f32:
      case v256f64: return 256;
      case v128i1:
      case v128i2:
      case v128i4:
      case v128i8:
      case v128i16:
      case v128i32:
      case v128i64:
      case v128f16:
      case v128bf16:
      case v128f32:
      case v128f64: return 128;
      case v64i1:
      case v64i4:
      case v64i8:
      case v64i16:
      case v64i32:
      case v64i64:
      case v64f16:
      case v64bf16:
      case v64f32:
      case v64f64:
      case nxv64i1:
      case nxv64i8: return 64;
      case v32i1:
      case v32i8:
      case v32i16:
      case v32i32:
      case v32i64:
      case v32f16:
      case v32bf16:
      case v32f32:
      case v32f64:
      case nxv32i1:
      case nxv32i8:
      case nxv32i16:
      case nxv32i32:
      case nxv32i64:
      case nxv32f16:
      case nxv32bf16: return 32;
      case v16i1:
      case v16i8:
      case v16i16:
      case v16i32:
      case v16i64:
      case v16f16:
      case v16bf16:
      case v16f32:
      case v16f64:
      case nxv16i1:
      case nxv16i8:
      case nxv16i16:
      case nxv16i32:
      case nxv16i64:
      case nxv16f16:
      case nxv16bf16:
      case nxv16f32: return 16;
      case v12i32:
      case v12f32: return 12;
      case v11i32:
      case v11f32: return 11;
      case v10i32:
      case v10f32: return 10;
      case v9i32:
      case v9f32: return 9;
      case v8i1:
      case v8i8:
      case v8i16:
      case v8i32:
      case v8i64:
      case v8f16:
      case v8bf16:
      case v8f32:
      case v8f64:
      case nxv8i1:
      case nxv8i8:
      case nxv8i16:
      case nxv8i32:
      case nxv8i64:
      case nxv8f16:
      case nxv8bf16:
      case nxv8f32:
      case nxv8f64: return 8;
      case v7i32:
      case v7f32: return 7;
      case v6i32:
      case v6f32: return 6;
      case v5i32:
      case v5f32: return 5;
      case v4i1:
      case v4i8:
      case v4i16:
      case v4i32:
      case v4i64:
      case v4f16:
      case v4bf16:
      case v4f32:
      case v4f64:
      case nxv4i1:
      case nxv4i8:
      case nxv4i16:
      case nxv4i32:
      case nxv4i64:
      case nxv4f16:
      case nxv4bf16:
      case nxv4f32:
      case nxv4f64: return 4;
      case v3i16:
      case v3i32:
      case v3i64:
      case v3f16:
      case v3bf16:
      case v3f32:
      case v3f64: return 3;
      case v2i1:
      case v2i8:
      case v2i16:
      case v2i32:
      case v2i64:
      case v2f16:
      case v2bf16:
      case v2f32:
      case v2f64:
      case nxv2i1:
      case nxv2i8:
      case nxv2i16:
      case nxv2i32:
      case nxv2i64:
      case nxv2f16:
      case nxv2bf16:
      case nxv2f32:
      case nxv2f64: return 2;
      case v1i1:
      case v1i8:
      case v1i16:
      case v1i32:
      case v1i64:
      case v1i128:
      case v1f16:
      case v1f32:
      case v1f64:
      case nxv1i1:
      case nxv1i8:
      case nxv1i16:
      case nxv1i32:
      case nxv1i64:
      case nxv1f16:
      case nxv1bf16:
      case nxv1f32:
      case nxv1f64: return 1;
      }
    }

    ElementCount getVectorElementCount() const {
      return ElementCount::get(getVectorMinNumElements(), isScalableVector());
    }

    unsigned getVectorNumElements() const {
      if (isScalableVector())
        llvm::reportInvalidSizeRequest(
            "Possible incorrect use of MVT::getVectorNumElements() for "
            "scalable vector. Scalable flag may be dropped, use "
            "MVT::getVectorElementCount() instead");
      return getVectorMinNumElements();
    }

    /// Returns the size of the specified MVT in bits.
    ///
    /// If the value type is a scalable vector type, the scalable property will
    /// be set and the runtime size will be a positive integer multiple of the
    /// base size.
    TypeSize getSizeInBits() const {
      switch (SimpleTy) {
      default:
        switch (SimpleTy) {
        default:
          llvm_unreachable("getSizeInBits called on extended MVT.");

#define GET_VT_ATTR(Ty, N, Sz)                                                 \
  case Ty:                                                                     \
    return (MVT(Ty).isScalableVector() ? TypeSize::Scalable(Sz)                \
                                       : TypeSize::Fixed(Sz));
#include "GenVT.inc"
#undef GET_VT_ATTR
        }
      case Other:
        llvm_unreachable("Value type is non-standard value, Other.");
      case iPTR:
        llvm_unreachable("Value type size is target-dependent. Ask TLI.");
      case iPTRAny:
      case iAny:
      case fAny:
      case vAny:
      case Any:
        llvm_unreachable("Value type is overloaded.");
      case token:
        llvm_unreachable("Token type is a sentinel that cannot be used "
                         "in codegen and has no size");
      case Metadata:
        llvm_unreachable("Value type is metadata.");
      }
    }

    /// Return the size of the specified fixed width value type in bits. The
    /// function will assert if the type is scalable.
    uint64_t getFixedSizeInBits() const {
      return getSizeInBits().getFixedValue();
    }

    uint64_t getScalarSizeInBits() const {
      return getScalarType().getSizeInBits().getFixedValue();
    }

    /// Return the number of bytes overwritten by a store of the specified value
    /// type.
    ///
    /// If the value type is a scalable vector type, the scalable property will
    /// be set and the runtime size will be a positive integer multiple of the
    /// base size.
    TypeSize getStoreSize() const {
      TypeSize BaseSize = getSizeInBits();
      return {(BaseSize.getKnownMinValue() + 7) / 8, BaseSize.isScalable()};
    }

    // Return the number of bytes overwritten by a store of this value type or
    // this value type's element type in the case of a vector.
    uint64_t getScalarStoreSize() const {
      return getScalarType().getStoreSize().getFixedValue();
    }

    /// Return the number of bits overwritten by a store of the specified value
    /// type.
    ///
    /// If the value type is a scalable vector type, the scalable property will
    /// be set and the runtime size will be a positive integer multiple of the
    /// base size.
    TypeSize getStoreSizeInBits() const {
      return getStoreSize() * 8;
    }

    /// Returns true if the number of bits for the type is a multiple of an
    /// 8-bit byte.
    bool isByteSized() const { return getSizeInBits().isKnownMultipleOf(8); }

    /// Return true if we know at compile time this has more bits than VT.
    bool knownBitsGT(MVT VT) const {
      return TypeSize::isKnownGT(getSizeInBits(), VT.getSizeInBits());
    }

    /// Return true if we know at compile time this has more than or the same
    /// bits as VT.
    bool knownBitsGE(MVT VT) const {
      return TypeSize::isKnownGE(getSizeInBits(), VT.getSizeInBits());
    }

    /// Return true if we know at compile time this has fewer bits than VT.
    bool knownBitsLT(MVT VT) const {
      return TypeSize::isKnownLT(getSizeInBits(), VT.getSizeInBits());
    }

    /// Return true if we know at compile time this has fewer than or the same
    /// bits as VT.
    bool knownBitsLE(MVT VT) const {
      return TypeSize::isKnownLE(getSizeInBits(), VT.getSizeInBits());
    }

    /// Return true if this has more bits than VT.
    bool bitsGT(MVT VT) const {
      assert(isScalableVector() == VT.isScalableVector() &&
             "Comparison between scalable and fixed types");
      return knownBitsGT(VT);
    }

    /// Return true if this has no less bits than VT.
    bool bitsGE(MVT VT) const {
      assert(isScalableVector() == VT.isScalableVector() &&
             "Comparison between scalable and fixed types");
      return knownBitsGE(VT);
    }

    /// Return true if this has less bits than VT.
    bool bitsLT(MVT VT) const {
      assert(isScalableVector() == VT.isScalableVector() &&
             "Comparison between scalable and fixed types");
      return knownBitsLT(VT);
    }

    /// Return true if this has no more bits than VT.
    bool bitsLE(MVT VT) const {
      assert(isScalableVector() == VT.isScalableVector() &&
             "Comparison between scalable and fixed types");
      return knownBitsLE(VT);
    }

    static MVT getFloatingPointVT(unsigned BitWidth) {
      switch (BitWidth) {
      default:
        llvm_unreachable("Bad bit width!");
      case 16:
        return MVT::f16;
      case 32:
        return MVT::f32;
      case 64:
        return MVT::f64;
      case 80:
        return MVT::f80;
      case 128:
        return MVT::f128;
      }
    }

    static MVT getIntegerVT(unsigned BitWidth) {
      switch (BitWidth) {
      default:
        return (MVT::SimpleValueType)(MVT::INVALID_SIMPLE_VALUE_TYPE);
      case 1:
        return MVT::i1;
      case 2:
        return MVT::i2;
      case 4:
        return MVT::i4;
      case 8:
        return MVT::i8;
      case 16:
        return MVT::i16;
      case 32:
        return MVT::i32;
      case 64:
        return MVT::i64;
      case 128:
        return MVT::i128;
      }
    }

    static MVT getVectorVT(MVT VT, unsigned NumElements) {
      // clang-format off
      switch (VT.SimpleTy) {
      default:
        break;
      case MVT::i1:
        if (NumElements == 1)    return MVT::v1i1;
        if (NumElements == 2)    return MVT::v2i1;
        if (NumElements == 4)    return MVT::v4i1;
        if (NumElements == 8)    return MVT::v8i1;
        if (NumElements == 16)   return MVT::v16i1;
        if (NumElements == 32)   return MVT::v32i1;
        if (NumElements == 64)   return MVT::v64i1;
        if (NumElements == 128)  return MVT::v128i1;
        if (NumElements == 256)  return MVT::v256i1;
        if (NumElements == 512)  return MVT::v512i1;
        if (NumElements == 1024) return MVT::v1024i1;
        if (NumElements == 2048) return MVT::v2048i1;
        break;
      case MVT::i2:
        if (NumElements == 128) return MVT::v128i2;
        if (NumElements == 256) return MVT::v256i2;
        break;
      case MVT::i4:
        if (NumElements == 64)  return MVT::v64i4;
        if (NumElements == 128) return MVT::v128i4;
        break;
      case MVT::i8:
        if (NumElements == 1)   return MVT::v1i8;
        if (NumElements == 2)   return MVT::v2i8;
        if (NumElements == 4)   return MVT::v4i8;
        if (NumElements == 8)   return MVT::v8i8;
        if (NumElements == 16)  return MVT::v16i8;
        if (NumElements == 32)  return MVT::v32i8;
        if (NumElements == 64)  return MVT::v64i8;
        if (NumElements == 128) return MVT::v128i8;
        if (NumElements == 256) return MVT::v256i8;
        if (NumElements == 512) return MVT::v512i8;
        if (NumElements == 1024) return MVT::v1024i8;
        break;
      case MVT::i16:
        if (NumElements == 1)   return MVT::v1i16;
        if (NumElements == 2)   return MVT::v2i16;
        if (NumElements == 3)   return MVT::v3i16;
        if (NumElements == 4)   return MVT::v4i16;
        if (NumElements == 8)   return MVT::v8i16;
        if (NumElements == 16)  return MVT::v16i16;
        if (NumElements == 32)  return MVT::v32i16;
        if (NumElements == 64)  return MVT::v64i16;
        if (NumElements == 128) return MVT::v128i16;
        if (NumElements == 256) return MVT::v256i16;
        if (NumElements == 512) return MVT::v512i16;
        break;
      case MVT::i32:
        if (NumElements == 1)    return MVT::v1i32;
        if (NumElements == 2)    return MVT::v2i32;
        if (NumElements == 3)    return MVT::v3i32;
        if (NumElements == 4)    return MVT::v4i32;
        if (NumElements == 5)    return MVT::v5i32;
        if (NumElements == 6)    return MVT::v6i32;
        if (NumElements == 7)    return MVT::v7i32;
        if (NumElements == 8)    return MVT::v8i32;
        if (NumElements == 9)    return MVT::v9i32;
        if (NumElements == 10)   return MVT::v10i32;
        if (NumElements == 11)   return MVT::v11i32;
        if (NumElements == 12)   return MVT::v12i32;
        if (NumElements == 16)   return MVT::v16i32;
        if (NumElements == 32)   return MVT::v32i32;
        if (NumElements == 64)   return MVT::v64i32;
        if (NumElements == 128)  return MVT::v128i32;
        if (NumElements == 256)  return MVT::v256i32;
        if (NumElements == 512)  return MVT::v512i32;
        if (NumElements == 1024) return MVT::v1024i32;
        if (NumElements == 2048) return MVT::v2048i32;
        break;
      case MVT::i64:
        if (NumElements == 1)  return MVT::v1i64;
        if (NumElements == 2)  return MVT::v2i64;
        if (NumElements == 3)  return MVT::v3i64;
        if (NumElements == 4)  return MVT::v4i64;
        if (NumElements == 8)  return MVT::v8i64;
        if (NumElements == 16) return MVT::v16i64;
        if (NumElements == 32) return MVT::v32i64;
        if (NumElements == 64) return MVT::v64i64;
        if (NumElements == 128) return MVT::v128i64;
        if (NumElements == 256) return MVT::v256i64;
        break;
      case MVT::i128:
        if (NumElements == 1)  return MVT::v1i128;
        break;
      case MVT::f16:
        if (NumElements == 1)   return MVT::v1f16;
        if (NumElements == 2)   return MVT::v2f16;
        if (NumElements == 3)   return MVT::v3f16;
        if (NumElements == 4)   return MVT::v4f16;
        if (NumElements == 8)   return MVT::v8f16;
        if (NumElements == 16)  return MVT::v16f16;
        if (NumElements == 32)  return MVT::v32f16;
        if (NumElements == 64)  return MVT::v64f16;
        if (NumElements == 128) return MVT::v128f16;
        if (NumElements == 256) return MVT::v256f16;
        if (NumElements == 512) return MVT::v512f16;
        break;
      case MVT::bf16:
        if (NumElements == 2)   return MVT::v2bf16;
        if (NumElements == 3)   return MVT::v3bf16;
        if (NumElements == 4)   return MVT::v4bf16;
        if (NumElements == 8)   return MVT::v8bf16;
        if (NumElements == 16)  return MVT::v16bf16;
        if (NumElements == 32)  return MVT::v32bf16;
        if (NumElements == 64)  return MVT::v64bf16;
        if (NumElements == 128) return MVT::v128bf16;
        break;
      case MVT::f32:
        if (NumElements == 1)    return MVT::v1f32;
        if (NumElements == 2)    return MVT::v2f32;
        if (NumElements == 3)    return MVT::v3f32;
        if (NumElements == 4)    return MVT::v4f32;
        if (NumElements == 5)    return MVT::v5f32;
        if (NumElements == 6)    return MVT::v6f32;
        if (NumElements == 7)    return MVT::v7f32;
        if (NumElements == 8)    return MVT::v8f32;
        if (NumElements == 9)    return MVT::v9f32;
        if (NumElements == 10)   return MVT::v10f32;
        if (NumElements == 11)   return MVT::v11f32;
        if (NumElements == 12)   return MVT::v12f32;
        if (NumElements == 16)   return MVT::v16f32;
        if (NumElements == 32)   return MVT::v32f32;
        if (NumElements == 64)   return MVT::v64f32;
        if (NumElements == 128)  return MVT::v128f32;
        if (NumElements == 256)  return MVT::v256f32;
        if (NumElements == 512)  return MVT::v512f32;
        if (NumElements == 1024) return MVT::v1024f32;
        if (NumElements == 2048) return MVT::v2048f32;
        break;
      case MVT::f64:
        if (NumElements == 1)  return MVT::v1f64;
        if (NumElements == 2)  return MVT::v2f64;
        if (NumElements == 3)  return MVT::v3f64;
        if (NumElements == 4)  return MVT::v4f64;
        if (NumElements == 8)  return MVT::v8f64;
        if (NumElements == 16) return MVT::v16f64;
        if (NumElements == 32) return MVT::v32f64;
        if (NumElements == 64) return MVT::v64f64;
        if (NumElements == 128) return MVT::v128f64;
        if (NumElements == 256) return MVT::v256f64;
        break;
      }
      return (MVT::SimpleValueType)(MVT::INVALID_SIMPLE_VALUE_TYPE);
      // clang-format on
    }

    static MVT getScalableVectorVT(MVT VT, unsigned NumElements) {
      switch(VT.SimpleTy) {
        default:
          break;
        case MVT::i1:
          if (NumElements == 1)  return MVT::nxv1i1;
          if (NumElements == 2)  return MVT::nxv2i1;
          if (NumElements == 4)  return MVT::nxv4i1;
          if (NumElements == 8)  return MVT::nxv8i1;
          if (NumElements == 16) return MVT::nxv16i1;
          if (NumElements == 32) return MVT::nxv32i1;
          if (NumElements == 64) return MVT::nxv64i1;
          break;
        case MVT::i8:
          if (NumElements == 1)  return MVT::nxv1i8;
          if (NumElements == 2)  return MVT::nxv2i8;
          if (NumElements == 4)  return MVT::nxv4i8;
          if (NumElements == 8)  return MVT::nxv8i8;
          if (NumElements == 16) return MVT::nxv16i8;
          if (NumElements == 32) return MVT::nxv32i8;
          if (NumElements == 64) return MVT::nxv64i8;
          break;
        case MVT::i16:
          if (NumElements == 1)  return MVT::nxv1i16;
          if (NumElements == 2)  return MVT::nxv2i16;
          if (NumElements == 4)  return MVT::nxv4i16;
          if (NumElements == 8)  return MVT::nxv8i16;
          if (NumElements == 16) return MVT::nxv16i16;
          if (NumElements == 32) return MVT::nxv32i16;
          break;
        case MVT::i32:
          if (NumElements == 1)  return MVT::nxv1i32;
          if (NumElements == 2)  return MVT::nxv2i32;
          if (NumElements == 4)  return MVT::nxv4i32;
          if (NumElements == 8)  return MVT::nxv8i32;
          if (NumElements == 16) return MVT::nxv16i32;
          if (NumElements == 32) return MVT::nxv32i32;
          break;
        case MVT::i64:
          if (NumElements == 1)  return MVT::nxv1i64;
          if (NumElements == 2)  return MVT::nxv2i64;
          if (NumElements == 4)  return MVT::nxv4i64;
          if (NumElements == 8)  return MVT::nxv8i64;
          if (NumElements == 16) return MVT::nxv16i64;
          if (NumElements == 32) return MVT::nxv32i64;
          break;
        case MVT::f16:
          if (NumElements == 1)  return MVT::nxv1f16;
          if (NumElements == 2)  return MVT::nxv2f16;
          if (NumElements == 4)  return MVT::nxv4f16;
          if (NumElements == 8)  return MVT::nxv8f16;
          if (NumElements == 16)  return MVT::nxv16f16;
          if (NumElements == 32)  return MVT::nxv32f16;
          break;
        case MVT::bf16:
          if (NumElements == 1)  return MVT::nxv1bf16;
          if (NumElements == 2)  return MVT::nxv2bf16;
          if (NumElements == 4)  return MVT::nxv4bf16;
          if (NumElements == 8)  return MVT::nxv8bf16;
          if (NumElements == 16)  return MVT::nxv16bf16;
          if (NumElements == 32)  return MVT::nxv32bf16;
          break;
        case MVT::f32:
          if (NumElements == 1)  return MVT::nxv1f32;
          if (NumElements == 2)  return MVT::nxv2f32;
          if (NumElements == 4)  return MVT::nxv4f32;
          if (NumElements == 8)  return MVT::nxv8f32;
          if (NumElements == 16) return MVT::nxv16f32;
          break;
        case MVT::f64:
          if (NumElements == 1)  return MVT::nxv1f64;
          if (NumElements == 2)  return MVT::nxv2f64;
          if (NumElements == 4)  return MVT::nxv4f64;
          if (NumElements == 8)  return MVT::nxv8f64;
          break;
      }
      return (MVT::SimpleValueType)(MVT::INVALID_SIMPLE_VALUE_TYPE);
    }

    static MVT getVectorVT(MVT VT, unsigned NumElements, bool IsScalable) {
      if (IsScalable)
        return getScalableVectorVT(VT, NumElements);
      return getVectorVT(VT, NumElements);
    }

    static MVT getVectorVT(MVT VT, ElementCount EC) {
      if (EC.isScalable())
        return getScalableVectorVT(VT, EC.getKnownMinValue());
      return getVectorVT(VT, EC.getKnownMinValue());
    }

    /// Return the value type corresponding to the specified type.  This returns
    /// all pointers as iPTR.  If HandleUnknown is true, unknown types are
    /// returned as Other, otherwise they are invalid.
    static MVT getVT(Type *Ty, bool HandleUnknown = false);

  public:
    /// SimpleValueType Iteration
    /// @{
    static auto all_valuetypes() {
      return enum_seq_inclusive(MVT::FIRST_VALUETYPE, MVT::LAST_VALUETYPE,
                                force_iteration_on_noniterable_enum);
    }

    static auto integer_valuetypes() {
      return enum_seq_inclusive(MVT::FIRST_INTEGER_VALUETYPE,
                                MVT::LAST_INTEGER_VALUETYPE,
                                force_iteration_on_noniterable_enum);
    }

    static auto fp_valuetypes() {
      return enum_seq_inclusive(MVT::FIRST_FP_VALUETYPE, MVT::LAST_FP_VALUETYPE,
                                force_iteration_on_noniterable_enum);
    }

    static auto vector_valuetypes() {
      return enum_seq_inclusive(MVT::FIRST_VECTOR_VALUETYPE,
                                MVT::LAST_VECTOR_VALUETYPE,
                                force_iteration_on_noniterable_enum);
    }

    static auto fixedlen_vector_valuetypes() {
      return enum_seq_inclusive(MVT::FIRST_FIXEDLEN_VECTOR_VALUETYPE,
                                MVT::LAST_FIXEDLEN_VECTOR_VALUETYPE,
                                force_iteration_on_noniterable_enum);
    }

    static auto scalable_vector_valuetypes() {
      return enum_seq_inclusive(MVT::FIRST_SCALABLE_VECTOR_VALUETYPE,
                                MVT::LAST_SCALABLE_VECTOR_VALUETYPE,
                                force_iteration_on_noniterable_enum);
    }

    static auto integer_fixedlen_vector_valuetypes() {
      return enum_seq_inclusive(MVT::FIRST_INTEGER_FIXEDLEN_VECTOR_VALUETYPE,
                                MVT::LAST_INTEGER_FIXEDLEN_VECTOR_VALUETYPE,
                                force_iteration_on_noniterable_enum);
    }

    static auto fp_fixedlen_vector_valuetypes() {
      return enum_seq_inclusive(MVT::FIRST_FP_FIXEDLEN_VECTOR_VALUETYPE,
                                MVT::LAST_FP_FIXEDLEN_VECTOR_VALUETYPE,
                                force_iteration_on_noniterable_enum);
    }

    static auto integer_scalable_vector_valuetypes() {
      return enum_seq_inclusive(MVT::FIRST_INTEGER_SCALABLE_VECTOR_VALUETYPE,
                                MVT::LAST_INTEGER_SCALABLE_VECTOR_VALUETYPE,
                                force_iteration_on_noniterable_enum);
    }

    static auto fp_scalable_vector_valuetypes() {
      return enum_seq_inclusive(MVT::FIRST_FP_SCALABLE_VECTOR_VALUETYPE,
                                MVT::LAST_FP_SCALABLE_VECTOR_VALUETYPE,
                                force_iteration_on_noniterable_enum);
    }
    /// @}
  };

  inline raw_ostream &operator<<(raw_ostream &OS, const MVT &VT) {
    VT.print(OS);
    return OS;
  }

} // namespace llvm::tmp

#endif // LLVM_TMP_MACHINEVALUETYPE_H

//===- Support/MachineValueType.h - Machine-Level types ---------*- C++ -*-===//
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
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_MACHINEVALUETYPE_H
#define LLVM_SUPPORT_MACHINEVALUETYPE_H

#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TypeSize.h"
#include <cassert>

namespace llvm {

  class Type;

  /// Machine Value Type. Every type that is supported natively by some
  /// processor targeted by LLVM occurs here. This means that any legal value
  /// type can be represented by an MVT.
  class MVT {
  public:
    enum SimpleValueType : uint8_t {
      // Simple value types that aren't explicitly part of this enumeration
      // are considered extended value types.
      INVALID_SIMPLE_VALUE_TYPE = 0,

      // If you change this numbering, you must change the values in
      // ValueTypes.td as well!
      Other          =   1,   // This is a non-standard value
      i1             =   2,   // This is a 1 bit integer value
      i8             =   3,   // This is an 8 bit integer value
      i16            =   4,   // This is a 16 bit integer value
      i32            =   5,   // This is a 32 bit integer value
      i64            =   6,   // This is a 64 bit integer value
      i128           =   7,   // This is a 128 bit integer value

      FIRST_INTEGER_VALUETYPE = i1,
      LAST_INTEGER_VALUETYPE  = i128,

      bf16           =   8,   // This is a 16 bit brain floating point value
      f16            =   9,   // This is a 16 bit floating point value
      f32            =  10,   // This is a 32 bit floating point value
      f64            =  11,   // This is a 64 bit floating point value
      f80            =  12,   // This is a 80 bit floating point value
      f128           =  13,   // This is a 128 bit floating point value
      ppcf128        =  14,   // This is a PPC 128-bit floating point value

      FIRST_FP_VALUETYPE = bf16,
      LAST_FP_VALUETYPE  = ppcf128,

      v1i1           =  15,   //    1 x i1
      v2i1           =  16,   //    2 x i1
      v4i1           =  17,   //    4 x i1
      v8i1           =  18,   //    8 x i1
      v16i1          =  19,   //   16 x i1
      v32i1          =  20,   //   32 x i1
      v64i1          =  21,   //   64 x i1
      v128i1         =  22,   //  128 x i1
      v256i1         =  23,   //  256 x i1
      v512i1         =  24,   //  512 x i1
      v1024i1        =  25,   // 1024 x i1

      v1i8           =  26,   //  1 x i8
      v2i8           =  27,   //  2 x i8
      v4i8           =  28,   //  4 x i8
      v8i8           =  29,   //  8 x i8
      v16i8          =  30,   // 16 x i8
      v32i8          =  31,   // 32 x i8
      v64i8          =  32,   // 64 x i8
      v128i8         =  33,   //128 x i8
      v256i8         =  34,   //256 x i8

      v1i16          =  35,   //  1 x i16
      v2i16          =  36,   //  2 x i16
      v3i16          =  37,   //  3 x i16
      v4i16          =  38,   //  4 x i16
      v8i16          =  39,   //  8 x i16
      v16i16         =  40,   // 16 x i16
      v32i16         =  41,   // 32 x i16
      v64i16         =  42,   // 64 x i16
      v128i16        =  43,   //128 x i16

      v1i32          =  44,   //    1 x i32
      v2i32          =  45,   //    2 x i32
      v3i32          =  46,   //    3 x i32
      v4i32          =  47,   //    4 x i32
      v5i32          =  48,   //    5 x i32
      v8i32          =  49,   //    8 x i32
      v16i32         =  50,   //   16 x i32
      v32i32         =  51,   //   32 x i32
      v64i32         =  52,   //   64 x i32
      v128i32        =  53,   //  128 x i32
      v256i32        =  54,   //  256 x i32
      v512i32        =  55,   //  512 x i32
      v1024i32       =  56,   // 1024 x i32
      v2048i32       =  57,   // 2048 x i32

      v1i64          =  58,   //  1 x i64
      v2i64          =  59,   //  2 x i64
      v4i64          =  60,   //  4 x i64
      v8i64          =  61,   //  8 x i64
      v16i64         =  62,   // 16 x i64
      v32i64         =  63,   // 32 x i64

      v1i128         =  64,   //  1 x i128

      FIRST_INTEGER_FIXEDLEN_VECTOR_VALUETYPE = v1i1,
      LAST_INTEGER_FIXEDLEN_VECTOR_VALUETYPE = v1i128,

      v2f16          =  65,   //    2 x f16
      v3f16          =  66,   //    3 x f16
      v4f16          =  67,   //    4 x f16
      v8f16          =  68,   //    8 x f16
      v16f16         =  69,   //   16 x f16
      v32f16         =  70,   //   32 x f16
      v64f16         =  71,   //   64 x f16
      v128f16        =  72,   //  128 x f16
      v2bf16         =  73,   //    2 x bf16
      v3bf16         =  74,   //    3 x bf16
      v4bf16         =  75,   //    4 x bf16
      v8bf16         =  76,   //    8 x bf16
      v16bf16        =  77,   //   16 x bf16
      v32bf16        =  78,   //   32 x bf16
      v64bf16        =  79,   //   64 x bf16
      v128bf16       =  80,   //  128 x bf16
      v1f32          =  81,   //    1 x f32
      v2f32          =  82,   //    2 x f32
      v3f32          =  83,   //    3 x f32
      v4f32          =  84,   //    4 x f32
      v5f32          =  85,   //    5 x f32
      v8f32          =  86,   //    8 x f32
      v16f32         =  87,   //   16 x f32
      v32f32         =  88,   //   32 x f32
      v64f32         =  89,   //   64 x f32
      v128f32        =  90,   //  128 x f32
      v256f32        =  91,   //  256 x f32
      v512f32        =  92,   //  512 x f32
      v1024f32       =  93,   // 1024 x f32
      v2048f32       =  94,   // 2048 x f32
      v1f64          =  95,   //    1 x f64
      v2f64          =  96,   //    2 x f64
      v4f64          =  97,   //    4 x f64
      v8f64          =  98,   //    8 x f64
      v16f64         =  99,   //   16 x f64
      v32f64         = 100,   //   32 x f64

      FIRST_FP_FIXEDLEN_VECTOR_VALUETYPE = v2f16,
      LAST_FP_FIXEDLEN_VECTOR_VALUETYPE = v32f64,

      FIRST_FIXEDLEN_VECTOR_VALUETYPE = v1i1,
      LAST_FIXEDLEN_VECTOR_VALUETYPE = v32f64,

      nxv1i1         = 101,   // n x  1 x i1
      nxv2i1         = 102,   // n x  2 x i1
      nxv4i1         = 103,   // n x  4 x i1
      nxv8i1         = 104,   // n x  8 x i1
      nxv16i1        = 105,   // n x 16 x i1
      nxv32i1        = 106,   // n x 32 x i1
      nxv64i1        = 107,   // n x  64 x i1

      nxv1i8         = 108,   // n x  1 x i8
      nxv2i8         = 109,   // n x  2 x i8
      nxv4i8         = 110,   // n x  4 x i8
      nxv8i8         = 111,   // n x  8 x i8
      nxv16i8        = 112,   // n x 16 x i8
      nxv32i8        = 113,   // n x 32 x i8
      nxv64i8        = 114,   // n x  64 x i8

      nxv1i16        = 115,  // n x  1 x i16
      nxv2i16        = 116,  // n x  2 x i16
      nxv4i16        = 117,  // n x  4 x i16
      nxv8i16        = 118,  // n x  8 x i16
      nxv16i16       = 119,  // n x 16 x i16
      nxv32i16       = 120,  // n x 32 x i16

      nxv1i32        = 121,  // n x  1 x i32
      nxv2i32        = 122,  // n x  2 x i32
      nxv4i32        = 123,  // n x  4 x i32
      nxv8i32        = 124,  // n x  8 x i32
      nxv16i32       = 125,  // n x 16 x i32
      nxv32i32       = 126,  // n x 32 x i32

      nxv1i64        = 127,  // n x  1 x i64
      nxv2i64        = 128,  // n x  2 x i64
      nxv4i64        = 129,  // n x  4 x i64
      nxv8i64        = 130,  // n x  8 x i64
      nxv16i64       = 131,  // n x 16 x i64
      nxv32i64       = 132,  // n x 32 x i64

      FIRST_INTEGER_SCALABLE_VECTOR_VALUETYPE = nxv1i1,
      LAST_INTEGER_SCALABLE_VECTOR_VALUETYPE = nxv32i64,

      nxv1f16        = 133,   // n x   1 x f16
      nxv2f16        = 134,  // n x  2 x f16
      nxv4f16        = 135,  // n x  4 x f16
      nxv8f16        = 136,  // n x  8 x f16
      nxv16f16       = 137,   // n x  16 x f16
      nxv32f16       = 138,   // n x  32 x f16
      nxv2bf16       = 139,  // n x  2 x bf16
      nxv4bf16       = 140,  // n x  4 x bf16
      nxv8bf16       = 141,  // n x  8 x bf16
      nxv1f32        = 142,  // n x  1 x f32
      nxv2f32        = 143,  // n x  2 x f32
      nxv4f32        = 144,  // n x  4 x f32
      nxv8f32        = 145,  // n x  8 x f32
      nxv16f32       = 146,  // n x 16 x f32
      nxv1f64        = 147,  // n x  1 x f64
      nxv2f64        = 148,  // n x  2 x f64
      nxv4f64        = 149,  // n x  4 x f64
      nxv8f64        = 150,  // n x  8 x f64

      FIRST_FP_SCALABLE_VECTOR_VALUETYPE = nxv1f16,
      LAST_FP_SCALABLE_VECTOR_VALUETYPE = nxv8f64,

      FIRST_SCALABLE_VECTOR_VALUETYPE = nxv1i1,
      LAST_SCALABLE_VECTOR_VALUETYPE = nxv8f64,

      FIRST_VECTOR_VALUETYPE = v1i1,
      LAST_VECTOR_VALUETYPE  = nxv8f64,

      x86mmx         = 151,   // This is an X86 MMX value

      Glue           = 152,   // This glues nodes together during pre-RA sched

      isVoid         = 153,   // This has no value

      Untyped        = 154,   // This value takes a register, but has
                              // unspecified type.  The register class
                              // will be determined by the opcode.

      exnref         = 155,   // WebAssembly's exnref type

      FIRST_VALUETYPE =  1,   // This is always the beginning of the list.
      LAST_VALUETYPE = 156,   // This always remains at the end of the list.

      // This is the current maximum for LAST_VALUETYPE.
      // MVT::MAX_ALLOWED_VALUETYPE is used for asserts and to size bit vectors
      // This value must be a multiple of 32.
      MAX_ALLOWED_VALUETYPE = 160,

      // A value of type llvm::TokenTy
      token          = 248,

      // This is MDNode or MDString.
      Metadata       = 249,

      // An int value the size of the pointer of the current
      // target to any address space. This must only be used internal to
      // tblgen. Other than for overloading, we treat iPTRAny the same as iPTR.
      iPTRAny        = 250,

      // A vector with any length and element size. This is used
      // for intrinsics that have overloadings based on vector types.
      // This is only for tblgen's consumption!
      vAny           = 251,

      // Any floating-point or vector floating-point value. This is used
      // for intrinsics that have overloadings based on floating-point types.
      // This is only for tblgen's consumption!
      fAny           = 252,

      // An integer or vector integer value of any bit width. This is
      // used for intrinsics that have overloadings based on integer bit widths.
      // This is only for tblgen's consumption!
      iAny           = 253,

      // An int value the size of the pointer of the current
      // target.  This should only be used internal to tblgen!
      iPTR           = 254,

      // Any type. This is used for intrinsics that have overloadings.
      // This is only for tblgen's consumption!
      Any            = 255
    };

    SimpleValueType SimpleTy = INVALID_SIMPLE_VALUE_TYPE;

    constexpr MVT() = default;
    constexpr MVT(SimpleValueType SVT) : SimpleTy(SVT) {}

    bool operator>(const MVT& S)  const { return SimpleTy >  S.SimpleTy; }
    bool operator<(const MVT& S)  const { return SimpleTy <  S.SimpleTy; }
    bool operator==(const MVT& S) const { return SimpleTy == S.SimpleTy; }
    bool operator!=(const MVT& S) const { return SimpleTy != S.SimpleTy; }
    bool operator>=(const MVT& S) const { return SimpleTy >= S.SimpleTy; }
    bool operator<=(const MVT& S) const { return SimpleTy <= S.SimpleTy; }

    /// Return true if this is a valid simple valuetype.
    bool isValid() const {
      return (SimpleTy >= MVT::FIRST_VALUETYPE &&
              SimpleTy < MVT::LAST_VALUETYPE);
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

    bool isFixedLengthVector() const {
      return (SimpleTy >= MVT::FIRST_FIXEDLEN_VECTOR_VALUETYPE &&
              SimpleTy <= MVT::LAST_FIXEDLEN_VECTOR_VALUETYPE);
    }

    /// Return true if this is a 16-bit vector type.
    bool is16BitVector() const {
      return (SimpleTy == MVT::v2i8  || SimpleTy == MVT::v1i16 ||
              SimpleTy == MVT::v16i1);
    }

    /// Return true if this is a 32-bit vector type.
    bool is32BitVector() const {
      return (SimpleTy == MVT::v32i1 || SimpleTy == MVT::v4i8   ||
              SimpleTy == MVT::v2i16 || SimpleTy == MVT::v1i32  ||
              SimpleTy == MVT::v2f16 || SimpleTy == MVT::v2bf16 ||
              SimpleTy == MVT::v1f32);
    }

    /// Return true if this is a 64-bit vector type.
    bool is64BitVector() const {
      return (SimpleTy == MVT::v64i1  || SimpleTy == MVT::v8i8  ||
              SimpleTy == MVT::v4i16  || SimpleTy == MVT::v2i32 ||
              SimpleTy == MVT::v1i64  || SimpleTy == MVT::v4f16 ||
              SimpleTy == MVT::v4bf16 ||SimpleTy == MVT::v2f32  ||
              SimpleTy == MVT::v1f64);
    }

    /// Return true if this is a 128-bit vector type.
    bool is128BitVector() const {
      return (SimpleTy == MVT::v128i1 || SimpleTy == MVT::v16i8  ||
              SimpleTy == MVT::v8i16  || SimpleTy == MVT::v4i32  ||
              SimpleTy == MVT::v2i64  || SimpleTy == MVT::v1i128 ||
              SimpleTy == MVT::v8f16  || SimpleTy == MVT::v8bf16 ||
              SimpleTy == MVT::v4f32  || SimpleTy == MVT::v2f64);
    }

    /// Return true if this is a 256-bit vector type.
    bool is256BitVector() const {
      return (SimpleTy == MVT::v16f16 || SimpleTy == MVT::v16bf16 ||
              SimpleTy == MVT::v8f32  || SimpleTy == MVT::v4f64   ||
              SimpleTy == MVT::v32i8  || SimpleTy == MVT::v16i16  ||
              SimpleTy == MVT::v8i32  || SimpleTy == MVT::v4i64   ||
              SimpleTy == MVT::v256i1);
    }

    /// Return true if this is a 512-bit vector type.
    bool is512BitVector() const {
      return (SimpleTy == MVT::v32f16 || SimpleTy == MVT::v32bf16 ||
              SimpleTy == MVT::v16f32 || SimpleTy == MVT::v8f64   ||
              SimpleTy == MVT::v512i1 || SimpleTy == MVT::v64i8   ||
              SimpleTy == MVT::v32i16 || SimpleTy == MVT::v16i32  ||
              SimpleTy == MVT::v8i64);
    }

    /// Return true if this is a 1024-bit vector type.
    bool is1024BitVector() const {
      return (SimpleTy == MVT::v1024i1 || SimpleTy == MVT::v128i8 ||
              SimpleTy == MVT::v64i16  || SimpleTy == MVT::v32i32 ||
              SimpleTy == MVT::v16i64  || SimpleTy == MVT::v64f16 ||
              SimpleTy == MVT::v32f32  || SimpleTy == MVT::v16f64 ||
              SimpleTy == MVT::v64bf16);
    }

    /// Return true if this is a 2048-bit vector type.
    bool is2048BitVector() const {
      return (SimpleTy == MVT::v256i8  || SimpleTy == MVT::v128i16 ||
              SimpleTy == MVT::v64i32  || SimpleTy == MVT::v32i64  ||
              SimpleTy == MVT::v128f16 || SimpleTy == MVT::v64f32  ||
              SimpleTy == MVT::v32f64  || SimpleTy == MVT::v128bf16);
    }

    /// Return true if this is an overloaded type for TableGen.
    bool isOverloaded() const {
      return (SimpleTy == MVT::Any || SimpleTy == MVT::iAny ||
              SimpleTy == MVT::fAny || SimpleTy == MVT::vAny ||
              SimpleTy == MVT::iPTRAny);
    }

    /// Return a VT for a vector type with the same element type but
    /// half the number of elements.
    MVT getHalfNumVectorElementsVT() const {
      MVT EltVT = getVectorElementType();
      auto EltCnt = getVectorElementCount();
      assert(!(EltCnt.Min & 1) && "Splitting vector, but not in half!");
      return getVectorVT(EltVT, EltCnt / 2);
    }

    /// Returns true if the given vector is a power of 2.
    bool isPow2VectorType() const {
      unsigned NElts = getVectorNumElements();
      return !(NElts & (NElts - 1));
    }

    /// Widens the length of the given vector MVT up to the nearest power of 2
    /// and returns that type.
    MVT getPow2VectorType() const {
      if (isPow2VectorType())
        return *this;

      unsigned NElts = getVectorNumElements();
      unsigned Pow2NElts = 1 << Log2_32_Ceil(NElts);
      return MVT::getVectorVT(getVectorElementType(), Pow2NElts);
    }

    /// If this is a vector, return the element type, otherwise return this.
    MVT getScalarType() const {
      return isVector() ? getVectorElementType() : *this;
    }

    MVT getVectorElementType() const {
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
      case nxv1i1:
      case nxv2i1:
      case nxv4i1:
      case nxv8i1:
      case nxv16i1:
      case nxv32i1:
      case nxv64i1: return i1;
      case v1i8:
      case v2i8:
      case v4i8:
      case v8i8:
      case v16i8:
      case v32i8:
      case v64i8:
      case v128i8:
      case v256i8:
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
      case v8i32:
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
      case v4i64:
      case v8i64:
      case v16i64:
      case v32i64:
      case nxv1i64:
      case nxv2i64:
      case nxv4i64:
      case nxv8i64:
      case nxv16i64:
      case nxv32i64: return i64;
      case v1i128: return i128;
      case v2f16:
      case v3f16:
      case v4f16:
      case v8f16:
      case v16f16:
      case v32f16:
      case v64f16:
      case v128f16:
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
      case nxv2bf16:
      case nxv4bf16:
      case nxv8bf16: return bf16;
      case v1f32:
      case v2f32:
      case v3f32:
      case v4f32:
      case v5f32:
      case v8f32:
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
      case v4f64:
      case v8f64:
      case v16f64:
      case v32f64:
      case nxv1f64:
      case nxv2f64:
      case nxv4f64:
      case nxv8f64: return f64;
      }
    }

    unsigned getVectorNumElements() const {
      switch (SimpleTy) {
      default:
        llvm_unreachable("Not a vector MVT!");
      case v2048i32:
      case v2048f32: return 2048;
      case v1024i1:
      case v1024i32:
      case v1024f32: return 1024;
      case v512i1:
      case v512i32:
      case v512f32: return 512;
      case v256i1:
      case v256i8:
      case v256i32:
      case v256f32: return 256;
      case v128i1:
      case v128i8:
      case v128i16:
      case v128i32:
      case v128f16:
      case v128bf16:
      case v128f32: return 128;
      case v64i1:
      case v64i8:
      case v64i16:
      case v64i32:
      case v64f16:
      case v64bf16:
      case v64f32:
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
      case nxv32f16: return 32;
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
      case nxv16f32: return 16;
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
      case v3f16:
      case v3bf16:
      case v3f32: return 3;
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
      case v1f32:
      case v1f64:
      case nxv1i1:
      case nxv1i8:
      case nxv1i16:
      case nxv1i32:
      case nxv1i64:
      case nxv1f16:
      case nxv1f32:
      case nxv1f64: return 1;
      }
    }

    ElementCount getVectorElementCount() const {
      return { getVectorNumElements(), isScalableVector() };
    }

    /// Given a vector type, return the minimum number of elements it contains.
    unsigned getVectorMinNumElements() const {
      return getVectorElementCount().Min;
    }

    /// Returns the size of the specified MVT in bits.
    ///
    /// If the value type is a scalable vector type, the scalable property will
    /// be set and the runtime size will be a positive integer multiple of the
    /// base size.
    TypeSize getSizeInBits() const {
      switch (SimpleTy) {
      default:
        llvm_unreachable("getSizeInBits called on extended MVT.");
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
      case i1:
      case v1i1: return TypeSize::Fixed(1);
      case nxv1i1: return TypeSize::Scalable(1);
      case v2i1: return TypeSize::Fixed(2);
      case nxv2i1: return TypeSize::Scalable(2);
      case v4i1: return TypeSize::Fixed(4);
      case nxv4i1: return TypeSize::Scalable(4);
      case i8  :
      case v1i8:
      case v8i1: return TypeSize::Fixed(8);
      case nxv1i8:
      case nxv8i1: return TypeSize::Scalable(8);
      case i16 :
      case f16:
      case bf16:
      case v16i1:
      case v2i8:
      case v1i16: return TypeSize::Fixed(16);
      case nxv16i1:
      case nxv2i8:
      case nxv1i16:
      case nxv1f16: return TypeSize::Scalable(16);
      case f32 :
      case i32 :
      case v32i1:
      case v4i8:
      case v2i16:
      case v2f16:
      case v2bf16:
      case v1f32:
      case v1i32: return TypeSize::Fixed(32);
      case nxv32i1:
      case nxv4i8:
      case nxv2i16:
      case nxv1i32:
      case nxv2f16:
      case nxv2bf16:
      case nxv1f32: return TypeSize::Scalable(32);
      case v3i16:
      case v3f16:
      case v3bf16: return TypeSize::Fixed(48);
      case x86mmx:
      case f64 :
      case i64 :
      case v64i1:
      case v8i8:
      case v4i16:
      case v2i32:
      case v1i64:
      case v4f16:
      case v4bf16:
      case v2f32:
      case v1f64: return TypeSize::Fixed(64);
      case nxv64i1:
      case nxv8i8:
      case nxv4i16:
      case nxv2i32:
      case nxv1i64:
      case nxv4f16:
      case nxv4bf16:
      case nxv2f32:
      case nxv1f64: return TypeSize::Scalable(64);
      case f80 :  return TypeSize::Fixed(80);
      case v3i32:
      case v3f32: return TypeSize::Fixed(96);
      case f128:
      case ppcf128:
      case i128:
      case v128i1:
      case v16i8:
      case v8i16:
      case v4i32:
      case v2i64:
      case v1i128:
      case v8f16:
      case v8bf16:
      case v4f32:
      case v2f64: return TypeSize::Fixed(128);
      case nxv16i8:
      case nxv8i16:
      case nxv4i32:
      case nxv2i64:
      case nxv8f16:
      case nxv8bf16:
      case nxv4f32:
      case nxv2f64: return TypeSize::Scalable(128);
      case v5i32:
      case v5f32: return TypeSize::Fixed(160);
      case v256i1:
      case v32i8:
      case v16i16:
      case v8i32:
      case v4i64:
      case v16f16:
      case v16bf16:
      case v8f32:
      case v4f64: return TypeSize::Fixed(256);
      case nxv32i8:
      case nxv16i16:
      case nxv8i32:
      case nxv4i64:
      case nxv16f16:
      case nxv8f32:
      case nxv4f64: return TypeSize::Scalable(256);
      case v512i1:
      case v64i8:
      case v32i16:
      case v16i32:
      case v8i64:
      case v32f16:
      case v32bf16:
      case v16f32:
      case v8f64: return TypeSize::Fixed(512);
      case nxv64i8:
      case nxv32i16:
      case nxv16i32:
      case nxv8i64:
      case nxv32f16:
      case nxv16f32:
      case nxv8f64: return TypeSize::Scalable(512);
      case v1024i1:
      case v128i8:
      case v64i16:
      case v32i32:
      case v16i64:
      case v64f16:
      case v64bf16:
      case v32f32:
      case v16f64: return TypeSize::Fixed(1024);
      case nxv32i32:
      case nxv16i64: return TypeSize::Scalable(1024);
      case v256i8:
      case v128i16:
      case v64i32:
      case v32i64:
      case v128f16:
      case v128bf16:
      case v64f32:
      case v32f64: return TypeSize::Fixed(2048);
      case nxv32i64: return TypeSize::Scalable(2048);
      case v128i32:
      case v128f32:  return TypeSize::Fixed(4096);
      case v256i32:
      case v256f32:  return TypeSize::Fixed(8192);
      case v512i32:
      case v512f32:  return TypeSize::Fixed(16384);
      case v1024i32:
      case v1024f32:  return TypeSize::Fixed(32768);
      case v2048i32:
      case v2048f32:  return TypeSize::Fixed(65536);
      case exnref: return TypeSize::Fixed(0); // opaque type
      }
    }

    TypeSize getScalarSizeInBits() const {
      return getScalarType().getSizeInBits();
    }

    /// Return the number of bytes overwritten by a store of the specified value
    /// type.
    ///
    /// If the value type is a scalable vector type, the scalable property will
    /// be set and the runtime size will be a positive integer multiple of the
    /// base size.
    TypeSize getStoreSize() const {
      TypeSize BaseSize = getSizeInBits();
      return {(BaseSize.getKnownMinSize() + 7) / 8, BaseSize.isScalable()};
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
    bool isByteSized() const {
      return getSizeInBits().isByteSized();
    }

    /// Return true if this has more bits than VT.
    bool bitsGT(MVT VT) const {
      return getSizeInBits() > VT.getSizeInBits();
    }

    /// Return true if this has no less bits than VT.
    bool bitsGE(MVT VT) const {
      return getSizeInBits() >= VT.getSizeInBits();
    }

    /// Return true if this has less bits than VT.
    bool bitsLT(MVT VT) const {
      return getSizeInBits() < VT.getSizeInBits();
    }

    /// Return true if this has no more bits than VT.
    bool bitsLE(MVT VT) const {
      return getSizeInBits() <= VT.getSizeInBits();
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
        break;
      case MVT::i32:
        if (NumElements == 1)    return MVT::v1i32;
        if (NumElements == 2)    return MVT::v2i32;
        if (NumElements == 3)    return MVT::v3i32;
        if (NumElements == 4)    return MVT::v4i32;
        if (NumElements == 5)    return MVT::v5i32;
        if (NumElements == 8)    return MVT::v8i32;
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
        if (NumElements == 4)  return MVT::v4i64;
        if (NumElements == 8)  return MVT::v8i64;
        if (NumElements == 16) return MVT::v16i64;
        if (NumElements == 32) return MVT::v32i64;
        break;
      case MVT::i128:
        if (NumElements == 1)  return MVT::v1i128;
        break;
      case MVT::f16:
        if (NumElements == 2)   return MVT::v2f16;
        if (NumElements == 3)   return MVT::v3f16;
        if (NumElements == 4)   return MVT::v4f16;
        if (NumElements == 8)   return MVT::v8f16;
        if (NumElements == 16)  return MVT::v16f16;
        if (NumElements == 32)  return MVT::v32f16;
        if (NumElements == 64)  return MVT::v64f16;
        if (NumElements == 128) return MVT::v128f16;
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
        if (NumElements == 8)    return MVT::v8f32;
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
        if (NumElements == 4)  return MVT::v4f64;
        if (NumElements == 8)  return MVT::v8f64;
        if (NumElements == 16) return MVT::v16f64;
        if (NumElements == 32) return MVT::v32f64;
        break;
      }
      return (MVT::SimpleValueType)(MVT::INVALID_SIMPLE_VALUE_TYPE);
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
          if (NumElements == 2)  return MVT::nxv2bf16;
          if (NumElements == 4)  return MVT::nxv4bf16;
          if (NumElements == 8)  return MVT::nxv8bf16;
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
      if (EC.Scalable)
        return getScalableVectorVT(VT, EC.Min);
      return getVectorVT(VT, EC.Min);
    }

    /// Return the value type corresponding to the specified type.  This returns
    /// all pointers as iPTR.  If HandleUnknown is true, unknown types are
    /// returned as Other, otherwise they are invalid.
    static MVT getVT(Type *Ty, bool HandleUnknown = false);

  private:
    /// A simple iterator over the MVT::SimpleValueType enum.
    struct mvt_iterator {
      SimpleValueType VT;

      mvt_iterator(SimpleValueType VT) : VT(VT) {}

      MVT operator*() const { return VT; }
      bool operator!=(const mvt_iterator &LHS) const { return VT != LHS.VT; }

      mvt_iterator& operator++() {
        VT = (MVT::SimpleValueType)((int)VT + 1);
        assert((int)VT <= MVT::MAX_ALLOWED_VALUETYPE &&
               "MVT iterator overflowed.");
        return *this;
      }
    };

    /// A range of the MVT::SimpleValueType enum.
    using mvt_range = iterator_range<mvt_iterator>;

  public:
    /// SimpleValueType Iteration
    /// @{
    static mvt_range all_valuetypes() {
      return mvt_range(MVT::FIRST_VALUETYPE, MVT::LAST_VALUETYPE);
    }

    static mvt_range integer_valuetypes() {
      return mvt_range(MVT::FIRST_INTEGER_VALUETYPE,
                       (MVT::SimpleValueType)(MVT::LAST_INTEGER_VALUETYPE + 1));
    }

    static mvt_range fp_valuetypes() {
      return mvt_range(MVT::FIRST_FP_VALUETYPE,
                       (MVT::SimpleValueType)(MVT::LAST_FP_VALUETYPE + 1));
    }

    static mvt_range vector_valuetypes() {
      return mvt_range(MVT::FIRST_VECTOR_VALUETYPE,
                       (MVT::SimpleValueType)(MVT::LAST_VECTOR_VALUETYPE + 1));
    }

    static mvt_range fixedlen_vector_valuetypes() {
      return mvt_range(
               MVT::FIRST_FIXEDLEN_VECTOR_VALUETYPE,
               (MVT::SimpleValueType)(MVT::LAST_FIXEDLEN_VECTOR_VALUETYPE + 1));
    }

    static mvt_range scalable_vector_valuetypes() {
      return mvt_range(
               MVT::FIRST_SCALABLE_VECTOR_VALUETYPE,
               (MVT::SimpleValueType)(MVT::LAST_SCALABLE_VECTOR_VALUETYPE + 1));
    }

    static mvt_range integer_fixedlen_vector_valuetypes() {
      return mvt_range(
       MVT::FIRST_INTEGER_FIXEDLEN_VECTOR_VALUETYPE,
       (MVT::SimpleValueType)(MVT::LAST_INTEGER_FIXEDLEN_VECTOR_VALUETYPE + 1));
    }

    static mvt_range fp_fixedlen_vector_valuetypes() {
      return mvt_range(
          MVT::FIRST_FP_FIXEDLEN_VECTOR_VALUETYPE,
          (MVT::SimpleValueType)(MVT::LAST_FP_FIXEDLEN_VECTOR_VALUETYPE + 1));
    }

    static mvt_range integer_scalable_vector_valuetypes() {
      return mvt_range(
       MVT::FIRST_INTEGER_SCALABLE_VECTOR_VALUETYPE,
       (MVT::SimpleValueType)(MVT::LAST_INTEGER_SCALABLE_VECTOR_VALUETYPE + 1));
    }

    static mvt_range fp_scalable_vector_valuetypes() {
      return mvt_range(
            MVT::FIRST_FP_SCALABLE_VECTOR_VALUETYPE,
            (MVT::SimpleValueType)(MVT::LAST_FP_SCALABLE_VECTOR_VALUETYPE + 1));
    }
    /// @}
  };

} // end namespace llvm

#endif // LLVM_CODEGEN_MACHINEVALUETYPE_H

//===- CodeGen/ValueTypes.h - Low-Level Target independ. types --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the set of low-level target independent types which various
// values in the code generator are.  This allows the target specific behavior
// of instructions to be described to target independent passes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_VALUETYPES_H
#define LLVM_CODEGEN_VALUETYPES_H

#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>
#include <string>

namespace llvm {
  class Type;
  class LLVMContext;
  struct EVT;

  /// MVT - Machine Value Type.  Every type that is supported natively by some
  /// processor targeted by LLVM occurs here.  This means that any legal value
  /// type can be represented by a MVT.
  class MVT {
  public:
    enum SimpleValueType {
      // If you change this numbering, you must change the values in
      // ValueTypes.td as well!
      Other          =   0,   // This is a non-standard value
      i1             =   1,   // This is a 1 bit integer value
      i8             =   2,   // This is an 8 bit integer value
      i16            =   3,   // This is a 16 bit integer value
      i32            =   4,   // This is a 32 bit integer value
      i64            =   5,   // This is a 64 bit integer value
      i128           =   6,   // This is a 128 bit integer value

      FIRST_INTEGER_VALUETYPE = i1,
      LAST_INTEGER_VALUETYPE  = i128,

      f16            =   7,   // This is a 16 bit floating point value
      f32            =   8,   // This is a 32 bit floating point value
      f64            =   9,   // This is a 64 bit floating point value
      f80            =  10,   // This is a 80 bit floating point value
      f128           =  11,   // This is a 128 bit floating point value
      ppcf128        =  12,   // This is a PPC 128-bit floating point value

      FIRST_FP_VALUETYPE = f16,
      LAST_FP_VALUETYPE  = ppcf128,

      v2i1           =  13,   //  2 x i1
      v4i1           =  14,   //  4 x i1
      v8i1           =  15,   //  8 x i1
      v16i1          =  16,   // 16 x i1
      v2i8           =  17,   //  2 x i8
      v4i8           =  18,   //  4 x i8
      v8i8           =  19,   //  8 x i8
      v16i8          =  20,   // 16 x i8
      v32i8          =  21,   // 32 x i8
      v1i16          =  22,   //  1 x i16
      v2i16          =  23,   //  2 x i16
      v4i16          =  24,   //  4 x i16
      v8i16          =  25,   //  8 x i16
      v16i16         =  26,   // 16 x i16
      v1i32          =  27,   //  1 x i32
      v2i32          =  28,   //  2 x i32
      v4i32          =  29,   //  4 x i32
      v8i32          =  30,   //  8 x i32
      v16i32         =  31,   // 16 x i32
      v1i64          =  32,   //  1 x i64
      v2i64          =  33,   //  2 x i64
      v4i64          =  34,   //  4 x i64
      v8i64          =  35,   //  8 x i64
      v16i64         =  36,   // 16 x i64

      v2f16          =  37,   //  2 x f16
      v2f32          =  38,   //  2 x f32
      v4f32          =  39,   //  4 x f32
      v8f32          =  40,   //  8 x f32
      v2f64          =  41,   //  2 x f64
      v4f64          =  42,   //  4 x f64

      FIRST_VECTOR_VALUETYPE = v2i1,
      LAST_VECTOR_VALUETYPE  = v4f64,
      FIRST_INTEGER_VECTOR_VALUETYPE = v2i1,
      LAST_INTEGER_VECTOR_VALUETYPE = v16i64,
      FIRST_FP_VECTOR_VALUETYPE = v2f16,
      LAST_FP_VECTOR_VALUETYPE = v4f64,

      x86mmx         =  43,   // This is an X86 MMX value

      Glue           =  44,   // This glues nodes together during pre-RA sched

      isVoid         =  45,   // This has no value

      Untyped        =  46,   // This value takes a register, but has
                              // unspecified type.  The register class
                              // will be determined by the opcode.

      LAST_VALUETYPE =  47,   // This always remains at the end of the list.

      // This is the current maximum for LAST_VALUETYPE.
      // MVT::MAX_ALLOWED_VALUETYPE is used for asserts and to size bit vectors
      // This value must be a multiple of 32.
      MAX_ALLOWED_VALUETYPE = 64,

      // Metadata - This is MDNode or MDString.
      Metadata       = 250,

      // iPTRAny - An int value the size of the pointer of the current
      // target to any address space. This must only be used internal to
      // tblgen. Other than for overloading, we treat iPTRAny the same as iPTR.
      iPTRAny        = 251,

      // vAny - A vector with any length and element size. This is used
      // for intrinsics that have overloadings based on vector types.
      // This is only for tblgen's consumption!
      vAny           = 252,

      // fAny - Any floating-point or vector floating-point value. This is used
      // for intrinsics that have overloadings based on floating-point types.
      // This is only for tblgen's consumption!
      fAny           = 253,

      // iAny - An integer or vector integer value of any bit width. This is
      // used for intrinsics that have overloadings based on integer bit widths.
      // This is only for tblgen's consumption!
      iAny           = 254,

      // iPTR - An int value the size of the pointer of the current
      // target.  This should only be used internal to tblgen!
      iPTR           = 255,

      // LastSimpleValueType - The greatest valid SimpleValueType value.
      LastSimpleValueType = 255,

      // INVALID_SIMPLE_VALUE_TYPE - Simple value types greater than or equal
      // to this are considered extended value types.
      INVALID_SIMPLE_VALUE_TYPE = LastSimpleValueType + 1
    };

    SimpleValueType SimpleTy;

    MVT() : SimpleTy((SimpleValueType)(INVALID_SIMPLE_VALUE_TYPE)) {}
    MVT(SimpleValueType SVT) : SimpleTy(SVT) { }

    bool operator>(const MVT& S)  const { return SimpleTy >  S.SimpleTy; }
    bool operator<(const MVT& S)  const { return SimpleTy <  S.SimpleTy; }
    bool operator==(const MVT& S) const { return SimpleTy == S.SimpleTy; }
    bool operator!=(const MVT& S) const { return SimpleTy != S.SimpleTy; }
    bool operator>=(const MVT& S) const { return SimpleTy >= S.SimpleTy; }
    bool operator<=(const MVT& S) const { return SimpleTy <= S.SimpleTy; }

    /// isFloatingPoint - Return true if this is a FP, or a vector FP type.
    bool isFloatingPoint() const {
      return ((SimpleTy >= MVT::FIRST_FP_VALUETYPE &&
               SimpleTy <= MVT::LAST_FP_VALUETYPE) ||
              (SimpleTy >= MVT::FIRST_FP_VECTOR_VALUETYPE &&
               SimpleTy <= MVT::LAST_FP_VECTOR_VALUETYPE));
    }

    /// isInteger - Return true if this is an integer, or a vector integer type.
    bool isInteger() const {
      return ((SimpleTy >= MVT::FIRST_INTEGER_VALUETYPE &&
               SimpleTy <= MVT::LAST_INTEGER_VALUETYPE) ||
              (SimpleTy >= MVT::FIRST_INTEGER_VECTOR_VALUETYPE &&
               SimpleTy <= MVT::LAST_INTEGER_VECTOR_VALUETYPE));
    }

    /// isVector - Return true if this is a vector value type.
    bool isVector() const {
      return (SimpleTy >= MVT::FIRST_VECTOR_VALUETYPE &&
              SimpleTy <= MVT::LAST_VECTOR_VALUETYPE);
    }

    /// is16BitVector - Return true if this is a 16-bit vector type.
    bool is16BitVector() const {
      return (SimpleTy == MVT::v2i8  || SimpleTy == MVT::v1i16 ||
              SimpleTy == MVT::v16i1);
    }

    /// is32BitVector - Return true if this is a 32-bit vector type.
    bool is32BitVector() const {
      return (SimpleTy == MVT::v4i8  || SimpleTy == MVT::v2i16 ||
              SimpleTy == MVT::v1i32);
    }

    /// is64BitVector - Return true if this is a 64-bit vector type.
    bool is64BitVector() const {
      return (SimpleTy == MVT::v8i8  || SimpleTy == MVT::v4i16 ||
              SimpleTy == MVT::v2i32 || SimpleTy == MVT::v1i64 ||
              SimpleTy == MVT::v2f32);
    }

    /// is128BitVector - Return true if this is a 128-bit vector type.
    bool is128BitVector() const {
      return (SimpleTy == MVT::v16i8 || SimpleTy == MVT::v8i16 ||
              SimpleTy == MVT::v4i32 || SimpleTy == MVT::v2i64 ||
              SimpleTy == MVT::v4f32 || SimpleTy == MVT::v2f64);
    }

    /// is256BitVector - Return true if this is a 256-bit vector type.
    bool is256BitVector() const {
      return (SimpleTy == MVT::v8f32 || SimpleTy == MVT::v4f64  ||
              SimpleTy == MVT::v32i8 || SimpleTy == MVT::v16i16 ||
              SimpleTy == MVT::v8i32 || SimpleTy == MVT::v4i64);
    }

    /// is512BitVector - Return true if this is a 512-bit vector type.
    bool is512BitVector() const {
      return (SimpleTy == MVT::v8i64 || SimpleTy == MVT::v16i32);
    }

    /// is1024BitVector - Return true if this is a 1024-bit vector type.
    bool is1024BitVector() const {
      return (SimpleTy == MVT::v16i64);
    }

    /// isPow2VectorType - Returns true if the given vector is a power of 2.
    bool isPow2VectorType() const {
      unsigned NElts = getVectorNumElements();
      return !(NElts & (NElts - 1));
    }

    /// getPow2VectorType - Widens the length of the given vector MVT up to
    /// the nearest power of 2 and returns that type.
    MVT getPow2VectorType() const {
      if (isPow2VectorType())
        return *this;

      unsigned NElts = getVectorNumElements();
      unsigned Pow2NElts = 1 << Log2_32_Ceil(NElts);
      return MVT::getVectorVT(getVectorElementType(), Pow2NElts);
    }

    /// getScalarType - If this is a vector type, return the element type,
    /// otherwise return this.
    MVT getScalarType() const {
      return isVector() ? getVectorElementType() : *this;
    }

    MVT getVectorElementType() const {
      switch (SimpleTy) {
      default:
        llvm_unreachable("Not a vector MVT!");
      case v2i1 :
      case v4i1 :
      case v8i1 :
      case v16i1: return i1;
      case v2i8 :
      case v4i8 :
      case v8i8 :
      case v16i8:
      case v32i8: return i8;
      case v1i16:
      case v2i16:
      case v4i16:
      case v8i16:
      case v16i16: return i16;
      case v1i32:
      case v2i32:
      case v4i32:
      case v8i32:
      case v16i32: return i32;
      case v1i64:
      case v2i64:
      case v4i64:
      case v8i64:
      case v16i64: return i64;
      case v2f16: return f16;
      case v2f32:
      case v4f32:
      case v8f32: return f32;
      case v2f64:
      case v4f64: return f64;
      }
    }

    unsigned getVectorNumElements() const {
      switch (SimpleTy) {
      default:
        llvm_unreachable("Not a vector MVT!");
      case v32i8: return 32;
      case v16i1:
      case v16i8:
      case v16i16:
      case v16i32:
      case v16i64:return 16;
      case v8i1:
      case v8i8 :
      case v8i16:
      case v8i32:
      case v8i64:
      case v8f32: return 8;
      case v4i1:
      case v4i8:
      case v4i16:
      case v4i32:
      case v4i64:
      case v4f32:
      case v4f64: return 4;
      case v2i1:
      case v2i8:
      case v2i16:
      case v2i32:
      case v2i64:
      case v2f16:
      case v2f32:
      case v2f64: return 2;
      case v1i16:
      case v1i32:
      case v1i64: return 1;
      }
    }

    unsigned getSizeInBits() const {
      switch (SimpleTy) {
      case iPTR:
        llvm_unreachable("Value type size is target-dependent. Ask TLI.");
      case iPTRAny:
      case iAny:
      case fAny:
        llvm_unreachable("Value type is overloaded.");
      default:
        llvm_unreachable("getSizeInBits called on extended MVT.");
      case i1  :  return 1;
      case v2i1:  return 2;
      case v4i1:  return 4;
      case i8  :
      case v8i1: return 8;
      case i16 :
      case f16:
      case v16i1:
      case v2i8:
      case v1i16: return 16;
      case f32 :
      case i32 :
      case v4i8:
      case v2i16:
      case v2f16: 
      case v1i32: return 32;
      case x86mmx:
      case f64 :
      case i64 :
      case v8i8:
      case v4i16:
      case v2i32:
      case v1i64:
      case v2f32: return 64;
      case f80 :  return 80;
      case f128:
      case ppcf128:
      case i128:
      case v16i8:
      case v8i16:
      case v4i32:
      case v2i64:
      case v4f32:
      case v2f64: return 128;
      case v32i8:
      case v16i16:
      case v8i32:
      case v4i64:
      case v8f32:
      case v4f64: return 256;
      case v16i32:
      case v8i64: return 512;
      case v16i64:return 1024;
      }
    }

    /// getStoreSize - Return the number of bytes overwritten by a store
    /// of the specified value type.
    unsigned getStoreSize() const {
      return (getSizeInBits() + 7) / 8;
    }

    /// getStoreSizeInBits - Return the number of bits overwritten by a store
    /// of the specified value type.
    unsigned getStoreSizeInBits() const {
      return getStoreSize() * 8;
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
        if (NumElements == 2)  return MVT::v2i1;
        if (NumElements == 4)  return MVT::v4i1;
        if (NumElements == 8)  return MVT::v8i1;
        if (NumElements == 16) return MVT::v16i1;
        break;
      case MVT::i8:
        if (NumElements == 2)  return MVT::v2i8;
        if (NumElements == 4)  return MVT::v4i8;
        if (NumElements == 8)  return MVT::v8i8;
        if (NumElements == 16) return MVT::v16i8;
        if (NumElements == 32) return MVT::v32i8;
        break;
      case MVT::i16:
        if (NumElements == 1)  return MVT::v1i16;
        if (NumElements == 2)  return MVT::v2i16;
        if (NumElements == 4)  return MVT::v4i16;
        if (NumElements == 8)  return MVT::v8i16;
        if (NumElements == 16) return MVT::v16i16;
        break;
      case MVT::i32:
        if (NumElements == 1)  return MVT::v1i32;
        if (NumElements == 2)  return MVT::v2i32;
        if (NumElements == 4)  return MVT::v4i32;
        if (NumElements == 8)  return MVT::v8i32;
        if (NumElements == 16) return MVT::v16i32;
        break;
      case MVT::i64:
        if (NumElements == 1)  return MVT::v1i64;
        if (NumElements == 2)  return MVT::v2i64;
        if (NumElements == 4)  return MVT::v4i64;
        if (NumElements == 8)  return MVT::v8i64;
        if (NumElements == 16) return MVT::v16i64;
        break;
      case MVT::f16:
        if (NumElements == 2)  return MVT::v2f16;
        break;
      case MVT::f32:
        if (NumElements == 2)  return MVT::v2f32;
        if (NumElements == 4)  return MVT::v4f32;
        if (NumElements == 8)  return MVT::v8f32;
        break;
      case MVT::f64:
        if (NumElements == 2)  return MVT::v2f64;
        if (NumElements == 4)  return MVT::v4f64;
        break;
      }
      return (MVT::SimpleValueType)(MVT::INVALID_SIMPLE_VALUE_TYPE);
    }
  };


  /// EVT - Extended Value Type.  Capable of holding value types which are not
  /// native for any processor (such as the i12345 type), as well as the types
  /// a MVT can represent.
  struct EVT {
  private:
    MVT V;
    Type *LLVMTy;

  public:
    EVT() : V((MVT::SimpleValueType)(MVT::INVALID_SIMPLE_VALUE_TYPE)),
            LLVMTy(0) {}
    EVT(MVT::SimpleValueType SVT) : V(SVT), LLVMTy(0) { }
    EVT(MVT S) : V(S), LLVMTy(0) {}

    bool operator==(EVT VT) const {
      return !(*this != VT);
    }
    bool operator!=(EVT VT) const {
      if (V.SimpleTy != VT.V.SimpleTy)
        return true;
      if (V.SimpleTy == MVT::INVALID_SIMPLE_VALUE_TYPE)
        return LLVMTy != VT.LLVMTy;
      return false;
    }

    /// getFloatingPointVT - Returns the EVT that represents a floating point
    /// type with the given number of bits.  There are two floating point types
    /// with 128 bits - this returns f128 rather than ppcf128.
    static EVT getFloatingPointVT(unsigned BitWidth) {
      return MVT::getFloatingPointVT(BitWidth);
    }

    /// getIntegerVT - Returns the EVT that represents an integer with the given
    /// number of bits.
    static EVT getIntegerVT(LLVMContext &Context, unsigned BitWidth) {
      MVT M = MVT::getIntegerVT(BitWidth);
      if (M.SimpleTy != MVT::INVALID_SIMPLE_VALUE_TYPE)
        return M;
      return getExtendedIntegerVT(Context, BitWidth);
    }

    /// getVectorVT - Returns the EVT that represents a vector NumElements in
    /// length, where each element is of type VT.
    static EVT getVectorVT(LLVMContext &Context, EVT VT, unsigned NumElements) {
      MVT M = MVT::getVectorVT(VT.V, NumElements);
      if (M.SimpleTy != MVT::INVALID_SIMPLE_VALUE_TYPE)
        return M;
      return getExtendedVectorVT(Context, VT, NumElements);
    }

    /// changeVectorElementTypeToInteger - Return a vector with the same number
    /// of elements as this vector, but with the element type converted to an
    /// integer type with the same bitwidth.
    EVT changeVectorElementTypeToInteger() const {
      if (!isSimple())
        return changeExtendedVectorElementTypeToInteger();
      MVT EltTy = getSimpleVT().getVectorElementType();
      unsigned BitWidth = EltTy.getSizeInBits();
      MVT IntTy = MVT::getIntegerVT(BitWidth);
      MVT VecTy = MVT::getVectorVT(IntTy, getVectorNumElements());
      assert(VecTy != MVT::INVALID_SIMPLE_VALUE_TYPE &&
             "Simple vector VT not representable by simple integer vector VT!");
      return VecTy;
    }

    /// isSimple - Test if the given EVT is simple (as opposed to being
    /// extended).
    bool isSimple() const {
      return V.SimpleTy <= MVT::LastSimpleValueType;
    }

    /// isExtended - Test if the given EVT is extended (as opposed to
    /// being simple).
    bool isExtended() const {
      return !isSimple();
    }

    /// isFloatingPoint - Return true if this is a FP, or a vector FP type.
    bool isFloatingPoint() const {
      return isSimple() ? V.isFloatingPoint() : isExtendedFloatingPoint();
    }

    /// isInteger - Return true if this is an integer, or a vector integer type.
    bool isInteger() const {
      return isSimple() ? V.isInteger() : isExtendedInteger();
    }

    /// isVector - Return true if this is a vector value type.
    bool isVector() const {
      return isSimple() ? V.isVector() : isExtendedVector();
    }

    /// is16BitVector - Return true if this is a 16-bit vector type.
    bool is16BitVector() const {
      return isSimple() ? V.is16BitVector() : isExtended16BitVector();
    }

    /// is32BitVector - Return true if this is a 32-bit vector type.
    bool is32BitVector() const {
      return isSimple() ? V.is32BitVector() : isExtended32BitVector();
    }

    /// is64BitVector - Return true if this is a 64-bit vector type.
    bool is64BitVector() const {
      return isSimple() ? V.is64BitVector() : isExtended64BitVector();
    }

    /// is128BitVector - Return true if this is a 128-bit vector type.
    bool is128BitVector() const {
      return isSimple() ? V.is128BitVector() : isExtended128BitVector();
    }

    /// is256BitVector - Return true if this is a 256-bit vector type.
    bool is256BitVector() const {
      return isSimple() ? V.is256BitVector() : isExtended256BitVector();
    }

    /// is512BitVector - Return true if this is a 512-bit vector type.
    bool is512BitVector() const {
      return isSimple() ? V.is512BitVector() : isExtended512BitVector();
    }

    /// is1024BitVector - Return true if this is a 1024-bit vector type.
    bool is1024BitVector() const {
      return isSimple() ? V.is1024BitVector() : isExtended1024BitVector();
    }

    /// isOverloaded - Return true if this is an overloaded type for TableGen.
    bool isOverloaded() const {
      return (V==MVT::iAny || V==MVT::fAny || V==MVT::vAny || V==MVT::iPTRAny);
    }

    /// isByteSized - Return true if the bit size is a multiple of 8.
    bool isByteSized() const {
      return (getSizeInBits() & 7) == 0;
    }

    /// isRound - Return true if the size is a power-of-two number of bytes.
    bool isRound() const {
      unsigned BitSize = getSizeInBits();
      return BitSize >= 8 && !(BitSize & (BitSize - 1));
    }

    /// bitsEq - Return true if this has the same number of bits as VT.
    bool bitsEq(EVT VT) const {
      if (EVT::operator==(VT)) return true;
      return getSizeInBits() == VT.getSizeInBits();
    }

    /// bitsGT - Return true if this has more bits than VT.
    bool bitsGT(EVT VT) const {
      if (EVT::operator==(VT)) return false;
      return getSizeInBits() > VT.getSizeInBits();
    }

    /// bitsGE - Return true if this has no less bits than VT.
    bool bitsGE(EVT VT) const {
      if (EVT::operator==(VT)) return true;
      return getSizeInBits() >= VT.getSizeInBits();
    }

    /// bitsLT - Return true if this has less bits than VT.
    bool bitsLT(EVT VT) const {
      if (EVT::operator==(VT)) return false;
      return getSizeInBits() < VT.getSizeInBits();
    }

    /// bitsLE - Return true if this has no more bits than VT.
    bool bitsLE(EVT VT) const {
      if (EVT::operator==(VT)) return true;
      return getSizeInBits() <= VT.getSizeInBits();
    }


    /// getSimpleVT - Return the SimpleValueType held in the specified
    /// simple EVT.
    MVT getSimpleVT() const {
      assert(isSimple() && "Expected a SimpleValueType!");
      return V;
    }

    /// getScalarType - If this is a vector type, return the element type,
    /// otherwise return this.
    EVT getScalarType() const {
      return isVector() ? getVectorElementType() : *this;
    }

    /// getVectorElementType - Given a vector type, return the type of
    /// each element.
    EVT getVectorElementType() const {
      assert(isVector() && "Invalid vector type!");
      if (isSimple())
        return V.getVectorElementType();
      return getExtendedVectorElementType();
    }

    /// getVectorNumElements - Given a vector type, return the number of
    /// elements it contains.
    unsigned getVectorNumElements() const {
      assert(isVector() && "Invalid vector type!");
      if (isSimple())
        return V.getVectorNumElements();
      return getExtendedVectorNumElements();
    }

    /// getSizeInBits - Return the size of the specified value type in bits.
    unsigned getSizeInBits() const {
      if (isSimple())
        return V.getSizeInBits();
      return getExtendedSizeInBits();
    }

    /// getStoreSize - Return the number of bytes overwritten by a store
    /// of the specified value type.
    unsigned getStoreSize() const {
      return (getSizeInBits() + 7) / 8;
    }

    /// getStoreSizeInBits - Return the number of bits overwritten by a store
    /// of the specified value type.
    unsigned getStoreSizeInBits() const {
      return getStoreSize() * 8;
    }

    /// getRoundIntegerType - Rounds the bit-width of the given integer EVT up
    /// to the nearest power of two (and at least to eight), and returns the
    /// integer EVT with that number of bits.
    EVT getRoundIntegerType(LLVMContext &Context) const {
      assert(isInteger() && !isVector() && "Invalid integer type!");
      unsigned BitWidth = getSizeInBits();
      if (BitWidth <= 8)
        return EVT(MVT::i8);
      return getIntegerVT(Context, 1 << Log2_32_Ceil(BitWidth));
    }

    /// getHalfSizedIntegerVT - Finds the smallest simple value type that is
    /// greater than or equal to half the width of this EVT. If no simple
    /// value type can be found, an extended integer value type of half the
    /// size (rounded up) is returned.
    EVT getHalfSizedIntegerVT(LLVMContext &Context) const {
      assert(isInteger() && !isVector() && "Invalid integer type!");
      unsigned EVTSize = getSizeInBits();
      for (unsigned IntVT = MVT::FIRST_INTEGER_VALUETYPE;
          IntVT <= MVT::LAST_INTEGER_VALUETYPE; ++IntVT) {
        EVT HalfVT = EVT((MVT::SimpleValueType)IntVT);
        if (HalfVT.getSizeInBits() * 2 >= EVTSize)
          return HalfVT;
      }
      return getIntegerVT(Context, (EVTSize + 1) / 2);
    }

    /// isPow2VectorType - Returns true if the given vector is a power of 2.
    bool isPow2VectorType() const {
      unsigned NElts = getVectorNumElements();
      return !(NElts & (NElts - 1));
    }

    /// getPow2VectorType - Widens the length of the given vector EVT up to
    /// the nearest power of 2 and returns that type.
    EVT getPow2VectorType(LLVMContext &Context) const {
      if (!isPow2VectorType()) {
        unsigned NElts = getVectorNumElements();
        unsigned Pow2NElts = 1 <<  Log2_32_Ceil(NElts);
        return EVT::getVectorVT(Context, getVectorElementType(), Pow2NElts);
      }
      else {
        return *this;
      }
    }

    /// getEVTString - This function returns value type as a string,
    /// e.g. "i32".
    std::string getEVTString() const;

    /// getTypeForEVT - This method returns an LLVM type corresponding to the
    /// specified EVT.  For integer types, this returns an unsigned type.  Note
    /// that this will abort for types that cannot be represented.
    Type *getTypeForEVT(LLVMContext &Context) const;

    /// getEVT - Return the value type corresponding to the specified type.
    /// This returns all pointers as iPTR.  If HandleUnknown is true, unknown
    /// types are returned as Other, otherwise they are invalid.
    static EVT getEVT(Type *Ty, bool HandleUnknown = false);

    intptr_t getRawBits() {
      if (isSimple())
        return V.SimpleTy;
      else
        return (intptr_t)(LLVMTy);
    }

    /// compareRawBits - A meaningless but well-behaved order, useful for
    /// constructing containers.
    struct compareRawBits {
      bool operator()(EVT L, EVT R) const {
        if (L.V.SimpleTy == R.V.SimpleTy)
          return L.LLVMTy < R.LLVMTy;
        else
          return L.V.SimpleTy < R.V.SimpleTy;
      }
    };

  private:
    // Methods for handling the Extended-type case in functions above.
    // These are all out-of-line to prevent users of this header file
    // from having a dependency on Type.h.
    EVT changeExtendedVectorElementTypeToInteger() const;
    static EVT getExtendedIntegerVT(LLVMContext &C, unsigned BitWidth);
    static EVT getExtendedVectorVT(LLVMContext &C, EVT VT,
                                   unsigned NumElements);
    bool isExtendedFloatingPoint() const;
    bool isExtendedInteger() const;
    bool isExtendedVector() const;
    bool isExtended16BitVector() const;
    bool isExtended32BitVector() const;
    bool isExtended64BitVector() const;
    bool isExtended128BitVector() const;
    bool isExtended256BitVector() const;
    bool isExtended512BitVector() const;
    bool isExtended1024BitVector() const;
    EVT getExtendedVectorElementType() const;
    unsigned getExtendedVectorNumElements() const;
    unsigned getExtendedSizeInBits() const;
  };

} // End llvm namespace

#endif

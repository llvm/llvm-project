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

#include <cassert>
#include <string>
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/MathExtras.h"

namespace llvm {
  class Type;

  struct MVT { // MVT = Machine Value Type
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

      f32            =   7,   // This is a 32 bit floating point value
      f64            =   8,   // This is a 64 bit floating point value
      f80            =   9,   // This is a 80 bit floating point value
      f128           =  10,   // This is a 128 bit floating point value
      ppcf128        =  11,   // This is a PPC 128-bit floating point value
      Flag           =  12,   // This is a condition code or machine flag.

      isVoid         =  13,   // This has no value

      v2i8           =  14,   //  2 x i8
      v4i8           =  15,   //  4 x i8
      v2i16          =  16,   //  2 x i16
      v8i8           =  17,   //  8 x i8
      v4i16          =  18,   //  4 x i16
      v2i32          =  19,   //  2 x i32
      v1i64          =  20,   //  1 x i64
      v16i8          =  21,   // 16 x i8
      v8i16          =  22,   //  8 x i16
      v3i32          =  23,   //  3 x i32
      v4i32          =  24,   //  4 x i32
      v2i64          =  25,   //  2 x i64

      v2f32          =  26,   //  2 x f32
      v3f32          =  27,   //  3 x f32
      v4f32          =  28,   //  4 x f32
      v2f64          =  29,   //  2 x f64

      FIRST_VECTOR_VALUETYPE = v2i8,
      LAST_VECTOR_VALUETYPE  = v2f64,

      LAST_VALUETYPE =  30,   // This always remains at the end of the list.

      // iPTRAny - An int value the size of the pointer of the current
      // target to any address space. This must only be used internal to
      // tblgen. Other than for overloading, we treat iPTRAny the same as iPTR.
      iPTRAny        =  252,

      // fAny - Any floating-point or vector floating-point value. This is used
      // for intrinsics that have overloadings based on floating-point types.
      // This is only for tblgen's consumption!
      fAny           =  253,

      // iAny - An integer or vector integer value of any bit width. This is
      // used for intrinsics that have overloadings based on integer bit widths.
      // This is only for tblgen's consumption!
      iAny           =  254,

      // iPTR - An int value the size of the pointer of the current
      // target.  This should only be used internal to tblgen!
      iPTR           =  255,

      // LastSimpleValueType - The greatest valid SimpleValueType value.
      LastSimpleValueType = 255
    };

  private:
    /// This union holds low-level value types. Valid values include any of
    /// the values in the SimpleValueType enum, or any value returned from one
    /// of the MVT methods.  Any value type equal to one of the SimpleValueType
    /// enum values is a "simple" value type.  All others are "extended".
    ///
    /// Note that simple doesn't necessary mean legal for the target machine.
    /// All legal value types must be simple, but often there are some simple
    /// value types that are not legal.
    ///
    union {
      uintptr_t V;
      SimpleValueType SimpleTy;
      const Type *LLVMTy;
    };

  public:
    MVT() {}
    MVT(SimpleValueType S) : V(S) {}

    bool operator==(const MVT VT) const {
      return getRawBits() == VT.getRawBits();
    }
    bool operator!=(const MVT VT) const {
      return getRawBits() != VT.getRawBits();
    }

    /// getFloatingPointVT - Returns the MVT that represents a floating point
    /// type with the given number of bits.  There are two floating point types
    /// with 128 bits - this returns f128 rather than ppcf128.
    static MVT getFloatingPointVT(unsigned BitWidth) {
      switch (BitWidth) {
      default:
        assert(false && "Bad bit width!");
      case 32:
        return f32;
      case 64:
        return f64;
      case 80:
        return f80;
      case 128:
        return f128;
      }
    }

    /// getIntegerVT - Returns the MVT that represents an integer with the given
    /// number of bits.
    static MVT getIntegerVT(unsigned BitWidth) {
      switch (BitWidth) {
      default:
        break;
      case 1:
        return i1;
      case 8:
        return i8;
      case 16:
        return i16;
      case 32:
        return i32;
      case 64:
        return i64;
      case 128:
        return i128;
      }
      return getExtendedIntegerVT(BitWidth);
    }

    /// getVectorVT - Returns the MVT that represents a vector NumElements in
    /// length, where each element is of type VT.
    static MVT getVectorVT(MVT VT, unsigned NumElements) {
      switch (VT.V) {
      default:
        break;
      case i8:
        if (NumElements == 2)  return v2i8;
        if (NumElements == 4)  return v4i8;
        if (NumElements == 8)  return v8i8;
        if (NumElements == 16) return v16i8;
        break;
      case i16:
        if (NumElements == 2)  return v2i16;
        if (NumElements == 4)  return v4i16;
        if (NumElements == 8)  return v8i16;
        break;
      case i32:
        if (NumElements == 2)  return v2i32;
        if (NumElements == 3)  return v3i32;
        if (NumElements == 4)  return v4i32;
        break;
      case i64:
        if (NumElements == 1)  return v1i64;
        if (NumElements == 2)  return v2i64;
        break;
      case f32:
        if (NumElements == 2)  return v2f32;
        if (NumElements == 3)  return v3f32;
        if (NumElements == 4)  return v4f32;
        break;
      case f64:
        if (NumElements == 2)  return v2f64;
        break;
      }
      return getExtendedVectorVT(VT, NumElements);
    }

    /// getIntVectorWithNumElements - Return any integer vector type that has
    /// the specified number of elements.
    static MVT getIntVectorWithNumElements(unsigned NumElts) {
      switch (NumElts) {
      default: return getVectorVT(i8, NumElts);
      case  1: return v1i64;
      case  2: return v2i32;
      case  3: return v3i32;
      case  4: return v4i16;
      case  8: return v8i8;
      case 16: return v16i8;
      }
    }

    /// isSimple - Test if the given MVT is simple (as opposed to being
    /// extended).
    bool isSimple() const {
      return V <= LastSimpleValueType;
    }

    /// isExtended - Test if the given MVT is extended (as opposed to
    /// being simple).
    bool isExtended() const {
      return !isSimple();
    }

    /// isFloatingPoint - Return true if this is a FP, or a vector FP type.
    bool isFloatingPoint() const {
      return isSimple() ?
             ((SimpleTy >= f32 && SimpleTy <= ppcf128) ||
              (SimpleTy >= v2f32 && SimpleTy <= v2f64)) :
             isExtendedFloatingPoint();
    }

    /// isInteger - Return true if this is an integer, or a vector integer type.
    bool isInteger() const {
      return isSimple() ?
             ((SimpleTy >= FIRST_INTEGER_VALUETYPE &&
               SimpleTy <= LAST_INTEGER_VALUETYPE) ||
              (SimpleTy >= v2i8 && SimpleTy <= v2i64)) :
             isExtendedInteger();
    }

    /// isVector - Return true if this is a vector value type.
    bool isVector() const {
      return isSimple() ?
             (SimpleTy >= FIRST_VECTOR_VALUETYPE &&
              SimpleTy <= LAST_VECTOR_VALUETYPE) :
             isExtendedVector();
    }

    /// is64BitVector - Return true if this is a 64-bit vector type.
    bool is64BitVector() const {
      return isSimple() ?
             (SimpleTy==v8i8 || SimpleTy==v4i16 || SimpleTy==v2i32 ||
              SimpleTy==v1i64 || SimpleTy==v2f32) :
             isExtended64BitVector();
    }

    /// is128BitVector - Return true if this is a 128-bit vector type.
    bool is128BitVector() const {
      return isSimple() ?
             (SimpleTy==v16i8 || SimpleTy==v8i16 || SimpleTy==v4i32 ||
              SimpleTy==v2i64 || SimpleTy==v4f32 || SimpleTy==v2f64) :
             isExtended128BitVector();
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
    bool bitsEq(MVT VT) const {
      return getSizeInBits() == VT.getSizeInBits();
    }

    /// bitsGT - Return true if this has more bits than VT.
    bool bitsGT(MVT VT) const {
      return getSizeInBits() > VT.getSizeInBits();
    }

    /// bitsGE - Return true if this has no less bits than VT.
    bool bitsGE(MVT VT) const {
      return getSizeInBits() >= VT.getSizeInBits();
    }

    /// bitsLT - Return true if this has less bits than VT.
    bool bitsLT(MVT VT) const {
      return getSizeInBits() < VT.getSizeInBits();
    }

    /// bitsLE - Return true if this has no more bits than VT.
    bool bitsLE(MVT VT) const {
      return getSizeInBits() <= VT.getSizeInBits();
    }


    /// getSimpleVT - Return the SimpleValueType held in the specified
    /// simple MVT.
    SimpleValueType getSimpleVT() const {
      assert(isSimple() && "Expected a SimpleValueType!");
      return SimpleTy;
    }

    /// getVectorElementType - Given a vector type, return the type of
    /// each element.
    MVT getVectorElementType() const {
      assert(isVector() && "Invalid vector type!");
      switch (V) {
      default:
        return getExtendedVectorElementType();
      case v2i8 :
      case v4i8 :
      case v8i8 :
      case v16i8: return i8;
      case v2i16:
      case v4i16:
      case v8i16: return i16;
      case v2i32:
      case v3i32:
      case v4i32: return i32;
      case v1i64:
      case v2i64: return i64;
      case v2f32:
      case v3f32:
      case v4f32: return f32;
      case v2f64: return f64;
      }
    }

    /// getVectorNumElements - Given a vector type, return the number of
    /// elements it contains.
    unsigned getVectorNumElements() const {
      assert(isVector() && "Invalid vector type!");
      switch (V) {
      default:
        return getExtendedVectorNumElements();
      case v16i8: return 16;
      case v8i8 :
      case v8i16: return 8;
      case v4i8:
      case v4i16:
      case v4i32:
      case v4f32: return 4;
      case v3i32:
      case v3f32: return 3;
      case v2i8:
      case v2i16:
      case v2i32:
      case v2i64:
      case v2f32:
      case v2f64: return 2;
      case v1i64: return 1;
      }
    }

    /// getSizeInBits - Return the size of the specified value type in bits.
    unsigned getSizeInBits() const {
      switch (V) {
      case iPTR:
        assert(0 && "Value type size is target-dependent. Ask TLI.");
      case iPTRAny:
      case iAny:
      case fAny:
        assert(0 && "Value type is overloaded.");
      default:
        return getExtendedSizeInBits();
      case i1  :  return 1;
      case i8  :  return 8;
      case i16 :
      case v2i8:  return 16;
      case f32 :
      case i32 :
      case v4i8:
      case v2i16: return 32;
      case f64 :
      case i64 :
      case v8i8:
      case v4i16:
      case v2i32:
      case v1i64:
      case v2f32: return 64;
      case f80 :  return 80;
      case v3i32:
      case v3f32: return 96;
      case f128:
      case ppcf128:
      case i128:
      case v16i8:
      case v8i16:
      case v4i32:
      case v2i64:
      case v4f32:
      case v2f64: return 128;
      }
    }

    /// getStoreSizeInBits - Return the number of bits overwritten by a store
    /// of the specified value type.
    unsigned getStoreSizeInBits() const {
      return (getSizeInBits() + 7)/8*8;
    }

    /// getRoundIntegerType - Rounds the bit-width of the given integer MVT up
    /// to the nearest power of two (and at least to eight), and returns the
    /// integer MVT with that number of bits.
    MVT getRoundIntegerType() const {
      assert(isInteger() && !isVector() && "Invalid integer type!");
      unsigned BitWidth = getSizeInBits();
      if (BitWidth <= 8)
        return i8;
      else
        return getIntegerVT(1 << Log2_32_Ceil(BitWidth));
    }

    /// isPow2VectorType - Retuns true if the given vector is a power of 2.
    bool isPow2VectorType() const {
      unsigned NElts = getVectorNumElements();
      return !(NElts & (NElts - 1));
    }

    /// getPow2VectorType - Widens the length of the given vector MVT up to
    /// the nearest power of 2 and returns that type.
    MVT getPow2VectorType() const {
      if (!isPow2VectorType()) {
        unsigned NElts = getVectorNumElements();
        unsigned Pow2NElts = 1 <<  Log2_32_Ceil(NElts);
        return MVT::getVectorVT(getVectorElementType(), Pow2NElts);
      }
      else {
        return *this;
      }
    }

    /// getMVTString - This function returns value type as a string,
    /// e.g. "i32".
    std::string getMVTString() const;

    /// getTypeForMVT - This method returns an LLVM type corresponding to the
    /// specified MVT.  For integer types, this returns an unsigned type.  Note
    /// that this will abort for types that cannot be represented.
    const Type *getTypeForMVT() const;

    /// getMVT - Return the value type corresponding to the specified type.
    /// This returns all pointers as iPTR.  If HandleUnknown is true, unknown
    /// types are returned as Other, otherwise they are invalid.
    static MVT getMVT(const Type *Ty, bool HandleUnknown = false);

    /// getRawBits - Represent the type as a bunch of bits.
    uintptr_t getRawBits() const { return V; }

    /// compareRawBits - A meaningless but well-behaved order, useful for
    /// constructing containers.
    struct compareRawBits {
      bool operator()(MVT L, MVT R) const {
        return L.getRawBits() < R.getRawBits();
      }
    };

  private:
    // Methods for handling the Extended-type case in functions above.
    // These are all out-of-line to prevent users of this header file
    // from having a dependency on Type.h.
    static MVT getExtendedIntegerVT(unsigned BitWidth);
    static MVT getExtendedVectorVT(MVT VT, unsigned NumElements);
    bool isExtendedFloatingPoint() const;
    bool isExtendedInteger() const;
    bool isExtendedVector() const;
    bool isExtended64BitVector() const;
    bool isExtended128BitVector() const;
    MVT getExtendedVectorElementType() const;
    unsigned getExtendedVectorNumElements() const;
    unsigned getExtendedSizeInBits() const;
  };

} // End llvm namespace

#endif

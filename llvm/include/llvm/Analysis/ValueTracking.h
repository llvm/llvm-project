//===- llvm/Analysis/ValueTracking.h - Walk computations --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains routines that help analyze properties that chains of
// computations have.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_VALUETRACKING_H
#define LLVM_ANALYSIS_VALUETRACKING_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {
  class Value;
  class Instruction;
  class APInt;
  class DataLayout;
  class StringRef;
  class MDNode;

  /// ComputeMaskedBits - Determine which of the bits specified in Mask are
  /// known to be either zero or one and return them in the KnownZero/KnownOne
  /// bit sets.  This code only analyzes bits in Mask, in order to short-circuit
  /// processing.
  ///
  /// This function is defined on values with integer type, values with pointer
  /// type (but only if TD is non-null), and vectors of integers.  In the case
  /// where V is a vector, the mask, known zero, and known one values are the
  /// same width as the vector element, and the bit is set only if it is true
  /// for all of the elements in the vector.
  void ComputeMaskedBits(Value *V,  APInt &KnownZero, APInt &KnownOne,
                         const DataLayout *TD = 0, unsigned Depth = 0);
  void computeMaskedBitsLoad(const MDNode &Ranges, APInt &KnownZero);

  /// ComputeSignBit - Determine whether the sign bit is known to be zero or
  /// one.  Convenience wrapper around ComputeMaskedBits.
  void ComputeSignBit(Value *V, bool &KnownZero, bool &KnownOne,
                      const DataLayout *TD = 0, unsigned Depth = 0);

  /// isPowerOfTwo - Return true if the given value is known to have exactly one
  /// bit set when defined. For vectors return true if every element is known to
  /// be a power of two when defined.  Supports values with integer or pointer
  /// type and vectors of integers.  If 'OrZero' is set then returns true if the
  /// given value is either a power of two or zero.
  bool isPowerOfTwo(Value *V, const DataLayout *TD = 0, bool OrZero = false,
                    unsigned Depth = 0);

  /// isKnownNonZero - Return true if the given value is known to be non-zero
  /// when defined.  For vectors return true if every element is known to be
  /// non-zero when defined.  Supports values with integer or pointer type and
  /// vectors of integers.
  bool isKnownNonZero(Value *V, const DataLayout *TD = 0, unsigned Depth = 0);

  /// MaskedValueIsZero - Return true if 'V & Mask' is known to be zero.  We use
  /// this predicate to simplify operations downstream.  Mask is known to be
  /// zero for bits that V cannot have.
  ///
  /// This function is defined on values with integer type, values with pointer
  /// type (but only if TD is non-null), and vectors of integers.  In the case
  /// where V is a vector, the mask, known zero, and known one values are the
  /// same width as the vector element, and the bit is set only if it is true
  /// for all of the elements in the vector.
  bool MaskedValueIsZero(Value *V, const APInt &Mask, 
                         const DataLayout *TD = 0, unsigned Depth = 0);

  
  /// ComputeNumSignBits - Return the number of times the sign bit of the
  /// register is replicated into the other bits.  We know that at least 1 bit
  /// is always equal to the sign bit (itself), but other cases can give us
  /// information.  For example, immediately after an "ashr X, 2", we know that
  /// the top 3 bits are all equal to each other, so we return 3.
  ///
  /// 'Op' must have a scalar integer type.
  ///
  unsigned ComputeNumSignBits(Value *Op, const DataLayout *TD = 0,
                              unsigned Depth = 0);

  /// ComputeMultiple - This function computes the integer multiple of Base that
  /// equals V.  If successful, it returns true and returns the multiple in
  /// Multiple.  If unsuccessful, it returns false.  Also, if V can be
  /// simplified to an integer, then the simplified V is returned in Val.  Look
  /// through sext only if LookThroughSExt=true.
  bool ComputeMultiple(Value *V, unsigned Base, Value *&Multiple,
                       bool LookThroughSExt = false,
                       unsigned Depth = 0);

  /// CannotBeNegativeZero - Return true if we can prove that the specified FP 
  /// value is never equal to -0.0.
  ///
  bool CannotBeNegativeZero(const Value *V, unsigned Depth = 0);

  /// isBytewiseValue - If the specified value can be set by repeating the same
  /// byte in memory, return the i8 value that it is represented with.  This is
  /// true for all i8 values obviously, but is also true for i32 0, i32 -1,
  /// i16 0xF0F0, double 0.0 etc.  If the value can't be handled with a repeated
  /// byte store (e.g. i16 0x1234), return null.
  Value *isBytewiseValue(Value *V);
    
  /// FindInsertedValue - Given an aggregrate and an sequence of indices, see if
  /// the scalar value indexed is already around as a register, for example if
  /// it were inserted directly into the aggregrate.
  ///
  /// If InsertBefore is not null, this function will duplicate (modified)
  /// insertvalues when a part of a nested struct is extracted.
  Value *FindInsertedValue(Value *V,
                           ArrayRef<unsigned> idx_range,
                           Instruction *InsertBefore = 0);

  /// GetPointerBaseWithConstantOffset - Analyze the specified pointer to see if
  /// it can be expressed as a base pointer plus a constant offset.  Return the
  /// base and offset to the caller.
  Value *GetPointerBaseWithConstantOffset(Value *Ptr, int64_t &Offset,
                                          const DataLayout &TD);
  static inline const Value *
  GetPointerBaseWithConstantOffset(const Value *Ptr, int64_t &Offset,
                                   const DataLayout &TD) {
    return GetPointerBaseWithConstantOffset(const_cast<Value*>(Ptr), Offset,TD);
  }
  
  /// getConstantStringInfo - This function computes the length of a
  /// null-terminated C string pointed to by V.  If successful, it returns true
  /// and returns the string in Str.  If unsuccessful, it returns false.  This
  /// does not include the trailing nul character by default.  If TrimAtNul is
  /// set to false, then this returns any trailing nul characters as well as any
  /// other characters that come after it.
  bool getConstantStringInfo(const Value *V, StringRef &Str,
                             uint64_t Offset = 0, bool TrimAtNul = true);

  /// GetStringLength - If we can compute the length of the string pointed to by
  /// the specified pointer, return 'len+1'.  If we can't, return 0.
  uint64_t GetStringLength(Value *V);

  /// GetUnderlyingObject - This method strips off any GEP address adjustments
  /// and pointer casts from the specified value, returning the original object
  /// being addressed.  Note that the returned value has pointer type if the
  /// specified value does.  If the MaxLookup value is non-zero, it limits the
  /// number of instructions to be stripped off.
  Value *GetUnderlyingObject(Value *V, const DataLayout *TD = 0,
                             unsigned MaxLookup = 6);
  static inline const Value *
  GetUnderlyingObject(const Value *V, const DataLayout *TD = 0,
                      unsigned MaxLookup = 6) {
    return GetUnderlyingObject(const_cast<Value *>(V), TD, MaxLookup);
  }

  /// GetUnderlyingObjects - This method is similar to GetUnderlyingObject
  /// except that it can look through phi and select instructions and return
  /// multiple objects.
  void GetUnderlyingObjects(Value *V,
                            SmallVectorImpl<Value *> &Objects,
                            const DataLayout *TD = 0,
                            unsigned MaxLookup = 6);

  /// onlyUsedByLifetimeMarkers - Return true if the only users of this pointer
  /// are lifetime markers.
  bool onlyUsedByLifetimeMarkers(const Value *V);

  /// isSafeToSpeculativelyExecute - Return true if the instruction does not
  /// have any effects besides calculating the result and does not have
  /// undefined behavior.
  ///
  /// This method never returns true for an instruction that returns true for
  /// mayHaveSideEffects; however, this method also does some other checks in
  /// addition. It checks for undefined behavior, like dividing by zero or
  /// loading from an invalid pointer (but not for undefined results, like a
  /// shift with a shift amount larger than the width of the result). It checks
  /// for malloc and alloca because speculatively executing them might cause a
  /// memory leak. It also returns false for instructions related to control
  /// flow, specifically terminators and PHI nodes.
  ///
  /// This method only looks at the instruction itself and its operands, so if
  /// this method returns true, it is safe to move the instruction as long as
  /// the correct dominance relationships for the operands and users hold.
  /// However, this method can return true for instructions that read memory;
  /// for such instructions, moving them may change the resulting value.
  bool isSafeToSpeculativelyExecute(const Value *V,
                                    const DataLayout *TD = 0);

} // end namespace llvm

#endif

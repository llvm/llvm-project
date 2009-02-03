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

#include "llvm/Support/DataTypes.h"
#include <string>

namespace llvm {
  class Value;
  class Instruction;
  class APInt;
  class TargetData;
  
  /// ComputeMaskedBits - Determine which of the bits specified in Mask are
  /// known to be either zero or one and return them in the KnownZero/KnownOne
  /// bit sets.  This code only analyzes bits in Mask, in order to short-circuit
  /// processing.
  void ComputeMaskedBits(Value *V, const APInt &Mask, APInt &KnownZero,
                         APInt &KnownOne, TargetData *TD = 0,
                         unsigned Depth = 0);
  
  /// MaskedValueIsZero - Return true if 'V & Mask' is known to be zero.  We use
  /// this predicate to simplify operations downstream.  Mask is known to be
  /// zero for bits that V cannot have.
  bool MaskedValueIsZero(Value *V, const APInt &Mask, 
                         TargetData *TD = 0, unsigned Depth = 0);

  
  /// ComputeNumSignBits - Return the number of times the sign bit of the
  /// register is replicated into the other bits.  We know that at least 1 bit
  /// is always equal to the sign bit (itself), but other cases can give us
  /// information.  For example, immediately after an "ashr X, 2", we know that
  /// the top 3 bits are all equal to each other, so we return 3.
  ///
  /// 'Op' must have a scalar integer type.
  ///
  unsigned ComputeNumSignBits(Value *Op, TargetData *TD = 0,
                              unsigned Depth = 0);

  /// CannotBeNegativeZero - Return true if we can prove that the specified FP 
  /// value is never equal to -0.0.
  ///
  bool CannotBeNegativeZero(const Value *V, unsigned Depth = 0);

  /// FindScalarValue - Given an aggregrate and an sequence of indices, see if
  /// the scalar value indexed is already around as a register, for example if
  /// it were inserted directly into the aggregrate.
  ///
  /// If InsertBefore is not null, this function will duplicate (modified)
  /// insertvalues when a part of a nested struct is extracted.
  Value *FindInsertedValue(Value *V,
                         const unsigned *idx_begin,
                         const unsigned *idx_end,
                         Instruction *InsertBefore = 0);

  /// This is a convenience wrapper for finding values indexed by a single index
  /// only.
  inline Value *FindInsertedValue(Value *V, const unsigned Idx,
                                 Instruction *InsertBefore = 0) {
    const unsigned Idxs[1] = { Idx };
    return FindInsertedValue(V, &Idxs[0], &Idxs[1], InsertBefore);
  }
  
  /// GetConstantStringInfo - This function computes the length of a
  /// null-terminated C string pointed to by V.  If successful, it returns true
  /// and returns the string in Str.  If unsuccessful, it returns false.  If
  /// StopAtNul is set to true (the default), the returned string is truncated
  /// by a nul character in the global.  If StopAtNul is false, the nul
  /// character is included in the result string.
  bool GetConstantStringInfo(Value *V, std::string &Str, uint64_t Offset = 0,
                             bool StopAtNul = true);
} // end namespace llvm

#endif

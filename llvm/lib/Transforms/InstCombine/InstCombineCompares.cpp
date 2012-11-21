//===- InstCombineCompares.cpp --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the visitICmp and visitFCmp functions.
//
//===----------------------------------------------------------------------===//

#include "InstCombine.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Support/ConstantRange.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/PatternMatch.h"
using namespace llvm;
using namespace PatternMatch;

static ConstantInt *getOne(Constant *C) {
  return ConstantInt::get(cast<IntegerType>(C->getType()), 1);
}

/// AddOne - Add one to a ConstantInt
static Constant *AddOne(Constant *C) {
  return ConstantExpr::getAdd(C, ConstantInt::get(C->getType(), 1));
}
/// SubOne - Subtract one from a ConstantInt
static Constant *SubOne(Constant *C) {
  return ConstantExpr::getSub(C, ConstantInt::get(C->getType(), 1));
}

static ConstantInt *ExtractElement(Constant *V, Constant *Idx) {
  return cast<ConstantInt>(ConstantExpr::getExtractElement(V, Idx));
}

static bool HasAddOverflow(ConstantInt *Result,
                           ConstantInt *In1, ConstantInt *In2,
                           bool IsSigned) {
  if (!IsSigned)
    return Result->getValue().ult(In1->getValue());

  if (In2->isNegative())
    return Result->getValue().sgt(In1->getValue());
  return Result->getValue().slt(In1->getValue());
}

/// AddWithOverflow - Compute Result = In1+In2, returning true if the result
/// overflowed for this type.
static bool AddWithOverflow(Constant *&Result, Constant *In1,
                            Constant *In2, bool IsSigned = false) {
  Result = ConstantExpr::getAdd(In1, In2);

  if (VectorType *VTy = dyn_cast<VectorType>(In1->getType())) {
    for (unsigned i = 0, e = VTy->getNumElements(); i != e; ++i) {
      Constant *Idx = ConstantInt::get(Type::getInt32Ty(In1->getContext()), i);
      if (HasAddOverflow(ExtractElement(Result, Idx),
                         ExtractElement(In1, Idx),
                         ExtractElement(In2, Idx),
                         IsSigned))
        return true;
    }
    return false;
  }

  return HasAddOverflow(cast<ConstantInt>(Result),
                        cast<ConstantInt>(In1), cast<ConstantInt>(In2),
                        IsSigned);
}

static bool HasSubOverflow(ConstantInt *Result,
                           ConstantInt *In1, ConstantInt *In2,
                           bool IsSigned) {
  if (!IsSigned)
    return Result->getValue().ugt(In1->getValue());

  if (In2->isNegative())
    return Result->getValue().slt(In1->getValue());

  return Result->getValue().sgt(In1->getValue());
}

/// SubWithOverflow - Compute Result = In1-In2, returning true if the result
/// overflowed for this type.
static bool SubWithOverflow(Constant *&Result, Constant *In1,
                            Constant *In2, bool IsSigned = false) {
  Result = ConstantExpr::getSub(In1, In2);

  if (VectorType *VTy = dyn_cast<VectorType>(In1->getType())) {
    for (unsigned i = 0, e = VTy->getNumElements(); i != e; ++i) {
      Constant *Idx = ConstantInt::get(Type::getInt32Ty(In1->getContext()), i);
      if (HasSubOverflow(ExtractElement(Result, Idx),
                         ExtractElement(In1, Idx),
                         ExtractElement(In2, Idx),
                         IsSigned))
        return true;
    }
    return false;
  }

  return HasSubOverflow(cast<ConstantInt>(Result),
                        cast<ConstantInt>(In1), cast<ConstantInt>(In2),
                        IsSigned);
}

/// isSignBitCheck - Given an exploded icmp instruction, return true if the
/// comparison only checks the sign bit.  If it only checks the sign bit, set
/// TrueIfSigned if the result of the comparison is true when the input value is
/// signed.
static bool isSignBitCheck(ICmpInst::Predicate pred, ConstantInt *RHS,
                           bool &TrueIfSigned) {
  switch (pred) {
  case ICmpInst::ICMP_SLT:   // True if LHS s< 0
    TrueIfSigned = true;
    return RHS->isZero();
  case ICmpInst::ICMP_SLE:   // True if LHS s<= RHS and RHS == -1
    TrueIfSigned = true;
    return RHS->isAllOnesValue();
  case ICmpInst::ICMP_SGT:   // True if LHS s> -1
    TrueIfSigned = false;
    return RHS->isAllOnesValue();
  case ICmpInst::ICMP_UGT:
    // True if LHS u> RHS and RHS == high-bit-mask - 1
    TrueIfSigned = true;
    return RHS->isMaxValue(true);
  case ICmpInst::ICMP_UGE:
    // True if LHS u>= RHS and RHS == high-bit-mask (2^7, 2^15, 2^31, etc)
    TrueIfSigned = true;
    return RHS->getValue().isSignBit();
  default:
    return false;
  }
}

// isHighOnes - Return true if the constant is of the form 1+0+.
// This is the same as lowones(~X).
static bool isHighOnes(const ConstantInt *CI) {
  return (~CI->getValue() + 1).isPowerOf2();
}

/// ComputeSignedMinMaxValuesFromKnownBits - Given a signed integer type and a
/// set of known zero and one bits, compute the maximum and minimum values that
/// could have the specified known zero and known one bits, returning them in
/// min/max.
static void ComputeSignedMinMaxValuesFromKnownBits(const APInt& KnownZero,
                                                   const APInt& KnownOne,
                                                   APInt& Min, APInt& Max) {
  assert(KnownZero.getBitWidth() == KnownOne.getBitWidth() &&
         KnownZero.getBitWidth() == Min.getBitWidth() &&
         KnownZero.getBitWidth() == Max.getBitWidth() &&
         "KnownZero, KnownOne and Min, Max must have equal bitwidth.");
  APInt UnknownBits = ~(KnownZero|KnownOne);

  // The minimum value is when all unknown bits are zeros, EXCEPT for the sign
  // bit if it is unknown.
  Min = KnownOne;
  Max = KnownOne|UnknownBits;

  if (UnknownBits.isNegative()) { // Sign bit is unknown
    Min.setBit(Min.getBitWidth()-1);
    Max.clearBit(Max.getBitWidth()-1);
  }
}

// ComputeUnsignedMinMaxValuesFromKnownBits - Given an unsigned integer type and
// a set of known zero and one bits, compute the maximum and minimum values that
// could have the specified known zero and known one bits, returning them in
// min/max.
static void ComputeUnsignedMinMaxValuesFromKnownBits(const APInt &KnownZero,
                                                     const APInt &KnownOne,
                                                     APInt &Min, APInt &Max) {
  assert(KnownZero.getBitWidth() == KnownOne.getBitWidth() &&
         KnownZero.getBitWidth() == Min.getBitWidth() &&
         KnownZero.getBitWidth() == Max.getBitWidth() &&
         "Ty, KnownZero, KnownOne and Min, Max must have equal bitwidth.");
  APInt UnknownBits = ~(KnownZero|KnownOne);

  // The minimum value is when the unknown bits are all zeros.
  Min = KnownOne;
  // The maximum value is when the unknown bits are all ones.
  Max = KnownOne|UnknownBits;
}



/// FoldCmpLoadFromIndexedGlobal - Called we see this pattern:
///   cmp pred (load (gep GV, ...)), cmpcst
/// where GV is a global variable with a constant initializer.  Try to simplify
/// this into some simple computation that does not need the load.  For example
/// we can optimize "icmp eq (load (gep "foo", 0, i)), 0" into "icmp eq i, 3".
///
/// If AndCst is non-null, then the loaded value is masked with that constant
/// before doing the comparison.  This handles cases like "A[i]&4 == 0".
Instruction *InstCombiner::
FoldCmpLoadFromIndexedGlobal(GetElementPtrInst *GEP, GlobalVariable *GV,
                             CmpInst &ICI, ConstantInt *AndCst) {
  // We need TD information to know the pointer size unless this is inbounds.
  if (!GEP->isInBounds() && TD == 0) return 0;

  Constant *Init = GV->getInitializer();
  if (!isa<ConstantArray>(Init) && !isa<ConstantDataArray>(Init))
    return 0;
  
  uint64_t ArrayElementCount = Init->getType()->getArrayNumElements();
  if (ArrayElementCount > 1024) return 0;  // Don't blow up on huge arrays.

  // There are many forms of this optimization we can handle, for now, just do
  // the simple index into a single-dimensional array.
  //
  // Require: GEP GV, 0, i {{, constant indices}}
  if (GEP->getNumOperands() < 3 ||
      !isa<ConstantInt>(GEP->getOperand(1)) ||
      !cast<ConstantInt>(GEP->getOperand(1))->isZero() ||
      isa<Constant>(GEP->getOperand(2)))
    return 0;

  // Check that indices after the variable are constants and in-range for the
  // type they index.  Collect the indices.  This is typically for arrays of
  // structs.
  SmallVector<unsigned, 4> LaterIndices;

  Type *EltTy = Init->getType()->getArrayElementType();
  for (unsigned i = 3, e = GEP->getNumOperands(); i != e; ++i) {
    ConstantInt *Idx = dyn_cast<ConstantInt>(GEP->getOperand(i));
    if (Idx == 0) return 0;  // Variable index.

    uint64_t IdxVal = Idx->getZExtValue();
    if ((unsigned)IdxVal != IdxVal) return 0; // Too large array index.

    if (StructType *STy = dyn_cast<StructType>(EltTy))
      EltTy = STy->getElementType(IdxVal);
    else if (ArrayType *ATy = dyn_cast<ArrayType>(EltTy)) {
      if (IdxVal >= ATy->getNumElements()) return 0;
      EltTy = ATy->getElementType();
    } else {
      return 0; // Unknown type.
    }

    LaterIndices.push_back(IdxVal);
  }

  enum { Overdefined = -3, Undefined = -2 };

  // Variables for our state machines.

  // FirstTrueElement/SecondTrueElement - Used to emit a comparison of the form
  // "i == 47 | i == 87", where 47 is the first index the condition is true for,
  // and 87 is the second (and last) index.  FirstTrueElement is -2 when
  // undefined, otherwise set to the first true element.  SecondTrueElement is
  // -2 when undefined, -3 when overdefined and >= 0 when that index is true.
  int FirstTrueElement = Undefined, SecondTrueElement = Undefined;

  // FirstFalseElement/SecondFalseElement - Used to emit a comparison of the
  // form "i != 47 & i != 87".  Same state transitions as for true elements.
  int FirstFalseElement = Undefined, SecondFalseElement = Undefined;

  /// TrueRangeEnd/FalseRangeEnd - In conjunction with First*Element, these
  /// define a state machine that triggers for ranges of values that the index
  /// is true or false for.  This triggers on things like "abbbbc"[i] == 'b'.
  /// This is -2 when undefined, -3 when overdefined, and otherwise the last
  /// index in the range (inclusive).  We use -2 for undefined here because we
  /// use relative comparisons and don't want 0-1 to match -1.
  int TrueRangeEnd = Undefined, FalseRangeEnd = Undefined;

  // MagicBitvector - This is a magic bitvector where we set a bit if the
  // comparison is true for element 'i'.  If there are 64 elements or less in
  // the array, this will fully represent all the comparison results.
  uint64_t MagicBitvector = 0;


  // Scan the array and see if one of our patterns matches.
  Constant *CompareRHS = cast<Constant>(ICI.getOperand(1));
  for (unsigned i = 0, e = ArrayElementCount; i != e; ++i) {
    Constant *Elt = Init->getAggregateElement(i);
    if (Elt == 0) return 0;

    // If this is indexing an array of structures, get the structure element.
    if (!LaterIndices.empty())
      Elt = ConstantExpr::getExtractValue(Elt, LaterIndices);

    // If the element is masked, handle it.
    if (AndCst) Elt = ConstantExpr::getAnd(Elt, AndCst);

    // Find out if the comparison would be true or false for the i'th element.
    Constant *C = ConstantFoldCompareInstOperands(ICI.getPredicate(), Elt,
                                                  CompareRHS, TD, TLI);
    // If the result is undef for this element, ignore it.
    if (isa<UndefValue>(C)) {
      // Extend range state machines to cover this element in case there is an
      // undef in the middle of the range.
      if (TrueRangeEnd == (int)i-1)
        TrueRangeEnd = i;
      if (FalseRangeEnd == (int)i-1)
        FalseRangeEnd = i;
      continue;
    }

    // If we can't compute the result for any of the elements, we have to give
    // up evaluating the entire conditional.
    if (!isa<ConstantInt>(C)) return 0;

    // Otherwise, we know if the comparison is true or false for this element,
    // update our state machines.
    bool IsTrueForElt = !cast<ConstantInt>(C)->isZero();

    // State machine for single/double/range index comparison.
    if (IsTrueForElt) {
      // Update the TrueElement state machine.
      if (FirstTrueElement == Undefined)
        FirstTrueElement = TrueRangeEnd = i;  // First true element.
      else {
        // Update double-compare state machine.
        if (SecondTrueElement == Undefined)
          SecondTrueElement = i;
        else
          SecondTrueElement = Overdefined;

        // Update range state machine.
        if (TrueRangeEnd == (int)i-1)
          TrueRangeEnd = i;
        else
          TrueRangeEnd = Overdefined;
      }
    } else {
      // Update the FalseElement state machine.
      if (FirstFalseElement == Undefined)
        FirstFalseElement = FalseRangeEnd = i; // First false element.
      else {
        // Update double-compare state machine.
        if (SecondFalseElement == Undefined)
          SecondFalseElement = i;
        else
          SecondFalseElement = Overdefined;

        // Update range state machine.
        if (FalseRangeEnd == (int)i-1)
          FalseRangeEnd = i;
        else
          FalseRangeEnd = Overdefined;
      }
    }


    // If this element is in range, update our magic bitvector.
    if (i < 64 && IsTrueForElt)
      MagicBitvector |= 1ULL << i;

    // If all of our states become overdefined, bail out early.  Since the
    // predicate is expensive, only check it every 8 elements.  This is only
    // really useful for really huge arrays.
    if ((i & 8) == 0 && i >= 64 && SecondTrueElement == Overdefined &&
        SecondFalseElement == Overdefined && TrueRangeEnd == Overdefined &&
        FalseRangeEnd == Overdefined)
      return 0;
  }

  // Now that we've scanned the entire array, emit our new comparison(s).  We
  // order the state machines in complexity of the generated code.
  Value *Idx = GEP->getOperand(2);

  // If the index is larger than the pointer size of the target, truncate the
  // index down like the GEP would do implicitly.  We don't have to do this for
  // an inbounds GEP because the index can't be out of range.
  if (!GEP->isInBounds() &&
      Idx->getType()->getPrimitiveSizeInBits() > TD->getPointerSizeInBits())
    Idx = Builder->CreateTrunc(Idx, TD->getIntPtrType(Idx->getContext()));

  // If the comparison is only true for one or two elements, emit direct
  // comparisons.
  if (SecondTrueElement != Overdefined) {
    // None true -> false.
    if (FirstTrueElement == Undefined)
      return ReplaceInstUsesWith(ICI, ConstantInt::getFalse(GEP->getContext()));

    Value *FirstTrueIdx = ConstantInt::get(Idx->getType(), FirstTrueElement);

    // True for one element -> 'i == 47'.
    if (SecondTrueElement == Undefined)
      return new ICmpInst(ICmpInst::ICMP_EQ, Idx, FirstTrueIdx);

    // True for two elements -> 'i == 47 | i == 72'.
    Value *C1 = Builder->CreateICmpEQ(Idx, FirstTrueIdx);
    Value *SecondTrueIdx = ConstantInt::get(Idx->getType(), SecondTrueElement);
    Value *C2 = Builder->CreateICmpEQ(Idx, SecondTrueIdx);
    return BinaryOperator::CreateOr(C1, C2);
  }

  // If the comparison is only false for one or two elements, emit direct
  // comparisons.
  if (SecondFalseElement != Overdefined) {
    // None false -> true.
    if (FirstFalseElement == Undefined)
      return ReplaceInstUsesWith(ICI, ConstantInt::getTrue(GEP->getContext()));

    Value *FirstFalseIdx = ConstantInt::get(Idx->getType(), FirstFalseElement);

    // False for one element -> 'i != 47'.
    if (SecondFalseElement == Undefined)
      return new ICmpInst(ICmpInst::ICMP_NE, Idx, FirstFalseIdx);

    // False for two elements -> 'i != 47 & i != 72'.
    Value *C1 = Builder->CreateICmpNE(Idx, FirstFalseIdx);
    Value *SecondFalseIdx = ConstantInt::get(Idx->getType(),SecondFalseElement);
    Value *C2 = Builder->CreateICmpNE(Idx, SecondFalseIdx);
    return BinaryOperator::CreateAnd(C1, C2);
  }

  // If the comparison can be replaced with a range comparison for the elements
  // where it is true, emit the range check.
  if (TrueRangeEnd != Overdefined) {
    assert(TrueRangeEnd != FirstTrueElement && "Should emit single compare");

    // Generate (i-FirstTrue) <u (TrueRangeEnd-FirstTrue+1).
    if (FirstTrueElement) {
      Value *Offs = ConstantInt::get(Idx->getType(), -FirstTrueElement);
      Idx = Builder->CreateAdd(Idx, Offs);
    }

    Value *End = ConstantInt::get(Idx->getType(),
                                  TrueRangeEnd-FirstTrueElement+1);
    return new ICmpInst(ICmpInst::ICMP_ULT, Idx, End);
  }

  // False range check.
  if (FalseRangeEnd != Overdefined) {
    assert(FalseRangeEnd != FirstFalseElement && "Should emit single compare");
    // Generate (i-FirstFalse) >u (FalseRangeEnd-FirstFalse).
    if (FirstFalseElement) {
      Value *Offs = ConstantInt::get(Idx->getType(), -FirstFalseElement);
      Idx = Builder->CreateAdd(Idx, Offs);
    }

    Value *End = ConstantInt::get(Idx->getType(),
                                  FalseRangeEnd-FirstFalseElement);
    return new ICmpInst(ICmpInst::ICMP_UGT, Idx, End);
  }


  // If a 32-bit or 64-bit magic bitvector captures the entire comparison state
  // of this load, replace it with computation that does:
  //   ((magic_cst >> i) & 1) != 0
  if (ArrayElementCount <= 32 ||
      (TD && ArrayElementCount <= 64 && TD->isLegalInteger(64))) {
    Type *Ty;
    if (ArrayElementCount <= 32)
      Ty = Type::getInt32Ty(Init->getContext());
    else
      Ty = Type::getInt64Ty(Init->getContext());
    Value *V = Builder->CreateIntCast(Idx, Ty, false);
    V = Builder->CreateLShr(ConstantInt::get(Ty, MagicBitvector), V);
    V = Builder->CreateAnd(ConstantInt::get(Ty, 1), V);
    return new ICmpInst(ICmpInst::ICMP_NE, V, ConstantInt::get(Ty, 0));
  }

  return 0;
}


/// EvaluateGEPOffsetExpression - Return a value that can be used to compare
/// the *offset* implied by a GEP to zero.  For example, if we have &A[i], we
/// want to return 'i' for "icmp ne i, 0".  Note that, in general, indices can
/// be complex, and scales are involved.  The above expression would also be
/// legal to codegen as "icmp ne (i*4), 0" (assuming A is a pointer to i32).
/// This later form is less amenable to optimization though, and we are allowed
/// to generate the first by knowing that pointer arithmetic doesn't overflow.
///
/// If we can't emit an optimized form for this expression, this returns null.
///
static Value *EvaluateGEPOffsetExpression(User *GEP, InstCombiner &IC) {
  DataLayout &TD = *IC.getDataLayout();
  gep_type_iterator GTI = gep_type_begin(GEP);

  // Check to see if this gep only has a single variable index.  If so, and if
  // any constant indices are a multiple of its scale, then we can compute this
  // in terms of the scale of the variable index.  For example, if the GEP
  // implies an offset of "12 + i*4", then we can codegen this as "3 + i",
  // because the expression will cross zero at the same point.
  unsigned i, e = GEP->getNumOperands();
  int64_t Offset = 0;
  for (i = 1; i != e; ++i, ++GTI) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(GEP->getOperand(i))) {
      // Compute the aggregate offset of constant indices.
      if (CI->isZero()) continue;

      // Handle a struct index, which adds its field offset to the pointer.
      if (StructType *STy = dyn_cast<StructType>(*GTI)) {
        Offset += TD.getStructLayout(STy)->getElementOffset(CI->getZExtValue());
      } else {
        uint64_t Size = TD.getTypeAllocSize(GTI.getIndexedType());
        Offset += Size*CI->getSExtValue();
      }
    } else {
      // Found our variable index.
      break;
    }
  }

  // If there are no variable indices, we must have a constant offset, just
  // evaluate it the general way.
  if (i == e) return 0;

  Value *VariableIdx = GEP->getOperand(i);
  // Determine the scale factor of the variable element.  For example, this is
  // 4 if the variable index is into an array of i32.
  uint64_t VariableScale = TD.getTypeAllocSize(GTI.getIndexedType());

  // Verify that there are no other variable indices.  If so, emit the hard way.
  for (++i, ++GTI; i != e; ++i, ++GTI) {
    ConstantInt *CI = dyn_cast<ConstantInt>(GEP->getOperand(i));
    if (!CI) return 0;

    // Compute the aggregate offset of constant indices.
    if (CI->isZero()) continue;

    // Handle a struct index, which adds its field offset to the pointer.
    if (StructType *STy = dyn_cast<StructType>(*GTI)) {
      Offset += TD.getStructLayout(STy)->getElementOffset(CI->getZExtValue());
    } else {
      uint64_t Size = TD.getTypeAllocSize(GTI.getIndexedType());
      Offset += Size*CI->getSExtValue();
    }
  }

  // Okay, we know we have a single variable index, which must be a
  // pointer/array/vector index.  If there is no offset, life is simple, return
  // the index.
  unsigned IntPtrWidth = TD.getPointerSizeInBits();
  if (Offset == 0) {
    // Cast to intptrty in case a truncation occurs.  If an extension is needed,
    // we don't need to bother extending: the extension won't affect where the
    // computation crosses zero.
    if (VariableIdx->getType()->getPrimitiveSizeInBits() > IntPtrWidth) {
      Type *IntPtrTy = TD.getIntPtrType(VariableIdx->getContext());
      VariableIdx = IC.Builder->CreateTrunc(VariableIdx, IntPtrTy);
    }
    return VariableIdx;
  }

  // Otherwise, there is an index.  The computation we will do will be modulo
  // the pointer size, so get it.
  uint64_t PtrSizeMask = ~0ULL >> (64-IntPtrWidth);

  Offset &= PtrSizeMask;
  VariableScale &= PtrSizeMask;

  // To do this transformation, any constant index must be a multiple of the
  // variable scale factor.  For example, we can evaluate "12 + 4*i" as "3 + i",
  // but we can't evaluate "10 + 3*i" in terms of i.  Check that the offset is a
  // multiple of the variable scale.
  int64_t NewOffs = Offset / (int64_t)VariableScale;
  if (Offset != NewOffs*(int64_t)VariableScale)
    return 0;

  // Okay, we can do this evaluation.  Start by converting the index to intptr.
  Type *IntPtrTy = TD.getIntPtrType(VariableIdx->getContext());
  if (VariableIdx->getType() != IntPtrTy)
    VariableIdx = IC.Builder->CreateIntCast(VariableIdx, IntPtrTy,
                                            true /*Signed*/);
  Constant *OffsetVal = ConstantInt::get(IntPtrTy, NewOffs);
  return IC.Builder->CreateAdd(VariableIdx, OffsetVal, "offset");
}

/// FoldGEPICmp - Fold comparisons between a GEP instruction and something
/// else.  At this point we know that the GEP is on the LHS of the comparison.
Instruction *InstCombiner::FoldGEPICmp(GEPOperator *GEPLHS, Value *RHS,
                                       ICmpInst::Predicate Cond,
                                       Instruction &I) {
  // Don't transform signed compares of GEPs into index compares. Even if the
  // GEP is inbounds, the final add of the base pointer can have signed overflow
  // and would change the result of the icmp.
  // e.g. "&foo[0] <s &foo[1]" can't be folded to "true" because "foo" could be
  // the maximum signed value for the pointer type.
  if (ICmpInst::isSigned(Cond))
    return 0;

  // Look through bitcasts.
  if (BitCastInst *BCI = dyn_cast<BitCastInst>(RHS))
    RHS = BCI->getOperand(0);

  Value *PtrBase = GEPLHS->getOperand(0);
  if (TD && PtrBase == RHS && GEPLHS->isInBounds()) {
    // ((gep Ptr, OFFSET) cmp Ptr)   ---> (OFFSET cmp 0).
    // This transformation (ignoring the base and scales) is valid because we
    // know pointers can't overflow since the gep is inbounds.  See if we can
    // output an optimized form.
    Value *Offset = EvaluateGEPOffsetExpression(GEPLHS, *this);

    // If not, synthesize the offset the hard way.
    if (Offset == 0)
      Offset = EmitGEPOffset(GEPLHS);
    return new ICmpInst(ICmpInst::getSignedPredicate(Cond), Offset,
                        Constant::getNullValue(Offset->getType()));
  } else if (GEPOperator *GEPRHS = dyn_cast<GEPOperator>(RHS)) {
    // If the base pointers are different, but the indices are the same, just
    // compare the base pointer.
    if (PtrBase != GEPRHS->getOperand(0)) {
      bool IndicesTheSame = GEPLHS->getNumOperands()==GEPRHS->getNumOperands();
      IndicesTheSame &= GEPLHS->getOperand(0)->getType() ==
                        GEPRHS->getOperand(0)->getType();
      if (IndicesTheSame)
        for (unsigned i = 1, e = GEPLHS->getNumOperands(); i != e; ++i)
          if (GEPLHS->getOperand(i) != GEPRHS->getOperand(i)) {
            IndicesTheSame = false;
            break;
          }

      // If all indices are the same, just compare the base pointers.
      if (IndicesTheSame)
        return new ICmpInst(ICmpInst::getSignedPredicate(Cond),
                            GEPLHS->getOperand(0), GEPRHS->getOperand(0));

      // If we're comparing GEPs with two base pointers that only differ in type
      // and both GEPs have only constant indices or just one use, then fold
      // the compare with the adjusted indices.
      if (TD && GEPLHS->isInBounds() && GEPRHS->isInBounds() &&
          (GEPLHS->hasAllConstantIndices() || GEPLHS->hasOneUse()) &&
          (GEPRHS->hasAllConstantIndices() || GEPRHS->hasOneUse()) &&
          PtrBase->stripPointerCasts() ==
            GEPRHS->getOperand(0)->stripPointerCasts()) {
        Value *Cmp = Builder->CreateICmp(ICmpInst::getSignedPredicate(Cond),
                                         EmitGEPOffset(GEPLHS),
                                         EmitGEPOffset(GEPRHS));
        return ReplaceInstUsesWith(I, Cmp);
      }

      // Otherwise, the base pointers are different and the indices are
      // different, bail out.
      return 0;
    }

    // If one of the GEPs has all zero indices, recurse.
    bool AllZeros = true;
    for (unsigned i = 1, e = GEPLHS->getNumOperands(); i != e; ++i)
      if (!isa<Constant>(GEPLHS->getOperand(i)) ||
          !cast<Constant>(GEPLHS->getOperand(i))->isNullValue()) {
        AllZeros = false;
        break;
      }
    if (AllZeros)
      return FoldGEPICmp(GEPRHS, GEPLHS->getOperand(0),
                          ICmpInst::getSwappedPredicate(Cond), I);

    // If the other GEP has all zero indices, recurse.
    AllZeros = true;
    for (unsigned i = 1, e = GEPRHS->getNumOperands(); i != e; ++i)
      if (!isa<Constant>(GEPRHS->getOperand(i)) ||
          !cast<Constant>(GEPRHS->getOperand(i))->isNullValue()) {
        AllZeros = false;
        break;
      }
    if (AllZeros)
      return FoldGEPICmp(GEPLHS, GEPRHS->getOperand(0), Cond, I);

    bool GEPsInBounds = GEPLHS->isInBounds() && GEPRHS->isInBounds();
    if (GEPLHS->getNumOperands() == GEPRHS->getNumOperands()) {
      // If the GEPs only differ by one index, compare it.
      unsigned NumDifferences = 0;  // Keep track of # differences.
      unsigned DiffOperand = 0;     // The operand that differs.
      for (unsigned i = 1, e = GEPRHS->getNumOperands(); i != e; ++i)
        if (GEPLHS->getOperand(i) != GEPRHS->getOperand(i)) {
          if (GEPLHS->getOperand(i)->getType()->getPrimitiveSizeInBits() !=
                   GEPRHS->getOperand(i)->getType()->getPrimitiveSizeInBits()) {
            // Irreconcilable differences.
            NumDifferences = 2;
            break;
          } else {
            if (NumDifferences++) break;
            DiffOperand = i;
          }
        }

      if (NumDifferences == 0)   // SAME GEP?
        return ReplaceInstUsesWith(I, // No comparison is needed here.
                               ConstantInt::get(Type::getInt1Ty(I.getContext()),
                                             ICmpInst::isTrueWhenEqual(Cond)));

      else if (NumDifferences == 1 && GEPsInBounds) {
        Value *LHSV = GEPLHS->getOperand(DiffOperand);
        Value *RHSV = GEPRHS->getOperand(DiffOperand);
        // Make sure we do a signed comparison here.
        return new ICmpInst(ICmpInst::getSignedPredicate(Cond), LHSV, RHSV);
      }
    }

    // Only lower this if the icmp is the only user of the GEP or if we expect
    // the result to fold to a constant!
    if (TD &&
        GEPsInBounds &&
        (isa<ConstantExpr>(GEPLHS) || GEPLHS->hasOneUse()) &&
        (isa<ConstantExpr>(GEPRHS) || GEPRHS->hasOneUse())) {
      // ((gep Ptr, OFFSET1) cmp (gep Ptr, OFFSET2)  --->  (OFFSET1 cmp OFFSET2)
      Value *L = EmitGEPOffset(GEPLHS);
      Value *R = EmitGEPOffset(GEPRHS);
      return new ICmpInst(ICmpInst::getSignedPredicate(Cond), L, R);
    }
  }
  return 0;
}

/// FoldICmpAddOpCst - Fold "icmp pred (X+CI), X".
Instruction *InstCombiner::FoldICmpAddOpCst(ICmpInst &ICI,
                                            Value *X, ConstantInt *CI,
                                            ICmpInst::Predicate Pred,
                                            Value *TheAdd) {
  // If we have X+0, exit early (simplifying logic below) and let it get folded
  // elsewhere.   icmp X+0, X  -> icmp X, X
  if (CI->isZero()) {
    bool isTrue = ICmpInst::isTrueWhenEqual(Pred);
    return ReplaceInstUsesWith(ICI, ConstantInt::get(ICI.getType(), isTrue));
  }

  // (X+4) == X -> false.
  if (Pred == ICmpInst::ICMP_EQ)
    return ReplaceInstUsesWith(ICI, ConstantInt::getFalse(X->getContext()));

  // (X+4) != X -> true.
  if (Pred == ICmpInst::ICMP_NE)
    return ReplaceInstUsesWith(ICI, ConstantInt::getTrue(X->getContext()));

  // From this point on, we know that (X+C <= X) --> (X+C < X) because C != 0,
  // so the values can never be equal.  Similarly for all other "or equals"
  // operators.

  // (X+1) <u X        --> X >u (MAXUINT-1)        --> X == 255
  // (X+2) <u X        --> X >u (MAXUINT-2)        --> X > 253
  // (X+MAXUINT) <u X  --> X >u (MAXUINT-MAXUINT)  --> X != 0
  if (Pred == ICmpInst::ICMP_ULT || Pred == ICmpInst::ICMP_ULE) {
    Value *R =
      ConstantExpr::getSub(ConstantInt::getAllOnesValue(CI->getType()), CI);
    return new ICmpInst(ICmpInst::ICMP_UGT, X, R);
  }

  // (X+1) >u X        --> X <u (0-1)        --> X != 255
  // (X+2) >u X        --> X <u (0-2)        --> X <u 254
  // (X+MAXUINT) >u X  --> X <u (0-MAXUINT)  --> X <u 1  --> X == 0
  if (Pred == ICmpInst::ICMP_UGT || Pred == ICmpInst::ICMP_UGE)
    return new ICmpInst(ICmpInst::ICMP_ULT, X, ConstantExpr::getNeg(CI));

  unsigned BitWidth = CI->getType()->getPrimitiveSizeInBits();
  ConstantInt *SMax = ConstantInt::get(X->getContext(),
                                       APInt::getSignedMaxValue(BitWidth));

  // (X+ 1) <s X       --> X >s (MAXSINT-1)          --> X == 127
  // (X+ 2) <s X       --> X >s (MAXSINT-2)          --> X >s 125
  // (X+MAXSINT) <s X  --> X >s (MAXSINT-MAXSINT)    --> X >s 0
  // (X+MINSINT) <s X  --> X >s (MAXSINT-MINSINT)    --> X >s -1
  // (X+ -2) <s X      --> X >s (MAXSINT- -2)        --> X >s 126
  // (X+ -1) <s X      --> X >s (MAXSINT- -1)        --> X != 127
  if (Pred == ICmpInst::ICMP_SLT || Pred == ICmpInst::ICMP_SLE)
    return new ICmpInst(ICmpInst::ICMP_SGT, X, ConstantExpr::getSub(SMax, CI));

  // (X+ 1) >s X       --> X <s (MAXSINT-(1-1))       --> X != 127
  // (X+ 2) >s X       --> X <s (MAXSINT-(2-1))       --> X <s 126
  // (X+MAXSINT) >s X  --> X <s (MAXSINT-(MAXSINT-1)) --> X <s 1
  // (X+MINSINT) >s X  --> X <s (MAXSINT-(MINSINT-1)) --> X <s -2
  // (X+ -2) >s X      --> X <s (MAXSINT-(-2-1))      --> X <s -126
  // (X+ -1) >s X      --> X <s (MAXSINT-(-1-1))      --> X == -128

  assert(Pred == ICmpInst::ICMP_SGT || Pred == ICmpInst::ICMP_SGE);
  Constant *C = ConstantInt::get(X->getContext(), CI->getValue()-1);
  return new ICmpInst(ICmpInst::ICMP_SLT, X, ConstantExpr::getSub(SMax, C));
}

/// FoldICmpDivCst - Fold "icmp pred, ([su]div X, DivRHS), CmpRHS" where DivRHS
/// and CmpRHS are both known to be integer constants.
Instruction *InstCombiner::FoldICmpDivCst(ICmpInst &ICI, BinaryOperator *DivI,
                                          ConstantInt *DivRHS) {
  ConstantInt *CmpRHS = cast<ConstantInt>(ICI.getOperand(1));
  const APInt &CmpRHSV = CmpRHS->getValue();

  // FIXME: If the operand types don't match the type of the divide
  // then don't attempt this transform. The code below doesn't have the
  // logic to deal with a signed divide and an unsigned compare (and
  // vice versa). This is because (x /s C1) <s C2  produces different
  // results than (x /s C1) <u C2 or (x /u C1) <s C2 or even
  // (x /u C1) <u C2.  Simply casting the operands and result won't
  // work. :(  The if statement below tests that condition and bails
  // if it finds it.
  bool DivIsSigned = DivI->getOpcode() == Instruction::SDiv;
  if (!ICI.isEquality() && DivIsSigned != ICI.isSigned())
    return 0;
  if (DivRHS->isZero())
    return 0; // The ProdOV computation fails on divide by zero.
  if (DivIsSigned && DivRHS->isAllOnesValue())
    return 0; // The overflow computation also screws up here
  if (DivRHS->isOne()) {
    // This eliminates some funny cases with INT_MIN.
    ICI.setOperand(0, DivI->getOperand(0));   // X/1 == X.
    return &ICI;
  }

  // Compute Prod = CI * DivRHS. We are essentially solving an equation
  // of form X/C1=C2. We solve for X by multiplying C1 (DivRHS) and
  // C2 (CI). By solving for X we can turn this into a range check
  // instead of computing a divide.
  Constant *Prod = ConstantExpr::getMul(CmpRHS, DivRHS);

  // Determine if the product overflows by seeing if the product is
  // not equal to the divide. Make sure we do the same kind of divide
  // as in the LHS instruction that we're folding.
  bool ProdOV = (DivIsSigned ? ConstantExpr::getSDiv(Prod, DivRHS) :
                 ConstantExpr::getUDiv(Prod, DivRHS)) != CmpRHS;

  // Get the ICmp opcode
  ICmpInst::Predicate Pred = ICI.getPredicate();

  /// If the division is known to be exact, then there is no remainder from the
  /// divide, so the covered range size is unit, otherwise it is the divisor.
  ConstantInt *RangeSize = DivI->isExact() ? getOne(Prod) : DivRHS;

  // Figure out the interval that is being checked.  For example, a comparison
  // like "X /u 5 == 0" is really checking that X is in the interval [0, 5).
  // Compute this interval based on the constants involved and the signedness of
  // the compare/divide.  This computes a half-open interval, keeping track of
  // whether either value in the interval overflows.  After analysis each
  // overflow variable is set to 0 if it's corresponding bound variable is valid
  // -1 if overflowed off the bottom end, or +1 if overflowed off the top end.
  int LoOverflow = 0, HiOverflow = 0;
  Constant *LoBound = 0, *HiBound = 0;

  if (!DivIsSigned) {  // udiv
    // e.g. X/5 op 3  --> [15, 20)
    LoBound = Prod;
    HiOverflow = LoOverflow = ProdOV;
    if (!HiOverflow) {
      // If this is not an exact divide, then many values in the range collapse
      // to the same result value.
      HiOverflow = AddWithOverflow(HiBound, LoBound, RangeSize, false);
    }

  } else if (DivRHS->getValue().isStrictlyPositive()) { // Divisor is > 0.
    if (CmpRHSV == 0) {       // (X / pos) op 0
      // Can't overflow.  e.g.  X/2 op 0 --> [-1, 2)
      LoBound = ConstantExpr::getNeg(SubOne(RangeSize));
      HiBound = RangeSize;
    } else if (CmpRHSV.isStrictlyPositive()) {   // (X / pos) op pos
      LoBound = Prod;     // e.g.   X/5 op 3 --> [15, 20)
      HiOverflow = LoOverflow = ProdOV;
      if (!HiOverflow)
        HiOverflow = AddWithOverflow(HiBound, Prod, RangeSize, true);
    } else {                       // (X / pos) op neg
      // e.g. X/5 op -3  --> [-15-4, -15+1) --> [-19, -14)
      HiBound = AddOne(Prod);
      LoOverflow = HiOverflow = ProdOV ? -1 : 0;
      if (!LoOverflow) {
        ConstantInt *DivNeg =cast<ConstantInt>(ConstantExpr::getNeg(RangeSize));
        LoOverflow = AddWithOverflow(LoBound, HiBound, DivNeg, true) ? -1 : 0;
      }
    }
  } else if (DivRHS->isNegative()) { // Divisor is < 0.
    if (DivI->isExact())
      RangeSize = cast<ConstantInt>(ConstantExpr::getNeg(RangeSize));
    if (CmpRHSV == 0) {       // (X / neg) op 0
      // e.g. X/-5 op 0  --> [-4, 5)
      LoBound = AddOne(RangeSize);
      HiBound = cast<ConstantInt>(ConstantExpr::getNeg(RangeSize));
      if (HiBound == DivRHS) {     // -INTMIN = INTMIN
        HiOverflow = 1;            // [INTMIN+1, overflow)
        HiBound = 0;               // e.g. X/INTMIN = 0 --> X > INTMIN
      }
    } else if (CmpRHSV.isStrictlyPositive()) {   // (X / neg) op pos
      // e.g. X/-5 op 3  --> [-19, -14)
      HiBound = AddOne(Prod);
      HiOverflow = LoOverflow = ProdOV ? -1 : 0;
      if (!LoOverflow)
        LoOverflow = AddWithOverflow(LoBound, HiBound, RangeSize, true) ? -1:0;
    } else {                       // (X / neg) op neg
      LoBound = Prod;       // e.g. X/-5 op -3  --> [15, 20)
      LoOverflow = HiOverflow = ProdOV;
      if (!HiOverflow)
        HiOverflow = SubWithOverflow(HiBound, Prod, RangeSize, true);
    }

    // Dividing by a negative swaps the condition.  LT <-> GT
    Pred = ICmpInst::getSwappedPredicate(Pred);
  }

  Value *X = DivI->getOperand(0);
  switch (Pred) {
  default: llvm_unreachable("Unhandled icmp opcode!");
  case ICmpInst::ICMP_EQ:
    if (LoOverflow && HiOverflow)
      return ReplaceInstUsesWith(ICI, ConstantInt::getFalse(ICI.getContext()));
    if (HiOverflow)
      return new ICmpInst(DivIsSigned ? ICmpInst::ICMP_SGE :
                          ICmpInst::ICMP_UGE, X, LoBound);
    if (LoOverflow)
      return new ICmpInst(DivIsSigned ? ICmpInst::ICMP_SLT :
                          ICmpInst::ICMP_ULT, X, HiBound);
    return ReplaceInstUsesWith(ICI, InsertRangeTest(X, LoBound, HiBound,
                                                    DivIsSigned, true));
  case ICmpInst::ICMP_NE:
    if (LoOverflow && HiOverflow)
      return ReplaceInstUsesWith(ICI, ConstantInt::getTrue(ICI.getContext()));
    if (HiOverflow)
      return new ICmpInst(DivIsSigned ? ICmpInst::ICMP_SLT :
                          ICmpInst::ICMP_ULT, X, LoBound);
    if (LoOverflow)
      return new ICmpInst(DivIsSigned ? ICmpInst::ICMP_SGE :
                          ICmpInst::ICMP_UGE, X, HiBound);
    return ReplaceInstUsesWith(ICI, InsertRangeTest(X, LoBound, HiBound,
                                                    DivIsSigned, false));
  case ICmpInst::ICMP_ULT:
  case ICmpInst::ICMP_SLT:
    if (LoOverflow == +1)   // Low bound is greater than input range.
      return ReplaceInstUsesWith(ICI, ConstantInt::getTrue(ICI.getContext()));
    if (LoOverflow == -1)   // Low bound is less than input range.
      return ReplaceInstUsesWith(ICI, ConstantInt::getFalse(ICI.getContext()));
    return new ICmpInst(Pred, X, LoBound);
  case ICmpInst::ICMP_UGT:
  case ICmpInst::ICMP_SGT:
    if (HiOverflow == +1)       // High bound greater than input range.
      return ReplaceInstUsesWith(ICI, ConstantInt::getFalse(ICI.getContext()));
    if (HiOverflow == -1)       // High bound less than input range.
      return ReplaceInstUsesWith(ICI, ConstantInt::getTrue(ICI.getContext()));
    if (Pred == ICmpInst::ICMP_UGT)
      return new ICmpInst(ICmpInst::ICMP_UGE, X, HiBound);
    return new ICmpInst(ICmpInst::ICMP_SGE, X, HiBound);
  }
}

/// FoldICmpShrCst - Handle "icmp(([al]shr X, cst1), cst2)".
Instruction *InstCombiner::FoldICmpShrCst(ICmpInst &ICI, BinaryOperator *Shr,
                                          ConstantInt *ShAmt) {
  const APInt &CmpRHSV = cast<ConstantInt>(ICI.getOperand(1))->getValue();

  // Check that the shift amount is in range.  If not, don't perform
  // undefined shifts.  When the shift is visited it will be
  // simplified.
  uint32_t TypeBits = CmpRHSV.getBitWidth();
  uint32_t ShAmtVal = (uint32_t)ShAmt->getLimitedValue(TypeBits);
  if (ShAmtVal >= TypeBits || ShAmtVal == 0)
    return 0;

  if (!ICI.isEquality()) {
    // If we have an unsigned comparison and an ashr, we can't simplify this.
    // Similarly for signed comparisons with lshr.
    if (ICI.isSigned() != (Shr->getOpcode() == Instruction::AShr))
      return 0;

    // Otherwise, all lshr and most exact ashr's are equivalent to a udiv/sdiv
    // by a power of 2.  Since we already have logic to simplify these,
    // transform to div and then simplify the resultant comparison.
    if (Shr->getOpcode() == Instruction::AShr &&
        (!Shr->isExact() || ShAmtVal == TypeBits - 1))
      return 0;

    // Revisit the shift (to delete it).
    Worklist.Add(Shr);

    Constant *DivCst =
      ConstantInt::get(Shr->getType(), APInt::getOneBitSet(TypeBits, ShAmtVal));

    Value *Tmp =
      Shr->getOpcode() == Instruction::AShr ?
      Builder->CreateSDiv(Shr->getOperand(0), DivCst, "", Shr->isExact()) :
      Builder->CreateUDiv(Shr->getOperand(0), DivCst, "", Shr->isExact());

    ICI.setOperand(0, Tmp);

    // If the builder folded the binop, just return it.
    BinaryOperator *TheDiv = dyn_cast<BinaryOperator>(Tmp);
    if (TheDiv == 0)
      return &ICI;

    // Otherwise, fold this div/compare.
    assert(TheDiv->getOpcode() == Instruction::SDiv ||
           TheDiv->getOpcode() == Instruction::UDiv);

    Instruction *Res = FoldICmpDivCst(ICI, TheDiv, cast<ConstantInt>(DivCst));
    assert(Res && "This div/cst should have folded!");
    return Res;
  }


  // If we are comparing against bits always shifted out, the
  // comparison cannot succeed.
  APInt Comp = CmpRHSV << ShAmtVal;
  ConstantInt *ShiftedCmpRHS = ConstantInt::get(ICI.getContext(), Comp);
  if (Shr->getOpcode() == Instruction::LShr)
    Comp = Comp.lshr(ShAmtVal);
  else
    Comp = Comp.ashr(ShAmtVal);

  if (Comp != CmpRHSV) { // Comparing against a bit that we know is zero.
    bool IsICMP_NE = ICI.getPredicate() == ICmpInst::ICMP_NE;
    Constant *Cst = ConstantInt::get(Type::getInt1Ty(ICI.getContext()),
                                     IsICMP_NE);
    return ReplaceInstUsesWith(ICI, Cst);
  }

  // Otherwise, check to see if the bits shifted out are known to be zero.
  // If so, we can compare against the unshifted value:
  //  (X & 4) >> 1 == 2  --> (X & 4) == 4.
  if (Shr->hasOneUse() && Shr->isExact())
    return new ICmpInst(ICI.getPredicate(), Shr->getOperand(0), ShiftedCmpRHS);

  if (Shr->hasOneUse()) {
    // Otherwise strength reduce the shift into an and.
    APInt Val(APInt::getHighBitsSet(TypeBits, TypeBits - ShAmtVal));
    Constant *Mask = ConstantInt::get(ICI.getContext(), Val);

    Value *And = Builder->CreateAnd(Shr->getOperand(0),
                                    Mask, Shr->getName()+".mask");
    return new ICmpInst(ICI.getPredicate(), And, ShiftedCmpRHS);
  }
  return 0;
}


/// visitICmpInstWithInstAndIntCst - Handle "icmp (instr, intcst)".
///
Instruction *InstCombiner::visitICmpInstWithInstAndIntCst(ICmpInst &ICI,
                                                          Instruction *LHSI,
                                                          ConstantInt *RHS) {
  const APInt &RHSV = RHS->getValue();

  switch (LHSI->getOpcode()) {
  case Instruction::Trunc:
    if (ICI.isEquality() && LHSI->hasOneUse()) {
      // Simplify icmp eq (trunc x to i8), 42 -> icmp eq x, 42|highbits if all
      // of the high bits truncated out of x are known.
      unsigned DstBits = LHSI->getType()->getPrimitiveSizeInBits(),
             SrcBits = LHSI->getOperand(0)->getType()->getPrimitiveSizeInBits();
      APInt KnownZero(SrcBits, 0), KnownOne(SrcBits, 0);
      ComputeMaskedBits(LHSI->getOperand(0), KnownZero, KnownOne);

      // If all the high bits are known, we can do this xform.
      if ((KnownZero|KnownOne).countLeadingOnes() >= SrcBits-DstBits) {
        // Pull in the high bits from known-ones set.
        APInt NewRHS = RHS->getValue().zext(SrcBits);
        NewRHS |= KnownOne & APInt::getHighBitsSet(SrcBits, SrcBits-DstBits);
        return new ICmpInst(ICI.getPredicate(), LHSI->getOperand(0),
                            ConstantInt::get(ICI.getContext(), NewRHS));
      }
    }
    break;

  case Instruction::Xor:         // (icmp pred (xor X, XorCST), CI)
    if (ConstantInt *XorCST = dyn_cast<ConstantInt>(LHSI->getOperand(1))) {
      // If this is a comparison that tests the signbit (X < 0) or (x > -1),
      // fold the xor.
      if ((ICI.getPredicate() == ICmpInst::ICMP_SLT && RHSV == 0) ||
          (ICI.getPredicate() == ICmpInst::ICMP_SGT && RHSV.isAllOnesValue())) {
        Value *CompareVal = LHSI->getOperand(0);

        // If the sign bit of the XorCST is not set, there is no change to
        // the operation, just stop using the Xor.
        if (!XorCST->isNegative()) {
          ICI.setOperand(0, CompareVal);
          Worklist.Add(LHSI);
          return &ICI;
        }

        // Was the old condition true if the operand is positive?
        bool isTrueIfPositive = ICI.getPredicate() == ICmpInst::ICMP_SGT;

        // If so, the new one isn't.
        isTrueIfPositive ^= true;

        if (isTrueIfPositive)
          return new ICmpInst(ICmpInst::ICMP_SGT, CompareVal,
                              SubOne(RHS));
        else
          return new ICmpInst(ICmpInst::ICMP_SLT, CompareVal,
                              AddOne(RHS));
      }

      if (LHSI->hasOneUse()) {
        // (icmp u/s (xor A SignBit), C) -> (icmp s/u A, (xor C SignBit))
        if (!ICI.isEquality() && XorCST->getValue().isSignBit()) {
          const APInt &SignBit = XorCST->getValue();
          ICmpInst::Predicate Pred = ICI.isSigned()
                                         ? ICI.getUnsignedPredicate()
                                         : ICI.getSignedPredicate();
          return new ICmpInst(Pred, LHSI->getOperand(0),
                              ConstantInt::get(ICI.getContext(),
                                               RHSV ^ SignBit));
        }

        // (icmp u/s (xor A ~SignBit), C) -> (icmp s/u (xor C ~SignBit), A)
        if (!ICI.isEquality() && XorCST->isMaxValue(true)) {
          const APInt &NotSignBit = XorCST->getValue();
          ICmpInst::Predicate Pred = ICI.isSigned()
                                         ? ICI.getUnsignedPredicate()
                                         : ICI.getSignedPredicate();
          Pred = ICI.getSwappedPredicate(Pred);
          return new ICmpInst(Pred, LHSI->getOperand(0),
                              ConstantInt::get(ICI.getContext(),
                                               RHSV ^ NotSignBit));
        }
      }
    }
    break;
  case Instruction::And:         // (icmp pred (and X, AndCST), RHS)
    if (LHSI->hasOneUse() && isa<ConstantInt>(LHSI->getOperand(1)) &&
        LHSI->getOperand(0)->hasOneUse()) {
      ConstantInt *AndCST = cast<ConstantInt>(LHSI->getOperand(1));

      // If the LHS is an AND of a truncating cast, we can widen the
      // and/compare to be the input width without changing the value
      // produced, eliminating a cast.
      if (TruncInst *Cast = dyn_cast<TruncInst>(LHSI->getOperand(0))) {
        // We can do this transformation if either the AND constant does not
        // have its sign bit set or if it is an equality comparison.
        // Extending a relational comparison when we're checking the sign
        // bit would not work.
        if (ICI.isEquality() ||
            (!AndCST->isNegative() && RHSV.isNonNegative())) {
          Value *NewAnd =
            Builder->CreateAnd(Cast->getOperand(0),
                               ConstantExpr::getZExt(AndCST, Cast->getSrcTy()));
          NewAnd->takeName(LHSI);
          return new ICmpInst(ICI.getPredicate(), NewAnd,
                              ConstantExpr::getZExt(RHS, Cast->getSrcTy()));
        }
      }

      // If the LHS is an AND of a zext, and we have an equality compare, we can
      // shrink the and/compare to the smaller type, eliminating the cast.
      if (ZExtInst *Cast = dyn_cast<ZExtInst>(LHSI->getOperand(0))) {
        IntegerType *Ty = cast<IntegerType>(Cast->getSrcTy());
        // Make sure we don't compare the upper bits, SimplifyDemandedBits
        // should fold the icmp to true/false in that case.
        if (ICI.isEquality() && RHSV.getActiveBits() <= Ty->getBitWidth()) {
          Value *NewAnd =
            Builder->CreateAnd(Cast->getOperand(0),
                               ConstantExpr::getTrunc(AndCST, Ty));
          NewAnd->takeName(LHSI);
          return new ICmpInst(ICI.getPredicate(), NewAnd,
                              ConstantExpr::getTrunc(RHS, Ty));
        }
      }

      // If this is: (X >> C1) & C2 != C3 (where any shift and any compare
      // could exist), turn it into (X & (C2 << C1)) != (C3 << C1).  This
      // happens a LOT in code produced by the C front-end, for bitfield
      // access.
      BinaryOperator *Shift = dyn_cast<BinaryOperator>(LHSI->getOperand(0));
      if (Shift && !Shift->isShift())
        Shift = 0;

      ConstantInt *ShAmt;
      ShAmt = Shift ? dyn_cast<ConstantInt>(Shift->getOperand(1)) : 0;
      Type *Ty = Shift ? Shift->getType() : 0;  // Type of the shift.
      Type *AndTy = AndCST->getType();          // Type of the and.

      // We can fold this as long as we can't shift unknown bits
      // into the mask.  This can only happen with signed shift
      // rights, as they sign-extend.
      if (ShAmt) {
        bool CanFold = Shift->isLogicalShift();
        if (!CanFold) {
          // To test for the bad case of the signed shr, see if any
          // of the bits shifted in could be tested after the mask.
          uint32_t TyBits = Ty->getPrimitiveSizeInBits();
          int ShAmtVal = TyBits - ShAmt->getLimitedValue(TyBits);

          uint32_t BitWidth = AndTy->getPrimitiveSizeInBits();
          if ((APInt::getHighBitsSet(BitWidth, BitWidth-ShAmtVal) &
               AndCST->getValue()) == 0)
            CanFold = true;
        }

        if (CanFold) {
          Constant *NewCst;
          if (Shift->getOpcode() == Instruction::Shl)
            NewCst = ConstantExpr::getLShr(RHS, ShAmt);
          else
            NewCst = ConstantExpr::getShl(RHS, ShAmt);

          // Check to see if we are shifting out any of the bits being
          // compared.
          if (ConstantExpr::get(Shift->getOpcode(),
                                       NewCst, ShAmt) != RHS) {
            // If we shifted bits out, the fold is not going to work out.
            // As a special case, check to see if this means that the
            // result is always true or false now.
            if (ICI.getPredicate() == ICmpInst::ICMP_EQ)
              return ReplaceInstUsesWith(ICI,
                                       ConstantInt::getFalse(ICI.getContext()));
            if (ICI.getPredicate() == ICmpInst::ICMP_NE)
              return ReplaceInstUsesWith(ICI,
                                       ConstantInt::getTrue(ICI.getContext()));
          } else {
            ICI.setOperand(1, NewCst);
            Constant *NewAndCST;
            if (Shift->getOpcode() == Instruction::Shl)
              NewAndCST = ConstantExpr::getLShr(AndCST, ShAmt);
            else
              NewAndCST = ConstantExpr::getShl(AndCST, ShAmt);
            LHSI->setOperand(1, NewAndCST);
            LHSI->setOperand(0, Shift->getOperand(0));
            Worklist.Add(Shift); // Shift is dead.
            return &ICI;
          }
        }
      }

      // Turn ((X >> Y) & C) == 0  into  (X & (C << Y)) == 0.  The later is
      // preferable because it allows the C<<Y expression to be hoisted out
      // of a loop if Y is invariant and X is not.
      if (Shift && Shift->hasOneUse() && RHSV == 0 &&
          ICI.isEquality() && !Shift->isArithmeticShift() &&
          !isa<Constant>(Shift->getOperand(0))) {
        // Compute C << Y.
        Value *NS;
        if (Shift->getOpcode() == Instruction::LShr) {
          NS = Builder->CreateShl(AndCST, Shift->getOperand(1));
        } else {
          // Insert a logical shift.
          NS = Builder->CreateLShr(AndCST, Shift->getOperand(1));
        }

        // Compute X & (C << Y).
        Value *NewAnd =
          Builder->CreateAnd(Shift->getOperand(0), NS, LHSI->getName());

        ICI.setOperand(0, NewAnd);
        return &ICI;
      }
    }

    // Try to optimize things like "A[i]&42 == 0" to index computations.
    if (LoadInst *LI = dyn_cast<LoadInst>(LHSI->getOperand(0))) {
      if (GetElementPtrInst *GEP =
          dyn_cast<GetElementPtrInst>(LI->getOperand(0)))
        if (GlobalVariable *GV = dyn_cast<GlobalVariable>(GEP->getOperand(0)))
          if (GV->isConstant() && GV->hasDefinitiveInitializer() &&
              !LI->isVolatile() && isa<ConstantInt>(LHSI->getOperand(1))) {
            ConstantInt *C = cast<ConstantInt>(LHSI->getOperand(1));
            if (Instruction *Res = FoldCmpLoadFromIndexedGlobal(GEP, GV,ICI, C))
              return Res;
          }
    }
    break;

  case Instruction::Or: {
    if (!ICI.isEquality() || !RHS->isNullValue() || !LHSI->hasOneUse())
      break;
    Value *P, *Q;
    if (match(LHSI, m_Or(m_PtrToInt(m_Value(P)), m_PtrToInt(m_Value(Q))))) {
      // Simplify icmp eq (or (ptrtoint P), (ptrtoint Q)), 0
      // -> and (icmp eq P, null), (icmp eq Q, null).
      Value *ICIP = Builder->CreateICmp(ICI.getPredicate(), P,
                                        Constant::getNullValue(P->getType()));
      Value *ICIQ = Builder->CreateICmp(ICI.getPredicate(), Q,
                                        Constant::getNullValue(Q->getType()));
      Instruction *Op;
      if (ICI.getPredicate() == ICmpInst::ICMP_EQ)
        Op = BinaryOperator::CreateAnd(ICIP, ICIQ);
      else
        Op = BinaryOperator::CreateOr(ICIP, ICIQ);
      return Op;
    }
    break;
  }

  case Instruction::Shl: {       // (icmp pred (shl X, ShAmt), CI)
    ConstantInt *ShAmt = dyn_cast<ConstantInt>(LHSI->getOperand(1));
    if (!ShAmt) break;

    uint32_t TypeBits = RHSV.getBitWidth();

    // Check that the shift amount is in range.  If not, don't perform
    // undefined shifts.  When the shift is visited it will be
    // simplified.
    if (ShAmt->uge(TypeBits))
      break;

    if (ICI.isEquality()) {
      // If we are comparing against bits always shifted out, the
      // comparison cannot succeed.
      Constant *Comp =
        ConstantExpr::getShl(ConstantExpr::getLShr(RHS, ShAmt),
                                                                 ShAmt);
      if (Comp != RHS) {// Comparing against a bit that we know is zero.
        bool IsICMP_NE = ICI.getPredicate() == ICmpInst::ICMP_NE;
        Constant *Cst =
          ConstantInt::get(Type::getInt1Ty(ICI.getContext()), IsICMP_NE);
        return ReplaceInstUsesWith(ICI, Cst);
      }

      // If the shift is NUW, then it is just shifting out zeros, no need for an
      // AND.
      if (cast<BinaryOperator>(LHSI)->hasNoUnsignedWrap())
        return new ICmpInst(ICI.getPredicate(), LHSI->getOperand(0),
                            ConstantExpr::getLShr(RHS, ShAmt));

      if (LHSI->hasOneUse()) {
        // Otherwise strength reduce the shift into an and.
        uint32_t ShAmtVal = (uint32_t)ShAmt->getLimitedValue(TypeBits);
        Constant *Mask =
          ConstantInt::get(ICI.getContext(), APInt::getLowBitsSet(TypeBits,
                                                       TypeBits-ShAmtVal));

        Value *And =
          Builder->CreateAnd(LHSI->getOperand(0),Mask, LHSI->getName()+".mask");
        return new ICmpInst(ICI.getPredicate(), And,
                            ConstantExpr::getLShr(RHS, ShAmt));
      }
    }

    // Otherwise, if this is a comparison of the sign bit, simplify to and/test.
    bool TrueIfSigned = false;
    if (LHSI->hasOneUse() &&
        isSignBitCheck(ICI.getPredicate(), RHS, TrueIfSigned)) {
      // (X << 31) <s 0  --> (X&1) != 0
      Constant *Mask = ConstantInt::get(LHSI->getOperand(0)->getType(),
                                        APInt::getOneBitSet(TypeBits,
                                            TypeBits-ShAmt->getZExtValue()-1));
      Value *And =
        Builder->CreateAnd(LHSI->getOperand(0), Mask, LHSI->getName()+".mask");
      return new ICmpInst(TrueIfSigned ? ICmpInst::ICMP_NE : ICmpInst::ICMP_EQ,
                          And, Constant::getNullValue(And->getType()));
    }
    break;
  }

  case Instruction::LShr:         // (icmp pred (shr X, ShAmt), CI)
  case Instruction::AShr: {
    // Handle equality comparisons of shift-by-constant.
    BinaryOperator *BO = cast<BinaryOperator>(LHSI);
    if (ConstantInt *ShAmt = dyn_cast<ConstantInt>(LHSI->getOperand(1))) {
      if (Instruction *Res = FoldICmpShrCst(ICI, BO, ShAmt))
        return Res;
    }

    // Handle exact shr's.
    if (ICI.isEquality() && BO->isExact() && BO->hasOneUse()) {
      if (RHSV.isMinValue())
        return new ICmpInst(ICI.getPredicate(), BO->getOperand(0), RHS);
    }
    break;
  }

  case Instruction::SDiv:
  case Instruction::UDiv:
    // Fold: icmp pred ([us]div X, C1), C2 -> range test
    // Fold this div into the comparison, producing a range check.
    // Determine, based on the divide type, what the range is being
    // checked.  If there is an overflow on the low or high side, remember
    // it, otherwise compute the range [low, hi) bounding the new value.
    // See: InsertRangeTest above for the kinds of replacements possible.
    if (ConstantInt *DivRHS = dyn_cast<ConstantInt>(LHSI->getOperand(1)))
      if (Instruction *R = FoldICmpDivCst(ICI, cast<BinaryOperator>(LHSI),
                                          DivRHS))
        return R;
    break;

  case Instruction::Add:
    // Fold: icmp pred (add X, C1), C2
    if (!ICI.isEquality()) {
      ConstantInt *LHSC = dyn_cast<ConstantInt>(LHSI->getOperand(1));
      if (!LHSC) break;
      const APInt &LHSV = LHSC->getValue();

      ConstantRange CR = ICI.makeConstantRange(ICI.getPredicate(), RHSV)
                            .subtract(LHSV);

      if (ICI.isSigned()) {
        if (CR.getLower().isSignBit()) {
          return new ICmpInst(ICmpInst::ICMP_SLT, LHSI->getOperand(0),
                              ConstantInt::get(ICI.getContext(),CR.getUpper()));
        } else if (CR.getUpper().isSignBit()) {
          return new ICmpInst(ICmpInst::ICMP_SGE, LHSI->getOperand(0),
                              ConstantInt::get(ICI.getContext(),CR.getLower()));
        }
      } else {
        if (CR.getLower().isMinValue()) {
          return new ICmpInst(ICmpInst::ICMP_ULT, LHSI->getOperand(0),
                              ConstantInt::get(ICI.getContext(),CR.getUpper()));
        } else if (CR.getUpper().isMinValue()) {
          return new ICmpInst(ICmpInst::ICMP_UGE, LHSI->getOperand(0),
                              ConstantInt::get(ICI.getContext(),CR.getLower()));
        }
      }
    }
    break;
  }

  // Simplify icmp_eq and icmp_ne instructions with integer constant RHS.
  if (ICI.isEquality()) {
    bool isICMP_NE = ICI.getPredicate() == ICmpInst::ICMP_NE;

    // If the first operand is (add|sub|and|or|xor|rem) with a constant, and
    // the second operand is a constant, simplify a bit.
    if (BinaryOperator *BO = dyn_cast<BinaryOperator>(LHSI)) {
      switch (BO->getOpcode()) {
      case Instruction::SRem:
        // If we have a signed (X % (2^c)) == 0, turn it into an unsigned one.
        if (RHSV == 0 && isa<ConstantInt>(BO->getOperand(1)) &&BO->hasOneUse()){
          const APInt &V = cast<ConstantInt>(BO->getOperand(1))->getValue();
          if (V.sgt(1) && V.isPowerOf2()) {
            Value *NewRem =
              Builder->CreateURem(BO->getOperand(0), BO->getOperand(1),
                                  BO->getName());
            return new ICmpInst(ICI.getPredicate(), NewRem,
                                Constant::getNullValue(BO->getType()));
          }
        }
        break;
      case Instruction::Add:
        // Replace ((add A, B) != C) with (A != C-B) if B & C are constants.
        if (ConstantInt *BOp1C = dyn_cast<ConstantInt>(BO->getOperand(1))) {
          if (BO->hasOneUse())
            return new ICmpInst(ICI.getPredicate(), BO->getOperand(0),
                                ConstantExpr::getSub(RHS, BOp1C));
        } else if (RHSV == 0) {
          // Replace ((add A, B) != 0) with (A != -B) if A or B is
          // efficiently invertible, or if the add has just this one use.
          Value *BOp0 = BO->getOperand(0), *BOp1 = BO->getOperand(1);

          if (Value *NegVal = dyn_castNegVal(BOp1))
            return new ICmpInst(ICI.getPredicate(), BOp0, NegVal);
          if (Value *NegVal = dyn_castNegVal(BOp0))
            return new ICmpInst(ICI.getPredicate(), NegVal, BOp1);
          if (BO->hasOneUse()) {
            Value *Neg = Builder->CreateNeg(BOp1);
            Neg->takeName(BO);
            return new ICmpInst(ICI.getPredicate(), BOp0, Neg);
          }
        }
        break;
      case Instruction::Xor:
        // For the xor case, we can xor two constants together, eliminating
        // the explicit xor.
        if (Constant *BOC = dyn_cast<Constant>(BO->getOperand(1))) {
          return new ICmpInst(ICI.getPredicate(), BO->getOperand(0),
                              ConstantExpr::getXor(RHS, BOC));
        } else if (RHSV == 0) {
          // Replace ((xor A, B) != 0) with (A != B)
          return new ICmpInst(ICI.getPredicate(), BO->getOperand(0),
                              BO->getOperand(1));
        }
        break;
      case Instruction::Sub:
        // Replace ((sub A, B) != C) with (B != A-C) if A & C are constants.
        if (ConstantInt *BOp0C = dyn_cast<ConstantInt>(BO->getOperand(0))) {
          if (BO->hasOneUse())
            return new ICmpInst(ICI.getPredicate(), BO->getOperand(1),
                                ConstantExpr::getSub(BOp0C, RHS));
        } else if (RHSV == 0) {
          // Replace ((sub A, B) != 0) with (A != B)
          return new ICmpInst(ICI.getPredicate(), BO->getOperand(0),
                              BO->getOperand(1));
        }
        break;
      case Instruction::Or:
        // If bits are being or'd in that are not present in the constant we
        // are comparing against, then the comparison could never succeed!
        if (ConstantInt *BOC = dyn_cast<ConstantInt>(BO->getOperand(1))) {
          Constant *NotCI = ConstantExpr::getNot(RHS);
          if (!ConstantExpr::getAnd(BOC, NotCI)->isNullValue())
            return ReplaceInstUsesWith(ICI,
                             ConstantInt::get(Type::getInt1Ty(ICI.getContext()),
                                       isICMP_NE));
        }
        break;

      case Instruction::And:
        if (ConstantInt *BOC = dyn_cast<ConstantInt>(BO->getOperand(1))) {
          // If bits are being compared against that are and'd out, then the
          // comparison can never succeed!
          if ((RHSV & ~BOC->getValue()) != 0)
            return ReplaceInstUsesWith(ICI,
                             ConstantInt::get(Type::getInt1Ty(ICI.getContext()),
                                       isICMP_NE));

          // If we have ((X & C) == C), turn it into ((X & C) != 0).
          if (RHS == BOC && RHSV.isPowerOf2())
            return new ICmpInst(isICMP_NE ? ICmpInst::ICMP_EQ :
                                ICmpInst::ICMP_NE, LHSI,
                                Constant::getNullValue(RHS->getType()));

          // Don't perform the following transforms if the AND has multiple uses
          if (!BO->hasOneUse())
            break;

          // Replace (and X, (1 << size(X)-1) != 0) with x s< 0
          if (BOC->getValue().isSignBit()) {
            Value *X = BO->getOperand(0);
            Constant *Zero = Constant::getNullValue(X->getType());
            ICmpInst::Predicate pred = isICMP_NE ?
              ICmpInst::ICMP_SLT : ICmpInst::ICMP_SGE;
            return new ICmpInst(pred, X, Zero);
          }

          // ((X & ~7) == 0) --> X < 8
          if (RHSV == 0 && isHighOnes(BOC)) {
            Value *X = BO->getOperand(0);
            Constant *NegX = ConstantExpr::getNeg(BOC);
            ICmpInst::Predicate pred = isICMP_NE ?
              ICmpInst::ICMP_UGE : ICmpInst::ICMP_ULT;
            return new ICmpInst(pred, X, NegX);
          }
        }
      default: break;
      }
    } else if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(LHSI)) {
      // Handle icmp {eq|ne} <intrinsic>, intcst.
      switch (II->getIntrinsicID()) {
      case Intrinsic::bswap:
        Worklist.Add(II);
        ICI.setOperand(0, II->getArgOperand(0));
        ICI.setOperand(1, ConstantInt::get(II->getContext(), RHSV.byteSwap()));
        return &ICI;
      case Intrinsic::ctlz:
      case Intrinsic::cttz:
        // ctz(A) == bitwidth(a)  ->  A == 0 and likewise for !=
        if (RHSV == RHS->getType()->getBitWidth()) {
          Worklist.Add(II);
          ICI.setOperand(0, II->getArgOperand(0));
          ICI.setOperand(1, ConstantInt::get(RHS->getType(), 0));
          return &ICI;
        }
        break;
      case Intrinsic::ctpop:
        // popcount(A) == 0  ->  A == 0 and likewise for !=
        if (RHS->isZero()) {
          Worklist.Add(II);
          ICI.setOperand(0, II->getArgOperand(0));
          ICI.setOperand(1, RHS);
          return &ICI;
        }
        break;
      default:
        break;
      }
    }
  }
  return 0;
}

/// visitICmpInstWithCastAndCast - Handle icmp (cast x to y), (cast/cst).
/// We only handle extending casts so far.
///
Instruction *InstCombiner::visitICmpInstWithCastAndCast(ICmpInst &ICI) {
  const CastInst *LHSCI = cast<CastInst>(ICI.getOperand(0));
  Value *LHSCIOp        = LHSCI->getOperand(0);
  Type *SrcTy     = LHSCIOp->getType();
  Type *DestTy    = LHSCI->getType();
  Value *RHSCIOp;

  // Turn icmp (ptrtoint x), (ptrtoint/c) into a compare of the input if the
  // integer type is the same size as the pointer type.
  if (TD && LHSCI->getOpcode() == Instruction::PtrToInt &&
      TD->getPointerSizeInBits() ==
         cast<IntegerType>(DestTy)->getBitWidth()) {
    Value *RHSOp = 0;
    if (Constant *RHSC = dyn_cast<Constant>(ICI.getOperand(1))) {
      RHSOp = ConstantExpr::getIntToPtr(RHSC, SrcTy);
    } else if (PtrToIntInst *RHSC = dyn_cast<PtrToIntInst>(ICI.getOperand(1))) {
      RHSOp = RHSC->getOperand(0);
      // If the pointer types don't match, insert a bitcast.
      if (LHSCIOp->getType() != RHSOp->getType())
        RHSOp = Builder->CreateBitCast(RHSOp, LHSCIOp->getType());
    }

    if (RHSOp)
      return new ICmpInst(ICI.getPredicate(), LHSCIOp, RHSOp);
  }

  // The code below only handles extension cast instructions, so far.
  // Enforce this.
  if (LHSCI->getOpcode() != Instruction::ZExt &&
      LHSCI->getOpcode() != Instruction::SExt)
    return 0;

  bool isSignedExt = LHSCI->getOpcode() == Instruction::SExt;
  bool isSignedCmp = ICI.isSigned();

  if (CastInst *CI = dyn_cast<CastInst>(ICI.getOperand(1))) {
    // Not an extension from the same type?
    RHSCIOp = CI->getOperand(0);
    if (RHSCIOp->getType() != LHSCIOp->getType())
      return 0;

    // If the signedness of the two casts doesn't agree (i.e. one is a sext
    // and the other is a zext), then we can't handle this.
    if (CI->getOpcode() != LHSCI->getOpcode())
      return 0;

    // Deal with equality cases early.
    if (ICI.isEquality())
      return new ICmpInst(ICI.getPredicate(), LHSCIOp, RHSCIOp);

    // A signed comparison of sign extended values simplifies into a
    // signed comparison.
    if (isSignedCmp && isSignedExt)
      return new ICmpInst(ICI.getPredicate(), LHSCIOp, RHSCIOp);

    // The other three cases all fold into an unsigned comparison.
    return new ICmpInst(ICI.getUnsignedPredicate(), LHSCIOp, RHSCIOp);
  }

  // If we aren't dealing with a constant on the RHS, exit early
  ConstantInt *CI = dyn_cast<ConstantInt>(ICI.getOperand(1));
  if (!CI)
    return 0;

  // Compute the constant that would happen if we truncated to SrcTy then
  // reextended to DestTy.
  Constant *Res1 = ConstantExpr::getTrunc(CI, SrcTy);
  Constant *Res2 = ConstantExpr::getCast(LHSCI->getOpcode(),
                                                Res1, DestTy);

  // If the re-extended constant didn't change...
  if (Res2 == CI) {
    // Deal with equality cases early.
    if (ICI.isEquality())
      return new ICmpInst(ICI.getPredicate(), LHSCIOp, Res1);

    // A signed comparison of sign extended values simplifies into a
    // signed comparison.
    if (isSignedExt && isSignedCmp)
      return new ICmpInst(ICI.getPredicate(), LHSCIOp, Res1);

    // The other three cases all fold into an unsigned comparison.
    return new ICmpInst(ICI.getUnsignedPredicate(), LHSCIOp, Res1);
  }

  // The re-extended constant changed so the constant cannot be represented
  // in the shorter type. Consequently, we cannot emit a simple comparison.
  // All the cases that fold to true or false will have already been handled
  // by SimplifyICmpInst, so only deal with the tricky case.

  if (isSignedCmp || !isSignedExt)
    return 0;

  // Evaluate the comparison for LT (we invert for GT below). LE and GE cases
  // should have been folded away previously and not enter in here.

  // We're performing an unsigned comp with a sign extended value.
  // This is true if the input is >= 0. [aka >s -1]
  Constant *NegOne = Constant::getAllOnesValue(SrcTy);
  Value *Result = Builder->CreateICmpSGT(LHSCIOp, NegOne, ICI.getName());

  // Finally, return the value computed.
  if (ICI.getPredicate() == ICmpInst::ICMP_ULT)
    return ReplaceInstUsesWith(ICI, Result);

  assert(ICI.getPredicate() == ICmpInst::ICMP_UGT && "ICmp should be folded!");
  return BinaryOperator::CreateNot(Result);
}

/// ProcessUGT_ADDCST_ADD - The caller has matched a pattern of the form:
///   I = icmp ugt (add (add A, B), CI2), CI1
/// If this is of the form:
///   sum = a + b
///   if (sum+128 >u 255)
/// Then replace it with llvm.sadd.with.overflow.i8.
///
static Instruction *ProcessUGT_ADDCST_ADD(ICmpInst &I, Value *A, Value *B,
                                          ConstantInt *CI2, ConstantInt *CI1,
                                          InstCombiner &IC) {
  // The transformation we're trying to do here is to transform this into an
  // llvm.sadd.with.overflow.  To do this, we have to replace the original add
  // with a narrower add, and discard the add-with-constant that is part of the
  // range check (if we can't eliminate it, this isn't profitable).

  // In order to eliminate the add-with-constant, the compare can be its only
  // use.
  Instruction *AddWithCst = cast<Instruction>(I.getOperand(0));
  if (!AddWithCst->hasOneUse()) return 0;

  // If CI2 is 2^7, 2^15, 2^31, then it might be an sadd.with.overflow.
  if (!CI2->getValue().isPowerOf2()) return 0;
  unsigned NewWidth = CI2->getValue().countTrailingZeros();
  if (NewWidth != 7 && NewWidth != 15 && NewWidth != 31) return 0;

  // The width of the new add formed is 1 more than the bias.
  ++NewWidth;

  // Check to see that CI1 is an all-ones value with NewWidth bits.
  if (CI1->getBitWidth() == NewWidth ||
      CI1->getValue() != APInt::getLowBitsSet(CI1->getBitWidth(), NewWidth))
    return 0;

  // This is only really a signed overflow check if the inputs have been
  // sign-extended; check for that condition. For example, if CI2 is 2^31 and
  // the operands of the add are 64 bits wide, we need at least 33 sign bits.
  unsigned NeededSignBits = CI1->getBitWidth() - NewWidth + 1;
  if (IC.ComputeNumSignBits(A) < NeededSignBits ||
      IC.ComputeNumSignBits(B) < NeededSignBits)
    return 0;

  // In order to replace the original add with a narrower
  // llvm.sadd.with.overflow, the only uses allowed are the add-with-constant
  // and truncates that discard the high bits of the add.  Verify that this is
  // the case.
  Instruction *OrigAdd = cast<Instruction>(AddWithCst->getOperand(0));
  for (Value::use_iterator UI = OrigAdd->use_begin(), E = OrigAdd->use_end();
       UI != E; ++UI) {
    if (*UI == AddWithCst) continue;

    // Only accept truncates for now.  We would really like a nice recursive
    // predicate like SimplifyDemandedBits, but which goes downwards the use-def
    // chain to see which bits of a value are actually demanded.  If the
    // original add had another add which was then immediately truncated, we
    // could still do the transformation.
    TruncInst *TI = dyn_cast<TruncInst>(*UI);
    if (TI == 0 ||
        TI->getType()->getPrimitiveSizeInBits() > NewWidth) return 0;
  }

  // If the pattern matches, truncate the inputs to the narrower type and
  // use the sadd_with_overflow intrinsic to efficiently compute both the
  // result and the overflow bit.
  Module *M = I.getParent()->getParent()->getParent();

  Type *NewType = IntegerType::get(OrigAdd->getContext(), NewWidth);
  Value *F = Intrinsic::getDeclaration(M, Intrinsic::sadd_with_overflow,
                                       NewType);

  InstCombiner::BuilderTy *Builder = IC.Builder;

  // Put the new code above the original add, in case there are any uses of the
  // add between the add and the compare.
  Builder->SetInsertPoint(OrigAdd);

  Value *TruncA = Builder->CreateTrunc(A, NewType, A->getName()+".trunc");
  Value *TruncB = Builder->CreateTrunc(B, NewType, B->getName()+".trunc");
  CallInst *Call = Builder->CreateCall2(F, TruncA, TruncB, "sadd");
  Value *Add = Builder->CreateExtractValue(Call, 0, "sadd.result");
  Value *ZExt = Builder->CreateZExt(Add, OrigAdd->getType());

  // The inner add was the result of the narrow add, zero extended to the
  // wider type.  Replace it with the result computed by the intrinsic.
  IC.ReplaceInstUsesWith(*OrigAdd, ZExt);

  // The original icmp gets replaced with the overflow value.
  return ExtractValueInst::Create(Call, 1, "sadd.overflow");
}

static Instruction *ProcessUAddIdiom(Instruction &I, Value *OrigAddV,
                                     InstCombiner &IC) {
  // Don't bother doing this transformation for pointers, don't do it for
  // vectors.
  if (!isa<IntegerType>(OrigAddV->getType())) return 0;

  // If the add is a constant expr, then we don't bother transforming it.
  Instruction *OrigAdd = dyn_cast<Instruction>(OrigAddV);
  if (OrigAdd == 0) return 0;

  Value *LHS = OrigAdd->getOperand(0), *RHS = OrigAdd->getOperand(1);

  // Put the new code above the original add, in case there are any uses of the
  // add between the add and the compare.
  InstCombiner::BuilderTy *Builder = IC.Builder;
  Builder->SetInsertPoint(OrigAdd);

  Module *M = I.getParent()->getParent()->getParent();
  Type *Ty = LHS->getType();
  Value *F = Intrinsic::getDeclaration(M, Intrinsic::uadd_with_overflow, Ty);
  CallInst *Call = Builder->CreateCall2(F, LHS, RHS, "uadd");
  Value *Add = Builder->CreateExtractValue(Call, 0);

  IC.ReplaceInstUsesWith(*OrigAdd, Add);

  // The original icmp gets replaced with the overflow value.
  return ExtractValueInst::Create(Call, 1, "uadd.overflow");
}

// DemandedBitsLHSMask - When performing a comparison against a constant,
// it is possible that not all the bits in the LHS are demanded.  This helper
// method computes the mask that IS demanded.
static APInt DemandedBitsLHSMask(ICmpInst &I,
                                 unsigned BitWidth, bool isSignCheck) {
  if (isSignCheck)
    return APInt::getSignBit(BitWidth);

  ConstantInt *CI = dyn_cast<ConstantInt>(I.getOperand(1));
  if (!CI) return APInt::getAllOnesValue(BitWidth);
  const APInt &RHS = CI->getValue();

  switch (I.getPredicate()) {
  // For a UGT comparison, we don't care about any bits that
  // correspond to the trailing ones of the comparand.  The value of these
  // bits doesn't impact the outcome of the comparison, because any value
  // greater than the RHS must differ in a bit higher than these due to carry.
  case ICmpInst::ICMP_UGT: {
    unsigned trailingOnes = RHS.countTrailingOnes();
    APInt lowBitsSet = APInt::getLowBitsSet(BitWidth, trailingOnes);
    return ~lowBitsSet;
  }

  // Similarly, for a ULT comparison, we don't care about the trailing zeros.
  // Any value less than the RHS must differ in a higher bit because of carries.
  case ICmpInst::ICMP_ULT: {
    unsigned trailingZeros = RHS.countTrailingZeros();
    APInt lowBitsSet = APInt::getLowBitsSet(BitWidth, trailingZeros);
    return ~lowBitsSet;
  }

  default:
    return APInt::getAllOnesValue(BitWidth);
  }

}

Instruction *InstCombiner::visitICmpInst(ICmpInst &I) {
  bool Changed = false;
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  /// Orders the operands of the compare so that they are listed from most
  /// complex to least complex.  This puts constants before unary operators,
  /// before binary operators.
  if (getComplexity(Op0) < getComplexity(Op1)) {
    I.swapOperands();
    std::swap(Op0, Op1);
    Changed = true;
  }

  if (Value *V = SimplifyICmpInst(I.getPredicate(), Op0, Op1, TD))
    return ReplaceInstUsesWith(I, V);

  // comparing -val or val with non-zero is the same as just comparing val
  // ie, abs(val) != 0 -> val != 0
  if (I.getPredicate() == ICmpInst::ICMP_NE && match(Op1, m_Zero()))
  {
    Value *Cond, *SelectTrue, *SelectFalse;
    if (match(Op0, m_Select(m_Value(Cond), m_Value(SelectTrue),
                            m_Value(SelectFalse)))) {
      if (Value *V = dyn_castNegVal(SelectTrue)) {
        if (V == SelectFalse)
          return CmpInst::Create(Instruction::ICmp, I.getPredicate(), V, Op1);
      }
      else if (Value *V = dyn_castNegVal(SelectFalse)) {
        if (V == SelectTrue)
          return CmpInst::Create(Instruction::ICmp, I.getPredicate(), V, Op1);
      }
    }
  }

  Type *Ty = Op0->getType();

  // icmp's with boolean values can always be turned into bitwise operations
  if (Ty->isIntegerTy(1)) {
    switch (I.getPredicate()) {
    default: llvm_unreachable("Invalid icmp instruction!");
    case ICmpInst::ICMP_EQ: {               // icmp eq i1 A, B -> ~(A^B)
      Value *Xor = Builder->CreateXor(Op0, Op1, I.getName()+"tmp");
      return BinaryOperator::CreateNot(Xor);
    }
    case ICmpInst::ICMP_NE:                  // icmp eq i1 A, B -> A^B
      return BinaryOperator::CreateXor(Op0, Op1);

    case ICmpInst::ICMP_UGT:
      std::swap(Op0, Op1);                   // Change icmp ugt -> icmp ult
      // FALL THROUGH
    case ICmpInst::ICMP_ULT:{               // icmp ult i1 A, B -> ~A & B
      Value *Not = Builder->CreateNot(Op0, I.getName()+"tmp");
      return BinaryOperator::CreateAnd(Not, Op1);
    }
    case ICmpInst::ICMP_SGT:
      std::swap(Op0, Op1);                   // Change icmp sgt -> icmp slt
      // FALL THROUGH
    case ICmpInst::ICMP_SLT: {               // icmp slt i1 A, B -> A & ~B
      Value *Not = Builder->CreateNot(Op1, I.getName()+"tmp");
      return BinaryOperator::CreateAnd(Not, Op0);
    }
    case ICmpInst::ICMP_UGE:
      std::swap(Op0, Op1);                   // Change icmp uge -> icmp ule
      // FALL THROUGH
    case ICmpInst::ICMP_ULE: {               //  icmp ule i1 A, B -> ~A | B
      Value *Not = Builder->CreateNot(Op0, I.getName()+"tmp");
      return BinaryOperator::CreateOr(Not, Op1);
    }
    case ICmpInst::ICMP_SGE:
      std::swap(Op0, Op1);                   // Change icmp sge -> icmp sle
      // FALL THROUGH
    case ICmpInst::ICMP_SLE: {               //  icmp sle i1 A, B -> A | ~B
      Value *Not = Builder->CreateNot(Op1, I.getName()+"tmp");
      return BinaryOperator::CreateOr(Not, Op0);
    }
    }
  }

  unsigned BitWidth = 0;
  if (Ty->isIntOrIntVectorTy())
    BitWidth = Ty->getScalarSizeInBits();
  else if (TD)  // Pointers require TD info to get their size.
    BitWidth = TD->getTypeSizeInBits(Ty->getScalarType());

  bool isSignBit = false;

  // See if we are doing a comparison with a constant.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {
    Value *A = 0, *B = 0;

    // Match the following pattern, which is a common idiom when writing
    // overflow-safe integer arithmetic function.  The source performs an
    // addition in wider type, and explicitly checks for overflow using
    // comparisons against INT_MIN and INT_MAX.  Simplify this by using the
    // sadd_with_overflow intrinsic.
    //
    // TODO: This could probably be generalized to handle other overflow-safe
    // operations if we worked out the formulas to compute the appropriate
    // magic constants.
    //
    // sum = a + b
    // if (sum+128 >u 255)  ...  -> llvm.sadd.with.overflow.i8
    {
    ConstantInt *CI2;    // I = icmp ugt (add (add A, B), CI2), CI
    if (I.getPredicate() == ICmpInst::ICMP_UGT &&
        match(Op0, m_Add(m_Add(m_Value(A), m_Value(B)), m_ConstantInt(CI2))))
      if (Instruction *Res = ProcessUGT_ADDCST_ADD(I, A, B, CI2, CI, *this))
        return Res;
    }

    // (icmp ne/eq (sub A B) 0) -> (icmp ne/eq A, B)
    if (I.isEquality() && CI->isZero() &&
        match(Op0, m_Sub(m_Value(A), m_Value(B)))) {
      // (icmp cond A B) if cond is equality
      return new ICmpInst(I.getPredicate(), A, B);
    }

    // If we have an icmp le or icmp ge instruction, turn it into the
    // appropriate icmp lt or icmp gt instruction.  This allows us to rely on
    // them being folded in the code below.  The SimplifyICmpInst code has
    // already handled the edge cases for us, so we just assert on them.
    switch (I.getPredicate()) {
    default: break;
    case ICmpInst::ICMP_ULE:
      assert(!CI->isMaxValue(false));                 // A <=u MAX -> TRUE
      return new ICmpInst(ICmpInst::ICMP_ULT, Op0,
                          ConstantInt::get(CI->getContext(), CI->getValue()+1));
    case ICmpInst::ICMP_SLE:
      assert(!CI->isMaxValue(true));                  // A <=s MAX -> TRUE
      return new ICmpInst(ICmpInst::ICMP_SLT, Op0,
                          ConstantInt::get(CI->getContext(), CI->getValue()+1));
    case ICmpInst::ICMP_UGE:
      assert(!CI->isMinValue(false));                 // A >=u MIN -> TRUE
      return new ICmpInst(ICmpInst::ICMP_UGT, Op0,
                          ConstantInt::get(CI->getContext(), CI->getValue()-1));
    case ICmpInst::ICMP_SGE:
      assert(!CI->isMinValue(true));                  // A >=s MIN -> TRUE
      return new ICmpInst(ICmpInst::ICMP_SGT, Op0,
                          ConstantInt::get(CI->getContext(), CI->getValue()-1));
    }

    // If this comparison is a normal comparison, it demands all
    // bits, if it is a sign bit comparison, it only demands the sign bit.
    bool UnusedBit;
    isSignBit = isSignBitCheck(I.getPredicate(), CI, UnusedBit);
  }

  // See if we can fold the comparison based on range information we can get
  // by checking whether bits are known to be zero or one in the input.
  if (BitWidth != 0) {
    APInt Op0KnownZero(BitWidth, 0), Op0KnownOne(BitWidth, 0);
    APInt Op1KnownZero(BitWidth, 0), Op1KnownOne(BitWidth, 0);

    if (SimplifyDemandedBits(I.getOperandUse(0),
                             DemandedBitsLHSMask(I, BitWidth, isSignBit),
                             Op0KnownZero, Op0KnownOne, 0))
      return &I;
    if (SimplifyDemandedBits(I.getOperandUse(1),
                             APInt::getAllOnesValue(BitWidth),
                             Op1KnownZero, Op1KnownOne, 0))
      return &I;

    // Given the known and unknown bits, compute a range that the LHS could be
    // in.  Compute the Min, Max and RHS values based on the known bits. For the
    // EQ and NE we use unsigned values.
    APInt Op0Min(BitWidth, 0), Op0Max(BitWidth, 0);
    APInt Op1Min(BitWidth, 0), Op1Max(BitWidth, 0);
    if (I.isSigned()) {
      ComputeSignedMinMaxValuesFromKnownBits(Op0KnownZero, Op0KnownOne,
                                             Op0Min, Op0Max);
      ComputeSignedMinMaxValuesFromKnownBits(Op1KnownZero, Op1KnownOne,
                                             Op1Min, Op1Max);
    } else {
      ComputeUnsignedMinMaxValuesFromKnownBits(Op0KnownZero, Op0KnownOne,
                                               Op0Min, Op0Max);
      ComputeUnsignedMinMaxValuesFromKnownBits(Op1KnownZero, Op1KnownOne,
                                               Op1Min, Op1Max);
    }

    // If Min and Max are known to be the same, then SimplifyDemandedBits
    // figured out that the LHS is a constant.  Just constant fold this now so
    // that code below can assume that Min != Max.
    if (!isa<Constant>(Op0) && Op0Min == Op0Max)
      return new ICmpInst(I.getPredicate(),
                          ConstantInt::get(Op0->getType(), Op0Min), Op1);
    if (!isa<Constant>(Op1) && Op1Min == Op1Max)
      return new ICmpInst(I.getPredicate(), Op0,
                          ConstantInt::get(Op1->getType(), Op1Min));

    // Based on the range information we know about the LHS, see if we can
    // simplify this comparison.  For example, (x&4) < 8 is always true.
    switch (I.getPredicate()) {
    default: llvm_unreachable("Unknown icmp opcode!");
    case ICmpInst::ICMP_EQ: {
      if (Op0Max.ult(Op1Min) || Op0Min.ugt(Op1Max))
        return ReplaceInstUsesWith(I, ConstantInt::getFalse(I.getType()));

      // If all bits are known zero except for one, then we know at most one
      // bit is set.   If the comparison is against zero, then this is a check
      // to see if *that* bit is set.
      APInt Op0KnownZeroInverted = ~Op0KnownZero;
      if (~Op1KnownZero == 0 && Op0KnownZeroInverted.isPowerOf2()) {
        // If the LHS is an AND with the same constant, look through it.
        Value *LHS = 0;
        ConstantInt *LHSC = 0;
        if (!match(Op0, m_And(m_Value(LHS), m_ConstantInt(LHSC))) ||
            LHSC->getValue() != Op0KnownZeroInverted)
          LHS = Op0;

        // If the LHS is 1 << x, and we know the result is a power of 2 like 8,
        // then turn "((1 << x)&8) == 0" into "x != 3".
        Value *X = 0;
        if (match(LHS, m_Shl(m_One(), m_Value(X)))) {
          unsigned CmpVal = Op0KnownZeroInverted.countTrailingZeros();
          return new ICmpInst(ICmpInst::ICMP_NE, X,
                              ConstantInt::get(X->getType(), CmpVal));
        }

        // If the LHS is 8 >>u x, and we know the result is a power of 2 like 1,
        // then turn "((8 >>u x)&1) == 0" into "x != 3".
        const APInt *CI;
        if (Op0KnownZeroInverted == 1 &&
            match(LHS, m_LShr(m_Power2(CI), m_Value(X))))
          return new ICmpInst(ICmpInst::ICMP_NE, X,
                              ConstantInt::get(X->getType(),
                                               CI->countTrailingZeros()));
      }

      break;
    }
    case ICmpInst::ICMP_NE: {
      if (Op0Max.ult(Op1Min) || Op0Min.ugt(Op1Max))
        return ReplaceInstUsesWith(I, ConstantInt::getTrue(I.getType()));

      // If all bits are known zero except for one, then we know at most one
      // bit is set.   If the comparison is against zero, then this is a check
      // to see if *that* bit is set.
      APInt Op0KnownZeroInverted = ~Op0KnownZero;
      if (~Op1KnownZero == 0 && Op0KnownZeroInverted.isPowerOf2()) {
        // If the LHS is an AND with the same constant, look through it.
        Value *LHS = 0;
        ConstantInt *LHSC = 0;
        if (!match(Op0, m_And(m_Value(LHS), m_ConstantInt(LHSC))) ||
            LHSC->getValue() != Op0KnownZeroInverted)
          LHS = Op0;

        // If the LHS is 1 << x, and we know the result is a power of 2 like 8,
        // then turn "((1 << x)&8) != 0" into "x == 3".
        Value *X = 0;
        if (match(LHS, m_Shl(m_One(), m_Value(X)))) {
          unsigned CmpVal = Op0KnownZeroInverted.countTrailingZeros();
          return new ICmpInst(ICmpInst::ICMP_EQ, X,
                              ConstantInt::get(X->getType(), CmpVal));
        }

        // If the LHS is 8 >>u x, and we know the result is a power of 2 like 1,
        // then turn "((8 >>u x)&1) != 0" into "x == 3".
        const APInt *CI;
        if (Op0KnownZeroInverted == 1 &&
            match(LHS, m_LShr(m_Power2(CI), m_Value(X))))
          return new ICmpInst(ICmpInst::ICMP_EQ, X,
                              ConstantInt::get(X->getType(),
                                               CI->countTrailingZeros()));
      }

      break;
    }
    case ICmpInst::ICMP_ULT:
      if (Op0Max.ult(Op1Min))          // A <u B -> true if max(A) < min(B)
        return ReplaceInstUsesWith(I, ConstantInt::getTrue(I.getType()));
      if (Op0Min.uge(Op1Max))          // A <u B -> false if min(A) >= max(B)
        return ReplaceInstUsesWith(I, ConstantInt::getFalse(I.getType()));
      if (Op1Min == Op0Max)            // A <u B -> A != B if max(A) == min(B)
        return new ICmpInst(ICmpInst::ICMP_NE, Op0, Op1);
      if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {
        if (Op1Max == Op0Min+1)        // A <u C -> A == C-1 if min(A)+1 == C
          return new ICmpInst(ICmpInst::ICMP_EQ, Op0,
                          ConstantInt::get(CI->getContext(), CI->getValue()-1));

        // (x <u 2147483648) -> (x >s -1)  -> true if sign bit clear
        if (CI->isMinValue(true))
          return new ICmpInst(ICmpInst::ICMP_SGT, Op0,
                           Constant::getAllOnesValue(Op0->getType()));
      }
      break;
    case ICmpInst::ICMP_UGT:
      if (Op0Min.ugt(Op1Max))          // A >u B -> true if min(A) > max(B)
        return ReplaceInstUsesWith(I, ConstantInt::getTrue(I.getType()));
      if (Op0Max.ule(Op1Min))          // A >u B -> false if max(A) <= max(B)
        return ReplaceInstUsesWith(I, ConstantInt::getFalse(I.getType()));

      if (Op1Max == Op0Min)            // A >u B -> A != B if min(A) == max(B)
        return new ICmpInst(ICmpInst::ICMP_NE, Op0, Op1);
      if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {
        if (Op1Min == Op0Max-1)        // A >u C -> A == C+1 if max(a)-1 == C
          return new ICmpInst(ICmpInst::ICMP_EQ, Op0,
                          ConstantInt::get(CI->getContext(), CI->getValue()+1));

        // (x >u 2147483647) -> (x <s 0)  -> true if sign bit set
        if (CI->isMaxValue(true))
          return new ICmpInst(ICmpInst::ICMP_SLT, Op0,
                              Constant::getNullValue(Op0->getType()));
      }
      break;
    case ICmpInst::ICMP_SLT:
      if (Op0Max.slt(Op1Min))          // A <s B -> true if max(A) < min(C)
        return ReplaceInstUsesWith(I, ConstantInt::getTrue(I.getType()));
      if (Op0Min.sge(Op1Max))          // A <s B -> false if min(A) >= max(C)
        return ReplaceInstUsesWith(I, ConstantInt::getFalse(I.getType()));
      if (Op1Min == Op0Max)            // A <s B -> A != B if max(A) == min(B)
        return new ICmpInst(ICmpInst::ICMP_NE, Op0, Op1);
      if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {
        if (Op1Max == Op0Min+1)        // A <s C -> A == C-1 if min(A)+1 == C
          return new ICmpInst(ICmpInst::ICMP_EQ, Op0,
                          ConstantInt::get(CI->getContext(), CI->getValue()-1));
      }
      break;
    case ICmpInst::ICMP_SGT:
      if (Op0Min.sgt(Op1Max))          // A >s B -> true if min(A) > max(B)
        return ReplaceInstUsesWith(I, ConstantInt::getTrue(I.getType()));
      if (Op0Max.sle(Op1Min))          // A >s B -> false if max(A) <= min(B)
        return ReplaceInstUsesWith(I, ConstantInt::getFalse(I.getType()));

      if (Op1Max == Op0Min)            // A >s B -> A != B if min(A) == max(B)
        return new ICmpInst(ICmpInst::ICMP_NE, Op0, Op1);
      if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {
        if (Op1Min == Op0Max-1)        // A >s C -> A == C+1 if max(A)-1 == C
          return new ICmpInst(ICmpInst::ICMP_EQ, Op0,
                          ConstantInt::get(CI->getContext(), CI->getValue()+1));
      }
      break;
    case ICmpInst::ICMP_SGE:
      assert(!isa<ConstantInt>(Op1) && "ICMP_SGE with ConstantInt not folded!");
      if (Op0Min.sge(Op1Max))          // A >=s B -> true if min(A) >= max(B)
        return ReplaceInstUsesWith(I, ConstantInt::getTrue(I.getType()));
      if (Op0Max.slt(Op1Min))          // A >=s B -> false if max(A) < min(B)
        return ReplaceInstUsesWith(I, ConstantInt::getFalse(I.getType()));
      break;
    case ICmpInst::ICMP_SLE:
      assert(!isa<ConstantInt>(Op1) && "ICMP_SLE with ConstantInt not folded!");
      if (Op0Max.sle(Op1Min))          // A <=s B -> true if max(A) <= min(B)
        return ReplaceInstUsesWith(I, ConstantInt::getTrue(I.getType()));
      if (Op0Min.sgt(Op1Max))          // A <=s B -> false if min(A) > max(B)
        return ReplaceInstUsesWith(I, ConstantInt::getFalse(I.getType()));
      break;
    case ICmpInst::ICMP_UGE:
      assert(!isa<ConstantInt>(Op1) && "ICMP_UGE with ConstantInt not folded!");
      if (Op0Min.uge(Op1Max))          // A >=u B -> true if min(A) >= max(B)
        return ReplaceInstUsesWith(I, ConstantInt::getTrue(I.getType()));
      if (Op0Max.ult(Op1Min))          // A >=u B -> false if max(A) < min(B)
        return ReplaceInstUsesWith(I, ConstantInt::getFalse(I.getType()));
      break;
    case ICmpInst::ICMP_ULE:
      assert(!isa<ConstantInt>(Op1) && "ICMP_ULE with ConstantInt not folded!");
      if (Op0Max.ule(Op1Min))          // A <=u B -> true if max(A) <= min(B)
        return ReplaceInstUsesWith(I, ConstantInt::getTrue(I.getType()));
      if (Op0Min.ugt(Op1Max))          // A <=u B -> false if min(A) > max(B)
        return ReplaceInstUsesWith(I, ConstantInt::getFalse(I.getType()));
      break;
    }

    // Turn a signed comparison into an unsigned one if both operands
    // are known to have the same sign.
    if (I.isSigned() &&
        ((Op0KnownZero.isNegative() && Op1KnownZero.isNegative()) ||
         (Op0KnownOne.isNegative() && Op1KnownOne.isNegative())))
      return new ICmpInst(I.getUnsignedPredicate(), Op0, Op1);
  }

  // Test if the ICmpInst instruction is used exclusively by a select as
  // part of a minimum or maximum operation. If so, refrain from doing
  // any other folding. This helps out other analyses which understand
  // non-obfuscated minimum and maximum idioms, such as ScalarEvolution
  // and CodeGen. And in this case, at least one of the comparison
  // operands has at least one user besides the compare (the select),
  // which would often largely negate the benefit of folding anyway.
  if (I.hasOneUse())
    if (SelectInst *SI = dyn_cast<SelectInst>(*I.use_begin()))
      if ((SI->getOperand(1) == Op0 && SI->getOperand(2) == Op1) ||
          (SI->getOperand(2) == Op0 && SI->getOperand(1) == Op1))
        return 0;

  // See if we are doing a comparison between a constant and an instruction that
  // can be folded into the comparison.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {
    // Since the RHS is a ConstantInt (CI), if the left hand side is an
    // instruction, see if that instruction also has constants so that the
    // instruction can be folded into the icmp
    if (Instruction *LHSI = dyn_cast<Instruction>(Op0))
      if (Instruction *Res = visitICmpInstWithInstAndIntCst(I, LHSI, CI))
        return Res;
  }

  // Handle icmp with constant (but not simple integer constant) RHS
  if (Constant *RHSC = dyn_cast<Constant>(Op1)) {
    if (Instruction *LHSI = dyn_cast<Instruction>(Op0))
      switch (LHSI->getOpcode()) {
      case Instruction::GetElementPtr:
          // icmp pred GEP (P, int 0, int 0, int 0), null -> icmp pred P, null
        if (RHSC->isNullValue() &&
            cast<GetElementPtrInst>(LHSI)->hasAllZeroIndices())
          return new ICmpInst(I.getPredicate(), LHSI->getOperand(0),
                  Constant::getNullValue(LHSI->getOperand(0)->getType()));
        break;
      case Instruction::PHI:
        // Only fold icmp into the PHI if the phi and icmp are in the same
        // block.  If in the same block, we're encouraging jump threading.  If
        // not, we are just pessimizing the code by making an i1 phi.
        if (LHSI->getParent() == I.getParent())
          if (Instruction *NV = FoldOpIntoPhi(I))
            return NV;
        break;
      case Instruction::Select: {
        // If either operand of the select is a constant, we can fold the
        // comparison into the select arms, which will cause one to be
        // constant folded and the select turned into a bitwise or.
        Value *Op1 = 0, *Op2 = 0;
        if (Constant *C = dyn_cast<Constant>(LHSI->getOperand(1)))
          Op1 = ConstantExpr::getICmp(I.getPredicate(), C, RHSC);
        if (Constant *C = dyn_cast<Constant>(LHSI->getOperand(2)))
          Op2 = ConstantExpr::getICmp(I.getPredicate(), C, RHSC);

        // We only want to perform this transformation if it will not lead to
        // additional code. This is true if either both sides of the select
        // fold to a constant (in which case the icmp is replaced with a select
        // which will usually simplify) or this is the only user of the
        // select (in which case we are trading a select+icmp for a simpler
        // select+icmp).
        if ((Op1 && Op2) || (LHSI->hasOneUse() && (Op1 || Op2))) {
          if (!Op1)
            Op1 = Builder->CreateICmp(I.getPredicate(), LHSI->getOperand(1),
                                      RHSC, I.getName());
          if (!Op2)
            Op2 = Builder->CreateICmp(I.getPredicate(), LHSI->getOperand(2),
                                      RHSC, I.getName());
          return SelectInst::Create(LHSI->getOperand(0), Op1, Op2);
        }
        break;
      }
      case Instruction::IntToPtr:
        // icmp pred inttoptr(X), null -> icmp pred X, 0
        if (RHSC->isNullValue() && TD &&
            TD->getIntPtrType(RHSC->getContext()) ==
               LHSI->getOperand(0)->getType())
          return new ICmpInst(I.getPredicate(), LHSI->getOperand(0),
                        Constant::getNullValue(LHSI->getOperand(0)->getType()));
        break;

      case Instruction::Load:
        // Try to optimize things like "A[i] > 4" to index computations.
        if (GetElementPtrInst *GEP =
              dyn_cast<GetElementPtrInst>(LHSI->getOperand(0))) {
          if (GlobalVariable *GV = dyn_cast<GlobalVariable>(GEP->getOperand(0)))
            if (GV->isConstant() && GV->hasDefinitiveInitializer() &&
                !cast<LoadInst>(LHSI)->isVolatile())
              if (Instruction *Res = FoldCmpLoadFromIndexedGlobal(GEP, GV, I))
                return Res;
        }
        break;
      }
  }

  // If we can optimize a 'icmp GEP, P' or 'icmp P, GEP', do so now.
  if (GEPOperator *GEP = dyn_cast<GEPOperator>(Op0))
    if (Instruction *NI = FoldGEPICmp(GEP, Op1, I.getPredicate(), I))
      return NI;
  if (GEPOperator *GEP = dyn_cast<GEPOperator>(Op1))
    if (Instruction *NI = FoldGEPICmp(GEP, Op0,
                           ICmpInst::getSwappedPredicate(I.getPredicate()), I))
      return NI;

  // Test to see if the operands of the icmp are casted versions of other
  // values.  If the ptr->ptr cast can be stripped off both arguments, we do so
  // now.
  if (BitCastInst *CI = dyn_cast<BitCastInst>(Op0)) {
    if (Op0->getType()->isPointerTy() &&
        (isa<Constant>(Op1) || isa<BitCastInst>(Op1))) {
      // We keep moving the cast from the left operand over to the right
      // operand, where it can often be eliminated completely.
      Op0 = CI->getOperand(0);

      // If operand #1 is a bitcast instruction, it must also be a ptr->ptr cast
      // so eliminate it as well.
      if (BitCastInst *CI2 = dyn_cast<BitCastInst>(Op1))
        Op1 = CI2->getOperand(0);

      // If Op1 is a constant, we can fold the cast into the constant.
      if (Op0->getType() != Op1->getType()) {
        if (Constant *Op1C = dyn_cast<Constant>(Op1)) {
          Op1 = ConstantExpr::getBitCast(Op1C, Op0->getType());
        } else {
          // Otherwise, cast the RHS right before the icmp
          Op1 = Builder->CreateBitCast(Op1, Op0->getType());
        }
      }
      return new ICmpInst(I.getPredicate(), Op0, Op1);
    }
  }

  if (isa<CastInst>(Op0)) {
    // Handle the special case of: icmp (cast bool to X), <cst>
    // This comes up when you have code like
    //   int X = A < B;
    //   if (X) ...
    // For generality, we handle any zero-extension of any operand comparison
    // with a constant or another cast from the same type.
    if (isa<Constant>(Op1) || isa<CastInst>(Op1))
      if (Instruction *R = visitICmpInstWithCastAndCast(I))
        return R;
  }

  // Special logic for binary operators.
  BinaryOperator *BO0 = dyn_cast<BinaryOperator>(Op0);
  BinaryOperator *BO1 = dyn_cast<BinaryOperator>(Op1);
  if (BO0 || BO1) {
    CmpInst::Predicate Pred = I.getPredicate();
    bool NoOp0WrapProblem = false, NoOp1WrapProblem = false;
    if (BO0 && isa<OverflowingBinaryOperator>(BO0))
      NoOp0WrapProblem = ICmpInst::isEquality(Pred) ||
        (CmpInst::isUnsigned(Pred) && BO0->hasNoUnsignedWrap()) ||
        (CmpInst::isSigned(Pred) && BO0->hasNoSignedWrap());
    if (BO1 && isa<OverflowingBinaryOperator>(BO1))
      NoOp1WrapProblem = ICmpInst::isEquality(Pred) ||
        (CmpInst::isUnsigned(Pred) && BO1->hasNoUnsignedWrap()) ||
        (CmpInst::isSigned(Pred) && BO1->hasNoSignedWrap());

    // Analyze the case when either Op0 or Op1 is an add instruction.
    // Op0 = A + B (or A and B are null); Op1 = C + D (or C and D are null).
    Value *A = 0, *B = 0, *C = 0, *D = 0;
    if (BO0 && BO0->getOpcode() == Instruction::Add)
      A = BO0->getOperand(0), B = BO0->getOperand(1);
    if (BO1 && BO1->getOpcode() == Instruction::Add)
      C = BO1->getOperand(0), D = BO1->getOperand(1);

    // icmp (X+Y), X -> icmp Y, 0 for equalities or if there is no overflow.
    if ((A == Op1 || B == Op1) && NoOp0WrapProblem)
      return new ICmpInst(Pred, A == Op1 ? B : A,
                          Constant::getNullValue(Op1->getType()));

    // icmp X, (X+Y) -> icmp 0, Y for equalities or if there is no overflow.
    if ((C == Op0 || D == Op0) && NoOp1WrapProblem)
      return new ICmpInst(Pred, Constant::getNullValue(Op0->getType()),
                          C == Op0 ? D : C);

    // icmp (X+Y), (X+Z) -> icmp Y, Z for equalities or if there is no overflow.
    if (A && C && (A == C || A == D || B == C || B == D) &&
        NoOp0WrapProblem && NoOp1WrapProblem &&
        // Try not to increase register pressure.
        BO0->hasOneUse() && BO1->hasOneUse()) {
      // Determine Y and Z in the form icmp (X+Y), (X+Z).
      Value *Y, *Z;
      if (A == C) {
        Y = B;
        Z = D;
      } else if (A == D) {
        Y = B;
        Z = C;
      } else if (B == C) {
        Y = A;
        Z = D;
      } else if (B == D) {
        Y = A;
        Z = C;
      }
      return new ICmpInst(Pred, Y, Z);
    }

    // Analyze the case when either Op0 or Op1 is a sub instruction.
    // Op0 = A - B (or A and B are null); Op1 = C - D (or C and D are null).
    A = 0; B = 0; C = 0; D = 0;
    if (BO0 && BO0->getOpcode() == Instruction::Sub)
      A = BO0->getOperand(0), B = BO0->getOperand(1);
    if (BO1 && BO1->getOpcode() == Instruction::Sub)
      C = BO1->getOperand(0), D = BO1->getOperand(1);

    // icmp (X-Y), X -> icmp 0, Y for equalities or if there is no overflow.
    if (A == Op1 && NoOp0WrapProblem)
      return new ICmpInst(Pred, Constant::getNullValue(Op1->getType()), B);

    // icmp X, (X-Y) -> icmp Y, 0 for equalities or if there is no overflow.
    if (C == Op0 && NoOp1WrapProblem)
      return new ICmpInst(Pred, D, Constant::getNullValue(Op0->getType()));

    // icmp (Y-X), (Z-X) -> icmp Y, Z for equalities or if there is no overflow.
    if (B && D && B == D && NoOp0WrapProblem && NoOp1WrapProblem &&
        // Try not to increase register pressure.
        BO0->hasOneUse() && BO1->hasOneUse())
      return new ICmpInst(Pred, A, C);

    // icmp (X-Y), (X-Z) -> icmp Z, Y for equalities or if there is no overflow.
    if (A && C && A == C && NoOp0WrapProblem && NoOp1WrapProblem &&
        // Try not to increase register pressure.
        BO0->hasOneUse() && BO1->hasOneUse())
      return new ICmpInst(Pred, D, B);

    BinaryOperator *SRem = NULL;
    // icmp (srem X, Y), Y
    if (BO0 && BO0->getOpcode() == Instruction::SRem &&
        Op1 == BO0->getOperand(1))
      SRem = BO0;
    // icmp Y, (srem X, Y)
    else if (BO1 && BO1->getOpcode() == Instruction::SRem &&
             Op0 == BO1->getOperand(1))
      SRem = BO1;
    if (SRem) {
      // We don't check hasOneUse to avoid increasing register pressure because
      // the value we use is the same value this instruction was already using.
      switch (SRem == BO0 ? ICmpInst::getSwappedPredicate(Pred) : Pred) {
        default: break;
        case ICmpInst::ICMP_EQ:
          return ReplaceInstUsesWith(I, ConstantInt::getFalse(I.getType()));
        case ICmpInst::ICMP_NE:
          return ReplaceInstUsesWith(I, ConstantInt::getTrue(I.getType()));
        case ICmpInst::ICMP_SGT:
        case ICmpInst::ICMP_SGE:
          return new ICmpInst(ICmpInst::ICMP_SGT, SRem->getOperand(1),
                              Constant::getAllOnesValue(SRem->getType()));
        case ICmpInst::ICMP_SLT:
        case ICmpInst::ICMP_SLE:
          return new ICmpInst(ICmpInst::ICMP_SLT, SRem->getOperand(1),
                              Constant::getNullValue(SRem->getType()));
      }
    }

    if (BO0 && BO1 && BO0->getOpcode() == BO1->getOpcode() &&
        BO0->hasOneUse() && BO1->hasOneUse() &&
        BO0->getOperand(1) == BO1->getOperand(1)) {
      switch (BO0->getOpcode()) {
      default: break;
      case Instruction::Add:
      case Instruction::Sub:
      case Instruction::Xor:
        if (I.isEquality())    // a+x icmp eq/ne b+x --> a icmp b
          return new ICmpInst(I.getPredicate(), BO0->getOperand(0),
                              BO1->getOperand(0));
        // icmp u/s (a ^ signbit), (b ^ signbit) --> icmp s/u a, b
        if (ConstantInt *CI = dyn_cast<ConstantInt>(BO0->getOperand(1))) {
          if (CI->getValue().isSignBit()) {
            ICmpInst::Predicate Pred = I.isSigned()
                                           ? I.getUnsignedPredicate()
                                           : I.getSignedPredicate();
            return new ICmpInst(Pred, BO0->getOperand(0),
                                BO1->getOperand(0));
          }

          if (CI->isMaxValue(true)) {
            ICmpInst::Predicate Pred = I.isSigned()
                                           ? I.getUnsignedPredicate()
                                           : I.getSignedPredicate();
            Pred = I.getSwappedPredicate(Pred);
            return new ICmpInst(Pred, BO0->getOperand(0),
                                BO1->getOperand(0));
          }
        }
        break;
      case Instruction::Mul:
        if (!I.isEquality())
          break;

        if (ConstantInt *CI = dyn_cast<ConstantInt>(BO0->getOperand(1))) {
          // a * Cst icmp eq/ne b * Cst --> a & Mask icmp b & Mask
          // Mask = -1 >> count-trailing-zeros(Cst).
          if (!CI->isZero() && !CI->isOne()) {
            const APInt &AP = CI->getValue();
            ConstantInt *Mask = ConstantInt::get(I.getContext(),
                                    APInt::getLowBitsSet(AP.getBitWidth(),
                                                         AP.getBitWidth() -
                                                    AP.countTrailingZeros()));
            Value *And1 = Builder->CreateAnd(BO0->getOperand(0), Mask);
            Value *And2 = Builder->CreateAnd(BO1->getOperand(0), Mask);
            return new ICmpInst(I.getPredicate(), And1, And2);
          }
        }
        break;
      case Instruction::UDiv:
      case Instruction::LShr:
        if (I.isSigned())
          break;
        // fall-through
      case Instruction::SDiv:
      case Instruction::AShr:
        if (!BO0->isExact() || !BO1->isExact())
          break;
        return new ICmpInst(I.getPredicate(), BO0->getOperand(0),
                            BO1->getOperand(0));
      case Instruction::Shl: {
        bool NUW = BO0->hasNoUnsignedWrap() && BO1->hasNoUnsignedWrap();
        bool NSW = BO0->hasNoSignedWrap() && BO1->hasNoSignedWrap();
        if (!NUW && !NSW)
          break;
        if (!NSW && I.isSigned())
          break;
        return new ICmpInst(I.getPredicate(), BO0->getOperand(0),
                            BO1->getOperand(0));
      }
      }
    }
  }

  { Value *A, *B;
    // ~x < ~y --> y < x
    // ~x < cst --> ~cst < x
    if (match(Op0, m_Not(m_Value(A)))) {
      if (match(Op1, m_Not(m_Value(B))))
        return new ICmpInst(I.getPredicate(), B, A);
      if (ConstantInt *RHSC = dyn_cast<ConstantInt>(Op1))
        return new ICmpInst(I.getPredicate(), ConstantExpr::getNot(RHSC), A);
    }

    // (a+b) <u a  --> llvm.uadd.with.overflow.
    // (a+b) <u b  --> llvm.uadd.with.overflow.
    if (I.getPredicate() == ICmpInst::ICMP_ULT &&
        match(Op0, m_Add(m_Value(A), m_Value(B))) &&
        (Op1 == A || Op1 == B))
      if (Instruction *R = ProcessUAddIdiom(I, Op0, *this))
        return R;

    // a >u (a+b)  --> llvm.uadd.with.overflow.
    // b >u (a+b)  --> llvm.uadd.with.overflow.
    if (I.getPredicate() == ICmpInst::ICMP_UGT &&
        match(Op1, m_Add(m_Value(A), m_Value(B))) &&
        (Op0 == A || Op0 == B))
      if (Instruction *R = ProcessUAddIdiom(I, Op1, *this))
        return R;
  }

  if (I.isEquality()) {
    Value *A, *B, *C, *D;

    if (match(Op0, m_Xor(m_Value(A), m_Value(B)))) {
      if (A == Op1 || B == Op1) {    // (A^B) == A  ->  B == 0
        Value *OtherVal = A == Op1 ? B : A;
        return new ICmpInst(I.getPredicate(), OtherVal,
                            Constant::getNullValue(A->getType()));
      }

      if (match(Op1, m_Xor(m_Value(C), m_Value(D)))) {
        // A^c1 == C^c2 --> A == C^(c1^c2)
        ConstantInt *C1, *C2;
        if (match(B, m_ConstantInt(C1)) &&
            match(D, m_ConstantInt(C2)) && Op1->hasOneUse()) {
          Constant *NC = ConstantInt::get(I.getContext(),
                                          C1->getValue() ^ C2->getValue());
          Value *Xor = Builder->CreateXor(C, NC);
          return new ICmpInst(I.getPredicate(), A, Xor);
        }

        // A^B == A^D -> B == D
        if (A == C) return new ICmpInst(I.getPredicate(), B, D);
        if (A == D) return new ICmpInst(I.getPredicate(), B, C);
        if (B == C) return new ICmpInst(I.getPredicate(), A, D);
        if (B == D) return new ICmpInst(I.getPredicate(), A, C);
      }
    }

    if (match(Op1, m_Xor(m_Value(A), m_Value(B))) &&
        (A == Op0 || B == Op0)) {
      // A == (A^B)  ->  B == 0
      Value *OtherVal = A == Op0 ? B : A;
      return new ICmpInst(I.getPredicate(), OtherVal,
                          Constant::getNullValue(A->getType()));
    }

    // (X&Z) == (Y&Z) -> (X^Y) & Z == 0
    if (match(Op0, m_OneUse(m_And(m_Value(A), m_Value(B)))) &&
        match(Op1, m_OneUse(m_And(m_Value(C), m_Value(D))))) {
      Value *X = 0, *Y = 0, *Z = 0;

      if (A == C) {
        X = B; Y = D; Z = A;
      } else if (A == D) {
        X = B; Y = C; Z = A;
      } else if (B == C) {
        X = A; Y = D; Z = B;
      } else if (B == D) {
        X = A; Y = C; Z = B;
      }

      if (X) {   // Build (X^Y) & Z
        Op1 = Builder->CreateXor(X, Y);
        Op1 = Builder->CreateAnd(Op1, Z);
        I.setOperand(0, Op1);
        I.setOperand(1, Constant::getNullValue(Op1->getType()));
        return &I;
      }
    }

    // Transform (zext A) == (B & (1<<X)-1) --> A == (trunc B)
    // and       (B & (1<<X)-1) == (zext A) --> A == (trunc B)
    ConstantInt *Cst1;
    if ((Op0->hasOneUse() &&
         match(Op0, m_ZExt(m_Value(A))) &&
         match(Op1, m_And(m_Value(B), m_ConstantInt(Cst1)))) ||
        (Op1->hasOneUse() &&
         match(Op0, m_And(m_Value(B), m_ConstantInt(Cst1))) &&
         match(Op1, m_ZExt(m_Value(A))))) {
      APInt Pow2 = Cst1->getValue() + 1;
      if (Pow2.isPowerOf2() && isa<IntegerType>(A->getType()) &&
          Pow2.logBase2() == cast<IntegerType>(A->getType())->getBitWidth())
        return new ICmpInst(I.getPredicate(), A,
                            Builder->CreateTrunc(B, A->getType()));
    }

    // Transform "icmp eq (trunc (lshr(X, cst1)), cst" to
    // "icmp (and X, mask), cst"
    uint64_t ShAmt = 0;
    if (Op0->hasOneUse() &&
        match(Op0, m_Trunc(m_OneUse(m_LShr(m_Value(A),
                                           m_ConstantInt(ShAmt))))) &&
        match(Op1, m_ConstantInt(Cst1)) &&
        // Only do this when A has multiple uses.  This is most important to do
        // when it exposes other optimizations.
        !A->hasOneUse()) {
      unsigned ASize =cast<IntegerType>(A->getType())->getPrimitiveSizeInBits();

      if (ShAmt < ASize) {
        APInt MaskV =
          APInt::getLowBitsSet(ASize, Op0->getType()->getPrimitiveSizeInBits());
        MaskV <<= ShAmt;

        APInt CmpV = Cst1->getValue().zext(ASize);
        CmpV <<= ShAmt;

        Value *Mask = Builder->CreateAnd(A, Builder->getInt(MaskV));
        return new ICmpInst(I.getPredicate(), Mask, Builder->getInt(CmpV));
      }
    }
  }

  {
    Value *X; ConstantInt *Cst;
    // icmp X+Cst, X
    if (match(Op0, m_Add(m_Value(X), m_ConstantInt(Cst))) && Op1 == X)
      return FoldICmpAddOpCst(I, X, Cst, I.getPredicate(), Op0);

    // icmp X, X+Cst
    if (match(Op1, m_Add(m_Value(X), m_ConstantInt(Cst))) && Op0 == X)
      return FoldICmpAddOpCst(I, X, Cst, I.getSwappedPredicate(), Op1);
  }
  return Changed ? &I : 0;
}






/// FoldFCmp_IntToFP_Cst - Fold fcmp ([us]itofp x, cst) if possible.
///
Instruction *InstCombiner::FoldFCmp_IntToFP_Cst(FCmpInst &I,
                                                Instruction *LHSI,
                                                Constant *RHSC) {
  if (!isa<ConstantFP>(RHSC)) return 0;
  const APFloat &RHS = cast<ConstantFP>(RHSC)->getValueAPF();

  // Get the width of the mantissa.  We don't want to hack on conversions that
  // might lose information from the integer, e.g. "i64 -> float"
  int MantissaWidth = LHSI->getType()->getFPMantissaWidth();
  if (MantissaWidth == -1) return 0;  // Unknown.

  // Check to see that the input is converted from an integer type that is small
  // enough that preserves all bits.  TODO: check here for "known" sign bits.
  // This would allow us to handle (fptosi (x >>s 62) to float) if x is i64 f.e.
  unsigned InputSize = LHSI->getOperand(0)->getType()->getScalarSizeInBits();

  // If this is a uitofp instruction, we need an extra bit to hold the sign.
  bool LHSUnsigned = isa<UIToFPInst>(LHSI);
  if (LHSUnsigned)
    ++InputSize;

  // If the conversion would lose info, don't hack on this.
  if ((int)InputSize > MantissaWidth)
    return 0;

  // Otherwise, we can potentially simplify the comparison.  We know that it
  // will always come through as an integer value and we know the constant is
  // not a NAN (it would have been previously simplified).
  assert(!RHS.isNaN() && "NaN comparison not already folded!");

  ICmpInst::Predicate Pred;
  switch (I.getPredicate()) {
  default: llvm_unreachable("Unexpected predicate!");
  case FCmpInst::FCMP_UEQ:
  case FCmpInst::FCMP_OEQ:
    Pred = ICmpInst::ICMP_EQ;
    break;
  case FCmpInst::FCMP_UGT:
  case FCmpInst::FCMP_OGT:
    Pred = LHSUnsigned ? ICmpInst::ICMP_UGT : ICmpInst::ICMP_SGT;
    break;
  case FCmpInst::FCMP_UGE:
  case FCmpInst::FCMP_OGE:
    Pred = LHSUnsigned ? ICmpInst::ICMP_UGE : ICmpInst::ICMP_SGE;
    break;
  case FCmpInst::FCMP_ULT:
  case FCmpInst::FCMP_OLT:
    Pred = LHSUnsigned ? ICmpInst::ICMP_ULT : ICmpInst::ICMP_SLT;
    break;
  case FCmpInst::FCMP_ULE:
  case FCmpInst::FCMP_OLE:
    Pred = LHSUnsigned ? ICmpInst::ICMP_ULE : ICmpInst::ICMP_SLE;
    break;
  case FCmpInst::FCMP_UNE:
  case FCmpInst::FCMP_ONE:
    Pred = ICmpInst::ICMP_NE;
    break;
  case FCmpInst::FCMP_ORD:
    return ReplaceInstUsesWith(I, ConstantInt::getTrue(I.getContext()));
  case FCmpInst::FCMP_UNO:
    return ReplaceInstUsesWith(I, ConstantInt::getFalse(I.getContext()));
  }

  IntegerType *IntTy = cast<IntegerType>(LHSI->getOperand(0)->getType());

  // Now we know that the APFloat is a normal number, zero or inf.

  // See if the FP constant is too large for the integer.  For example,
  // comparing an i8 to 300.0.
  unsigned IntWidth = IntTy->getScalarSizeInBits();

  if (!LHSUnsigned) {
    // If the RHS value is > SignedMax, fold the comparison.  This handles +INF
    // and large values.
    APFloat SMax(RHS.getSemantics(), APFloat::fcZero, false);
    SMax.convertFromAPInt(APInt::getSignedMaxValue(IntWidth), true,
                          APFloat::rmNearestTiesToEven);
    if (SMax.compare(RHS) == APFloat::cmpLessThan) {  // smax < 13123.0
      if (Pred == ICmpInst::ICMP_NE  || Pred == ICmpInst::ICMP_SLT ||
          Pred == ICmpInst::ICMP_SLE)
        return ReplaceInstUsesWith(I, ConstantInt::getTrue(I.getContext()));
      return ReplaceInstUsesWith(I, ConstantInt::getFalse(I.getContext()));
    }
  } else {
    // If the RHS value is > UnsignedMax, fold the comparison. This handles
    // +INF and large values.
    APFloat UMax(RHS.getSemantics(), APFloat::fcZero, false);
    UMax.convertFromAPInt(APInt::getMaxValue(IntWidth), false,
                          APFloat::rmNearestTiesToEven);
    if (UMax.compare(RHS) == APFloat::cmpLessThan) {  // umax < 13123.0
      if (Pred == ICmpInst::ICMP_NE  || Pred == ICmpInst::ICMP_ULT ||
          Pred == ICmpInst::ICMP_ULE)
        return ReplaceInstUsesWith(I, ConstantInt::getTrue(I.getContext()));
      return ReplaceInstUsesWith(I, ConstantInt::getFalse(I.getContext()));
    }
  }

  if (!LHSUnsigned) {
    // See if the RHS value is < SignedMin.
    APFloat SMin(RHS.getSemantics(), APFloat::fcZero, false);
    SMin.convertFromAPInt(APInt::getSignedMinValue(IntWidth), true,
                          APFloat::rmNearestTiesToEven);
    if (SMin.compare(RHS) == APFloat::cmpGreaterThan) { // smin > 12312.0
      if (Pred == ICmpInst::ICMP_NE || Pred == ICmpInst::ICMP_SGT ||
          Pred == ICmpInst::ICMP_SGE)
        return ReplaceInstUsesWith(I, ConstantInt::getTrue(I.getContext()));
      return ReplaceInstUsesWith(I, ConstantInt::getFalse(I.getContext()));
    }
  } else {
    // See if the RHS value is < UnsignedMin.
    APFloat SMin(RHS.getSemantics(), APFloat::fcZero, false);
    SMin.convertFromAPInt(APInt::getMinValue(IntWidth), true,
                          APFloat::rmNearestTiesToEven);
    if (SMin.compare(RHS) == APFloat::cmpGreaterThan) { // umin > 12312.0
      if (Pred == ICmpInst::ICMP_NE || Pred == ICmpInst::ICMP_UGT ||
          Pred == ICmpInst::ICMP_UGE)
        return ReplaceInstUsesWith(I, ConstantInt::getTrue(I.getContext()));
      return ReplaceInstUsesWith(I, ConstantInt::getFalse(I.getContext()));
    }
  }

  // Okay, now we know that the FP constant fits in the range [SMIN, SMAX] or
  // [0, UMAX], but it may still be fractional.  See if it is fractional by
  // casting the FP value to the integer value and back, checking for equality.
  // Don't do this for zero, because -0.0 is not fractional.
  Constant *RHSInt = LHSUnsigned
    ? ConstantExpr::getFPToUI(RHSC, IntTy)
    : ConstantExpr::getFPToSI(RHSC, IntTy);
  if (!RHS.isZero()) {
    bool Equal = LHSUnsigned
      ? ConstantExpr::getUIToFP(RHSInt, RHSC->getType()) == RHSC
      : ConstantExpr::getSIToFP(RHSInt, RHSC->getType()) == RHSC;
    if (!Equal) {
      // If we had a comparison against a fractional value, we have to adjust
      // the compare predicate and sometimes the value.  RHSC is rounded towards
      // zero at this point.
      switch (Pred) {
      default: llvm_unreachable("Unexpected integer comparison!");
      case ICmpInst::ICMP_NE:  // (float)int != 4.4   --> true
        return ReplaceInstUsesWith(I, ConstantInt::getTrue(I.getContext()));
      case ICmpInst::ICMP_EQ:  // (float)int == 4.4   --> false
        return ReplaceInstUsesWith(I, ConstantInt::getFalse(I.getContext()));
      case ICmpInst::ICMP_ULE:
        // (float)int <= 4.4   --> int <= 4
        // (float)int <= -4.4  --> false
        if (RHS.isNegative())
          return ReplaceInstUsesWith(I, ConstantInt::getFalse(I.getContext()));
        break;
      case ICmpInst::ICMP_SLE:
        // (float)int <= 4.4   --> int <= 4
        // (float)int <= -4.4  --> int < -4
        if (RHS.isNegative())
          Pred = ICmpInst::ICMP_SLT;
        break;
      case ICmpInst::ICMP_ULT:
        // (float)int < -4.4   --> false
        // (float)int < 4.4    --> int <= 4
        if (RHS.isNegative())
          return ReplaceInstUsesWith(I, ConstantInt::getFalse(I.getContext()));
        Pred = ICmpInst::ICMP_ULE;
        break;
      case ICmpInst::ICMP_SLT:
        // (float)int < -4.4   --> int < -4
        // (float)int < 4.4    --> int <= 4
        if (!RHS.isNegative())
          Pred = ICmpInst::ICMP_SLE;
        break;
      case ICmpInst::ICMP_UGT:
        // (float)int > 4.4    --> int > 4
        // (float)int > -4.4   --> true
        if (RHS.isNegative())
          return ReplaceInstUsesWith(I, ConstantInt::getTrue(I.getContext()));
        break;
      case ICmpInst::ICMP_SGT:
        // (float)int > 4.4    --> int > 4
        // (float)int > -4.4   --> int >= -4
        if (RHS.isNegative())
          Pred = ICmpInst::ICMP_SGE;
        break;
      case ICmpInst::ICMP_UGE:
        // (float)int >= -4.4   --> true
        // (float)int >= 4.4    --> int > 4
        if (RHS.isNegative())
          return ReplaceInstUsesWith(I, ConstantInt::getTrue(I.getContext()));
        Pred = ICmpInst::ICMP_UGT;
        break;
      case ICmpInst::ICMP_SGE:
        // (float)int >= -4.4   --> int >= -4
        // (float)int >= 4.4    --> int > 4
        if (!RHS.isNegative())
          Pred = ICmpInst::ICMP_SGT;
        break;
      }
    }
  }

  // Lower this FP comparison into an appropriate integer version of the
  // comparison.
  return new ICmpInst(Pred, LHSI->getOperand(0), RHSInt);
}

Instruction *InstCombiner::visitFCmpInst(FCmpInst &I) {
  bool Changed = false;

  /// Orders the operands of the compare so that they are listed from most
  /// complex to least complex.  This puts constants before unary operators,
  /// before binary operators.
  if (getComplexity(I.getOperand(0)) < getComplexity(I.getOperand(1))) {
    I.swapOperands();
    Changed = true;
  }

  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyFCmpInst(I.getPredicate(), Op0, Op1, TD))
    return ReplaceInstUsesWith(I, V);

  // Simplify 'fcmp pred X, X'
  if (Op0 == Op1) {
    switch (I.getPredicate()) {
    default: llvm_unreachable("Unknown predicate!");
    case FCmpInst::FCMP_UNO:    // True if unordered: isnan(X) | isnan(Y)
    case FCmpInst::FCMP_ULT:    // True if unordered or less than
    case FCmpInst::FCMP_UGT:    // True if unordered or greater than
    case FCmpInst::FCMP_UNE:    // True if unordered or not equal
      // Canonicalize these to be 'fcmp uno %X, 0.0'.
      I.setPredicate(FCmpInst::FCMP_UNO);
      I.setOperand(1, Constant::getNullValue(Op0->getType()));
      return &I;

    case FCmpInst::FCMP_ORD:    // True if ordered (no nans)
    case FCmpInst::FCMP_OEQ:    // True if ordered and equal
    case FCmpInst::FCMP_OGE:    // True if ordered and greater than or equal
    case FCmpInst::FCMP_OLE:    // True if ordered and less than or equal
      // Canonicalize these to be 'fcmp ord %X, 0.0'.
      I.setPredicate(FCmpInst::FCMP_ORD);
      I.setOperand(1, Constant::getNullValue(Op0->getType()));
      return &I;
    }
  }

  // Handle fcmp with constant RHS
  if (Constant *RHSC = dyn_cast<Constant>(Op1)) {
    if (Instruction *LHSI = dyn_cast<Instruction>(Op0))
      switch (LHSI->getOpcode()) {
      case Instruction::FPExt: {
        // fcmp (fpext x), C -> fcmp x, (fptrunc C) if fptrunc is lossless
        FPExtInst *LHSExt = cast<FPExtInst>(LHSI);
        ConstantFP *RHSF = dyn_cast<ConstantFP>(RHSC);
        if (!RHSF)
          break;

        const fltSemantics *Sem;
        // FIXME: This shouldn't be here.
        if (LHSExt->getSrcTy()->isHalfTy())
          Sem = &APFloat::IEEEhalf;
        else if (LHSExt->getSrcTy()->isFloatTy())
          Sem = &APFloat::IEEEsingle;
        else if (LHSExt->getSrcTy()->isDoubleTy())
          Sem = &APFloat::IEEEdouble;
        else if (LHSExt->getSrcTy()->isFP128Ty())
          Sem = &APFloat::IEEEquad;
        else if (LHSExt->getSrcTy()->isX86_FP80Ty())
          Sem = &APFloat::x87DoubleExtended;
        else if (LHSExt->getSrcTy()->isPPC_FP128Ty())
          Sem = &APFloat::PPCDoubleDouble;
        else
          break;

        bool Lossy;
        APFloat F = RHSF->getValueAPF();
        F.convert(*Sem, APFloat::rmNearestTiesToEven, &Lossy);

        // Avoid lossy conversions and denormals. Zero is a special case
        // that's OK to convert.
        APFloat Fabs = F;
        Fabs.clearSign();
        if (!Lossy &&
            ((Fabs.compare(APFloat::getSmallestNormalized(*Sem)) !=
                 APFloat::cmpLessThan) || Fabs.isZero()))

          return new FCmpInst(I.getPredicate(), LHSExt->getOperand(0),
                              ConstantFP::get(RHSC->getContext(), F));
        break;
      }
      case Instruction::PHI:
        // Only fold fcmp into the PHI if the phi and fcmp are in the same
        // block.  If in the same block, we're encouraging jump threading.  If
        // not, we are just pessimizing the code by making an i1 phi.
        if (LHSI->getParent() == I.getParent())
          if (Instruction *NV = FoldOpIntoPhi(I))
            return NV;
        break;
      case Instruction::SIToFP:
      case Instruction::UIToFP:
        if (Instruction *NV = FoldFCmp_IntToFP_Cst(I, LHSI, RHSC))
          return NV;
        break;
      case Instruction::Select: {
        // If either operand of the select is a constant, we can fold the
        // comparison into the select arms, which will cause one to be
        // constant folded and the select turned into a bitwise or.
        Value *Op1 = 0, *Op2 = 0;
        if (LHSI->hasOneUse()) {
          if (Constant *C = dyn_cast<Constant>(LHSI->getOperand(1))) {
            // Fold the known value into the constant operand.
            Op1 = ConstantExpr::getCompare(I.getPredicate(), C, RHSC);
            // Insert a new FCmp of the other select operand.
            Op2 = Builder->CreateFCmp(I.getPredicate(),
                                      LHSI->getOperand(2), RHSC, I.getName());
          } else if (Constant *C = dyn_cast<Constant>(LHSI->getOperand(2))) {
            // Fold the known value into the constant operand.
            Op2 = ConstantExpr::getCompare(I.getPredicate(), C, RHSC);
            // Insert a new FCmp of the other select operand.
            Op1 = Builder->CreateFCmp(I.getPredicate(), LHSI->getOperand(1),
                                      RHSC, I.getName());
          }
        }

        if (Op1)
          return SelectInst::Create(LHSI->getOperand(0), Op1, Op2);
        break;
      }
      case Instruction::FSub: {
        // fcmp pred (fneg x), C -> fcmp swap(pred) x, -C
        Value *Op;
        if (match(LHSI, m_FNeg(m_Value(Op))))
          return new FCmpInst(I.getSwappedPredicate(), Op,
                              ConstantExpr::getFNeg(RHSC));
        break;
      }
      case Instruction::Load:
        if (GetElementPtrInst *GEP =
            dyn_cast<GetElementPtrInst>(LHSI->getOperand(0))) {
          if (GlobalVariable *GV = dyn_cast<GlobalVariable>(GEP->getOperand(0)))
            if (GV->isConstant() && GV->hasDefinitiveInitializer() &&
                !cast<LoadInst>(LHSI)->isVolatile())
              if (Instruction *Res = FoldCmpLoadFromIndexedGlobal(GEP, GV, I))
                return Res;
        }
        break;
      case Instruction::Call: {
        CallInst *CI = cast<CallInst>(LHSI);
        LibFunc::Func Func;
        // Various optimization for fabs compared with zero.
        if (RHSC->isNullValue() && CI->getCalledFunction() &&
            TLI->getLibFunc(CI->getCalledFunction()->getName(), Func) &&
            TLI->has(Func)) {
          if (Func == LibFunc::fabs || Func == LibFunc::fabsf ||
              Func == LibFunc::fabsl) {
            switch (I.getPredicate()) {
            default: break;
            // fabs(x) < 0 --> false
            case FCmpInst::FCMP_OLT:
              return ReplaceInstUsesWith(I, Builder->getFalse());
            // fabs(x) > 0 --> x != 0
            case FCmpInst::FCMP_OGT:
              return new FCmpInst(FCmpInst::FCMP_ONE, CI->getArgOperand(0),
                                  RHSC);
            // fabs(x) <= 0 --> x == 0
            case FCmpInst::FCMP_OLE:
              return new FCmpInst(FCmpInst::FCMP_OEQ, CI->getArgOperand(0),
                                  RHSC);
            // fabs(x) >= 0 --> !isnan(x)
            case FCmpInst::FCMP_OGE:
              return new FCmpInst(FCmpInst::FCMP_ORD, CI->getArgOperand(0),
                                  RHSC);
            // fabs(x) == 0 --> x == 0
            // fabs(x) != 0 --> x != 0
            case FCmpInst::FCMP_OEQ:
            case FCmpInst::FCMP_UEQ:
            case FCmpInst::FCMP_ONE:
            case FCmpInst::FCMP_UNE:
              return new FCmpInst(I.getPredicate(), CI->getArgOperand(0),
                                  RHSC);
            }
          }
        }
      }
      }
  }

  // fcmp pred (fneg x), (fneg y) -> fcmp swap(pred) x, y
  Value *X, *Y;
  if (match(Op0, m_FNeg(m_Value(X))) && match(Op1, m_FNeg(m_Value(Y))))
    return new FCmpInst(I.getSwappedPredicate(), X, Y);

  // fcmp (fpext x), (fpext y) -> fcmp x, y
  if (FPExtInst *LHSExt = dyn_cast<FPExtInst>(Op0))
    if (FPExtInst *RHSExt = dyn_cast<FPExtInst>(Op1))
      if (LHSExt->getSrcTy() == RHSExt->getSrcTy())
        return new FCmpInst(I.getPredicate(), LHSExt->getOperand(0),
                            RHSExt->getOperand(0));

  return Changed ? &I : 0;
}

//===- InstCombineVectorOps.cpp -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements instcombine for ExtractElement, InsertElement and
// ShuffleVector.
//
//===----------------------------------------------------------------------===//

#include "InstCombine.h"
using namespace llvm;

/// CheapToScalarize - Return true if the value is cheaper to scalarize than it
/// is to leave as a vector operation.  isConstant indicates whether we're
/// extracting one known element.  If false we're extracting a variable index.
static bool CheapToScalarize(Value *V, bool isConstant) {
  if (Constant *C = dyn_cast<Constant>(V)) {
    if (isConstant) return true;

    // If all elts are the same, we can extract it and use any of the values.
    Constant *Op0 = C->getAggregateElement(0U);
    for (unsigned i = 1, e = V->getType()->getVectorNumElements(); i != e; ++i)
      if (C->getAggregateElement(i) != Op0)
        return false;
    return true;
  }
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) return false;

  // Insert element gets simplified to the inserted element or is deleted if
  // this is constant idx extract element and its a constant idx insertelt.
  if (I->getOpcode() == Instruction::InsertElement && isConstant &&
      isa<ConstantInt>(I->getOperand(2)))
    return true;
  if (I->getOpcode() == Instruction::Load && I->hasOneUse())
    return true;
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(I))
    if (BO->hasOneUse() &&
        (CheapToScalarize(BO->getOperand(0), isConstant) ||
         CheapToScalarize(BO->getOperand(1), isConstant)))
      return true;
  if (CmpInst *CI = dyn_cast<CmpInst>(I))
    if (CI->hasOneUse() &&
        (CheapToScalarize(CI->getOperand(0), isConstant) ||
         CheapToScalarize(CI->getOperand(1), isConstant)))
      return true;

  return false;
}

/// FindScalarElement - Given a vector and an element number, see if the scalar
/// value is already around as a register, for example if it were inserted then
/// extracted from the vector.
static Value *FindScalarElement(Value *V, unsigned EltNo) {
  assert(V->getType()->isVectorTy() && "Not looking at a vector?");
  VectorType *VTy = cast<VectorType>(V->getType());
  unsigned Width = VTy->getNumElements();
  if (EltNo >= Width)  // Out of range access.
    return UndefValue::get(VTy->getElementType());

  if (Constant *C = dyn_cast<Constant>(V))
    return C->getAggregateElement(EltNo);

  if (InsertElementInst *III = dyn_cast<InsertElementInst>(V)) {
    // If this is an insert to a variable element, we don't know what it is.
    if (!isa<ConstantInt>(III->getOperand(2)))
      return 0;
    unsigned IIElt = cast<ConstantInt>(III->getOperand(2))->getZExtValue();

    // If this is an insert to the element we are looking for, return the
    // inserted value.
    if (EltNo == IIElt)
      return III->getOperand(1);

    // Otherwise, the insertelement doesn't modify the value, recurse on its
    // vector input.
    return FindScalarElement(III->getOperand(0), EltNo);
  }

  if (ShuffleVectorInst *SVI = dyn_cast<ShuffleVectorInst>(V)) {
    unsigned LHSWidth = SVI->getOperand(0)->getType()->getVectorNumElements();
    int InEl = SVI->getMaskValue(EltNo);
    if (InEl < 0)
      return UndefValue::get(VTy->getElementType());
    if (InEl < (int)LHSWidth)
      return FindScalarElement(SVI->getOperand(0), InEl);
    return FindScalarElement(SVI->getOperand(1), InEl - LHSWidth);
  }

  // Otherwise, we don't know.
  return 0;
}

Instruction *InstCombiner::visitExtractElementInst(ExtractElementInst &EI) {
  // If vector val is constant with all elements the same, replace EI with
  // that element.  We handle a known element # below.
  if (Constant *C = dyn_cast<Constant>(EI.getOperand(0)))
    if (CheapToScalarize(C, false))
      return ReplaceInstUsesWith(EI, C->getAggregateElement(0U));

  // If extracting a specified index from the vector, see if we can recursively
  // find a previously computed scalar that was inserted into the vector.
  if (ConstantInt *IdxC = dyn_cast<ConstantInt>(EI.getOperand(1))) {
    unsigned IndexVal = IdxC->getZExtValue();
    unsigned VectorWidth = EI.getVectorOperandType()->getNumElements();

    // If this is extracting an invalid index, turn this into undef, to avoid
    // crashing the code below.
    if (IndexVal >= VectorWidth)
      return ReplaceInstUsesWith(EI, UndefValue::get(EI.getType()));

    // This instruction only demands the single element from the input vector.
    // If the input vector has a single use, simplify it based on this use
    // property.
    if (EI.getOperand(0)->hasOneUse() && VectorWidth != 1) {
      APInt UndefElts(VectorWidth, 0);
      APInt DemandedMask(VectorWidth, 0);
      DemandedMask.setBit(IndexVal);
      if (Value *V = SimplifyDemandedVectorElts(EI.getOperand(0),
                                                DemandedMask, UndefElts)) {
        EI.setOperand(0, V);
        return &EI;
      }
    }

    if (Value *Elt = FindScalarElement(EI.getOperand(0), IndexVal))
      return ReplaceInstUsesWith(EI, Elt);

    // If the this extractelement is directly using a bitcast from a vector of
    // the same number of elements, see if we can find the source element from
    // it.  In this case, we will end up needing to bitcast the scalars.
    if (BitCastInst *BCI = dyn_cast<BitCastInst>(EI.getOperand(0))) {
      if (VectorType *VT = dyn_cast<VectorType>(BCI->getOperand(0)->getType()))
        if (VT->getNumElements() == VectorWidth)
          if (Value *Elt = FindScalarElement(BCI->getOperand(0), IndexVal))
            return new BitCastInst(Elt, EI.getType());
    }
  }

  if (Instruction *I = dyn_cast<Instruction>(EI.getOperand(0))) {
    // Push extractelement into predecessor operation if legal and
    // profitable to do so
    if (BinaryOperator *BO = dyn_cast<BinaryOperator>(I)) {
      if (I->hasOneUse() &&
          CheapToScalarize(BO, isa<ConstantInt>(EI.getOperand(1)))) {
        Value *newEI0 =
          Builder->CreateExtractElement(BO->getOperand(0), EI.getOperand(1),
                                        EI.getName()+".lhs");
        Value *newEI1 =
          Builder->CreateExtractElement(BO->getOperand(1), EI.getOperand(1),
                                        EI.getName()+".rhs");
        return BinaryOperator::Create(BO->getOpcode(), newEI0, newEI1);
      }
    } else if (InsertElementInst *IE = dyn_cast<InsertElementInst>(I)) {
      // Extracting the inserted element?
      if (IE->getOperand(2) == EI.getOperand(1))
        return ReplaceInstUsesWith(EI, IE->getOperand(1));
      // If the inserted and extracted elements are constants, they must not
      // be the same value, extract from the pre-inserted value instead.
      if (isa<Constant>(IE->getOperand(2)) && isa<Constant>(EI.getOperand(1))) {
        Worklist.AddValue(EI.getOperand(0));
        EI.setOperand(0, IE->getOperand(0));
        return &EI;
      }
    } else if (ShuffleVectorInst *SVI = dyn_cast<ShuffleVectorInst>(I)) {
      // If this is extracting an element from a shufflevector, figure out where
      // it came from and extract from the appropriate input element instead.
      if (ConstantInt *Elt = dyn_cast<ConstantInt>(EI.getOperand(1))) {
        int SrcIdx = SVI->getMaskValue(Elt->getZExtValue());
        Value *Src;
        unsigned LHSWidth =
          SVI->getOperand(0)->getType()->getVectorNumElements();

        if (SrcIdx < 0)
          return ReplaceInstUsesWith(EI, UndefValue::get(EI.getType()));
        if (SrcIdx < (int)LHSWidth)
          Src = SVI->getOperand(0);
        else {
          SrcIdx -= LHSWidth;
          Src = SVI->getOperand(1);
        }
        Type *Int32Ty = Type::getInt32Ty(EI.getContext());
        return ExtractElementInst::Create(Src,
                                          ConstantInt::get(Int32Ty,
                                                           SrcIdx, false));
      }
    } else if (CastInst *CI = dyn_cast<CastInst>(I)) {
      // Canonicalize extractelement(cast) -> cast(extractelement)
      // bitcasts can change the number of vector elements and they cost nothing
      if (CI->hasOneUse() && EI.hasOneUse() &&
          (CI->getOpcode() != Instruction::BitCast)) {
        Value *EE = Builder->CreateExtractElement(CI->getOperand(0),
                                                  EI.getIndexOperand());
        return CastInst::Create(CI->getOpcode(), EE, EI.getType());
      }
    }
  }
  return 0;
}

/// CollectSingleShuffleElements - If V is a shuffle of values that ONLY returns
/// elements from either LHS or RHS, return the shuffle mask and true.
/// Otherwise, return false.
static bool CollectSingleShuffleElements(Value *V, Value *LHS, Value *RHS,
                                         SmallVectorImpl<Constant*> &Mask) {
  assert(V->getType() == LHS->getType() && V->getType() == RHS->getType() &&
         "Invalid CollectSingleShuffleElements");
  unsigned NumElts = cast<VectorType>(V->getType())->getNumElements();

  if (isa<UndefValue>(V)) {
    Mask.assign(NumElts, UndefValue::get(Type::getInt32Ty(V->getContext())));
    return true;
  }

  if (V == LHS) {
    for (unsigned i = 0; i != NumElts; ++i)
      Mask.push_back(ConstantInt::get(Type::getInt32Ty(V->getContext()), i));
    return true;
  }

  if (V == RHS) {
    for (unsigned i = 0; i != NumElts; ++i)
      Mask.push_back(ConstantInt::get(Type::getInt32Ty(V->getContext()),
                                      i+NumElts));
    return true;
  }

  if (InsertElementInst *IEI = dyn_cast<InsertElementInst>(V)) {
    // If this is an insert of an extract from some other vector, include it.
    Value *VecOp    = IEI->getOperand(0);
    Value *ScalarOp = IEI->getOperand(1);
    Value *IdxOp    = IEI->getOperand(2);

    if (!isa<ConstantInt>(IdxOp))
      return false;
    unsigned InsertedIdx = cast<ConstantInt>(IdxOp)->getZExtValue();

    if (isa<UndefValue>(ScalarOp)) {  // inserting undef into vector.
      // Okay, we can handle this if the vector we are insertinting into is
      // transitively ok.
      if (CollectSingleShuffleElements(VecOp, LHS, RHS, Mask)) {
        // If so, update the mask to reflect the inserted undef.
        Mask[InsertedIdx] = UndefValue::get(Type::getInt32Ty(V->getContext()));
        return true;
      }
    } else if (ExtractElementInst *EI = dyn_cast<ExtractElementInst>(ScalarOp)){
      if (isa<ConstantInt>(EI->getOperand(1)) &&
          EI->getOperand(0)->getType() == V->getType()) {
        unsigned ExtractedIdx =
        cast<ConstantInt>(EI->getOperand(1))->getZExtValue();

        // This must be extracting from either LHS or RHS.
        if (EI->getOperand(0) == LHS || EI->getOperand(0) == RHS) {
          // Okay, we can handle this if the vector we are insertinting into is
          // transitively ok.
          if (CollectSingleShuffleElements(VecOp, LHS, RHS, Mask)) {
            // If so, update the mask to reflect the inserted value.
            if (EI->getOperand(0) == LHS) {
              Mask[InsertedIdx % NumElts] =
              ConstantInt::get(Type::getInt32Ty(V->getContext()),
                               ExtractedIdx);
            } else {
              assert(EI->getOperand(0) == RHS);
              Mask[InsertedIdx % NumElts] =
              ConstantInt::get(Type::getInt32Ty(V->getContext()),
                               ExtractedIdx+NumElts);
            }
            return true;
          }
        }
      }
    }
  }
  // TODO: Handle shufflevector here!

  return false;
}

/// CollectShuffleElements - We are building a shuffle of V, using RHS as the
/// RHS of the shuffle instruction, if it is not null.  Return a shuffle mask
/// that computes V and the LHS value of the shuffle.
static Value *CollectShuffleElements(Value *V, SmallVectorImpl<Constant*> &Mask,
                                     Value *&RHS) {
  assert(V->getType()->isVectorTy() &&
         (RHS == 0 || V->getType() == RHS->getType()) &&
         "Invalid shuffle!");
  unsigned NumElts = cast<VectorType>(V->getType())->getNumElements();

  if (isa<UndefValue>(V)) {
    Mask.assign(NumElts, UndefValue::get(Type::getInt32Ty(V->getContext())));
    return V;
  }
  
  if (isa<ConstantAggregateZero>(V)) {
    Mask.assign(NumElts, ConstantInt::get(Type::getInt32Ty(V->getContext()),0));
    return V;
  }
  
  if (InsertElementInst *IEI = dyn_cast<InsertElementInst>(V)) {
    // If this is an insert of an extract from some other vector, include it.
    Value *VecOp    = IEI->getOperand(0);
    Value *ScalarOp = IEI->getOperand(1);
    Value *IdxOp    = IEI->getOperand(2);

    if (ExtractElementInst *EI = dyn_cast<ExtractElementInst>(ScalarOp)) {
      if (isa<ConstantInt>(EI->getOperand(1)) && isa<ConstantInt>(IdxOp) &&
          EI->getOperand(0)->getType() == V->getType()) {
        unsigned ExtractedIdx =
          cast<ConstantInt>(EI->getOperand(1))->getZExtValue();
        unsigned InsertedIdx = cast<ConstantInt>(IdxOp)->getZExtValue();

        // Either the extracted from or inserted into vector must be RHSVec,
        // otherwise we'd end up with a shuffle of three inputs.
        if (EI->getOperand(0) == RHS || RHS == 0) {
          RHS = EI->getOperand(0);
          Value *V = CollectShuffleElements(VecOp, Mask, RHS);
          Mask[InsertedIdx % NumElts] =
            ConstantInt::get(Type::getInt32Ty(V->getContext()),
                             NumElts+ExtractedIdx);
          return V;
        }

        if (VecOp == RHS) {
          Value *V = CollectShuffleElements(EI->getOperand(0), Mask, RHS);
          // Everything but the extracted element is replaced with the RHS.
          for (unsigned i = 0; i != NumElts; ++i) {
            if (i != InsertedIdx)
              Mask[i] = ConstantInt::get(Type::getInt32Ty(V->getContext()),
                                         NumElts+i);
          }
          return V;
        }

        // If this insertelement is a chain that comes from exactly these two
        // vectors, return the vector and the effective shuffle.
        if (CollectSingleShuffleElements(IEI, EI->getOperand(0), RHS, Mask))
          return EI->getOperand(0);
      }
    }
  }
  // TODO: Handle shufflevector here!

  // Otherwise, can't do anything fancy.  Return an identity vector.
  for (unsigned i = 0; i != NumElts; ++i)
    Mask.push_back(ConstantInt::get(Type::getInt32Ty(V->getContext()), i));
  return V;
}

Instruction *InstCombiner::visitInsertElementInst(InsertElementInst &IE) {
  Value *VecOp    = IE.getOperand(0);
  Value *ScalarOp = IE.getOperand(1);
  Value *IdxOp    = IE.getOperand(2);

  // Inserting an undef or into an undefined place, remove this.
  if (isa<UndefValue>(ScalarOp) || isa<UndefValue>(IdxOp))
    ReplaceInstUsesWith(IE, VecOp);

  // If the inserted element was extracted from some other vector, and if the
  // indexes are constant, try to turn this into a shufflevector operation.
  if (ExtractElementInst *EI = dyn_cast<ExtractElementInst>(ScalarOp)) {
    if (isa<ConstantInt>(EI->getOperand(1)) && isa<ConstantInt>(IdxOp) &&
        EI->getOperand(0)->getType() == IE.getType()) {
      unsigned NumVectorElts = IE.getType()->getNumElements();
      unsigned ExtractedIdx =
        cast<ConstantInt>(EI->getOperand(1))->getZExtValue();
      unsigned InsertedIdx = cast<ConstantInt>(IdxOp)->getZExtValue();

      if (ExtractedIdx >= NumVectorElts) // Out of range extract.
        return ReplaceInstUsesWith(IE, VecOp);

      if (InsertedIdx >= NumVectorElts)  // Out of range insert.
        return ReplaceInstUsesWith(IE, UndefValue::get(IE.getType()));

      // If we are extracting a value from a vector, then inserting it right
      // back into the same place, just use the input vector.
      if (EI->getOperand(0) == VecOp && ExtractedIdx == InsertedIdx)
        return ReplaceInstUsesWith(IE, VecOp);

      // If this insertelement isn't used by some other insertelement, turn it
      // (and any insertelements it points to), into one big shuffle.
      if (!IE.hasOneUse() || !isa<InsertElementInst>(IE.use_back())) {
        SmallVector<Constant*, 16> Mask;
        Value *RHS = 0;
        Value *LHS = CollectShuffleElements(&IE, Mask, RHS);
        if (RHS == 0) RHS = UndefValue::get(LHS->getType());
        // We now have a shuffle of LHS, RHS, Mask.
        return new ShuffleVectorInst(LHS, RHS, ConstantVector::get(Mask));
      }
    }
  }

  unsigned VWidth = cast<VectorType>(VecOp->getType())->getNumElements();
  APInt UndefElts(VWidth, 0);
  APInt AllOnesEltMask(APInt::getAllOnesValue(VWidth));
  if (Value *V = SimplifyDemandedVectorElts(&IE, AllOnesEltMask, UndefElts)) {
    if (V != &IE)
      return ReplaceInstUsesWith(IE, V);
    return &IE;
  }

  return 0;
}


Instruction *InstCombiner::visitShuffleVectorInst(ShuffleVectorInst &SVI) {
  Value *LHS = SVI.getOperand(0);
  Value *RHS = SVI.getOperand(1);
  SmallVector<int, 16> Mask = SVI.getShuffleMask();

  bool MadeChange = false;

  // Undefined shuffle mask -> undefined value.
  if (isa<UndefValue>(SVI.getOperand(2)))
    return ReplaceInstUsesWith(SVI, UndefValue::get(SVI.getType()));

  unsigned VWidth = cast<VectorType>(SVI.getType())->getNumElements();

  APInt UndefElts(VWidth, 0);
  APInt AllOnesEltMask(APInt::getAllOnesValue(VWidth));
  if (Value *V = SimplifyDemandedVectorElts(&SVI, AllOnesEltMask, UndefElts)) {
    if (V != &SVI)
      return ReplaceInstUsesWith(SVI, V);
    LHS = SVI.getOperand(0);
    RHS = SVI.getOperand(1);
    MadeChange = true;
  }

  unsigned LHSWidth = cast<VectorType>(LHS->getType())->getNumElements();

  // Canonicalize shuffle(x    ,x,mask) -> shuffle(x, undef,mask')
  // Canonicalize shuffle(undef,x,mask) -> shuffle(x, undef,mask').
  if (LHS == RHS || isa<UndefValue>(LHS)) {
    if (isa<UndefValue>(LHS) && LHS == RHS) {
      // shuffle(undef,undef,mask) -> undef.
      Value* result = (VWidth == LHSWidth)
                      ? LHS : UndefValue::get(SVI.getType());
      return ReplaceInstUsesWith(SVI, result);
    }

    // Remap any references to RHS to use LHS.
    SmallVector<Constant*, 16> Elts;
    for (unsigned i = 0, e = LHSWidth; i != VWidth; ++i) {
      if (Mask[i] < 0) {
        Elts.push_back(UndefValue::get(Type::getInt32Ty(SVI.getContext())));
        continue;
      }

      if ((Mask[i] >= (int)e && isa<UndefValue>(RHS)) ||
          (Mask[i] <  (int)e && isa<UndefValue>(LHS))) {
        Mask[i] = -1;     // Turn into undef.
        Elts.push_back(UndefValue::get(Type::getInt32Ty(SVI.getContext())));
      } else {
        Mask[i] = Mask[i] % e;  // Force to LHS.
        Elts.push_back(ConstantInt::get(Type::getInt32Ty(SVI.getContext()),
                                        Mask[i]));
      }
    }
    SVI.setOperand(0, SVI.getOperand(1));
    SVI.setOperand(1, UndefValue::get(RHS->getType()));
    SVI.setOperand(2, ConstantVector::get(Elts));
    LHS = SVI.getOperand(0);
    RHS = SVI.getOperand(1);
    MadeChange = true;
  }

  if (VWidth == LHSWidth) {
    // Analyze the shuffle, are the LHS or RHS and identity shuffles?
    bool isLHSID = true, isRHSID = true;

    for (unsigned i = 0, e = Mask.size(); i != e; ++i) {
      if (Mask[i] < 0) continue;  // Ignore undef values.
      // Is this an identity shuffle of the LHS value?
      isLHSID &= (Mask[i] == (int)i);

      // Is this an identity shuffle of the RHS value?
      isRHSID &= (Mask[i]-e == i);
    }

    // Eliminate identity shuffles.
    if (isLHSID) return ReplaceInstUsesWith(SVI, LHS);
    if (isRHSID) return ReplaceInstUsesWith(SVI, RHS);
  }

  // If the LHS is a shufflevector itself, see if we can combine it with this
  // one without producing an unusual shuffle.
  // Cases that might be simplified:
  // 1.
  // x1=shuffle(v1,v2,mask1)
  //  x=shuffle(x1,undef,mask)
  //        ==>
  //  x=shuffle(v1,undef,newMask)
  // newMask[i] = (mask[i] < x1.size()) ? mask1[mask[i]] : -1
  // 2.
  // x1=shuffle(v1,undef,mask1)
  //  x=shuffle(x1,x2,mask)
  // where v1.size() == mask1.size()
  //        ==>
  //  x=shuffle(v1,x2,newMask)
  // newMask[i] = (mask[i] < x1.size()) ? mask1[mask[i]] : mask[i]
  // 3.
  // x2=shuffle(v2,undef,mask2)
  //  x=shuffle(x1,x2,mask)
  // where v2.size() == mask2.size()
  //        ==>
  //  x=shuffle(x1,v2,newMask)
  // newMask[i] = (mask[i] < x1.size())
  //              ? mask[i] : mask2[mask[i]-x1.size()]+x1.size()
  // 4.
  // x1=shuffle(v1,undef,mask1)
  // x2=shuffle(v2,undef,mask2)
  //  x=shuffle(x1,x2,mask)
  // where v1.size() == v2.size()
  //        ==>
  //  x=shuffle(v1,v2,newMask)
  // newMask[i] = (mask[i] < x1.size())
  //              ? mask1[mask[i]] : mask2[mask[i]-x1.size()]+v1.size()
  //
  // Here we are really conservative:
  // we are absolutely afraid of producing a shuffle mask not in the input
  // program, because the code gen may not be smart enough to turn a merged
  // shuffle into two specific shuffles: it may produce worse code.  As such,
  // we only merge two shuffles if the result is either a splat or one of the
  // input shuffle masks.  In this case, merging the shuffles just removes
  // one instruction, which we know is safe.  This is good for things like
  // turning: (splat(splat)) -> splat, or
  // merge(V[0..n], V[n+1..2n]) -> V[0..2n]
  ShuffleVectorInst* LHSShuffle = dyn_cast<ShuffleVectorInst>(LHS);
  ShuffleVectorInst* RHSShuffle = dyn_cast<ShuffleVectorInst>(RHS);
  if (LHSShuffle)
    if (!isa<UndefValue>(LHSShuffle->getOperand(1)) && !isa<UndefValue>(RHS))
      LHSShuffle = NULL;
  if (RHSShuffle)
    if (!isa<UndefValue>(RHSShuffle->getOperand(1)))
      RHSShuffle = NULL;
  if (!LHSShuffle && !RHSShuffle)
    return MadeChange ? &SVI : 0;

  Value* LHSOp0 = NULL;
  Value* LHSOp1 = NULL;
  Value* RHSOp0 = NULL;
  unsigned LHSOp0Width = 0;
  unsigned RHSOp0Width = 0;
  if (LHSShuffle) {
    LHSOp0 = LHSShuffle->getOperand(0);
    LHSOp1 = LHSShuffle->getOperand(1);
    LHSOp0Width = cast<VectorType>(LHSOp0->getType())->getNumElements();
  }
  if (RHSShuffle) {
    RHSOp0 = RHSShuffle->getOperand(0);
    RHSOp0Width = cast<VectorType>(RHSOp0->getType())->getNumElements();
  }
  Value* newLHS = LHS;
  Value* newRHS = RHS;
  if (LHSShuffle) {
    // case 1
    if (isa<UndefValue>(RHS)) {
      newLHS = LHSOp0;
      newRHS = LHSOp1;
    }
    // case 2 or 4
    else if (LHSOp0Width == LHSWidth) {
      newLHS = LHSOp0;
    }
  }
  // case 3 or 4
  if (RHSShuffle && RHSOp0Width == LHSWidth) {
    newRHS = RHSOp0;
  }
  // case 4
  if (LHSOp0 == RHSOp0) {
    newLHS = LHSOp0;
    newRHS = NULL;
  }

  if (newLHS == LHS && newRHS == RHS)
    return MadeChange ? &SVI : 0;

  SmallVector<int, 16> LHSMask;
  SmallVector<int, 16> RHSMask;
  if (newLHS != LHS)
    LHSMask = LHSShuffle->getShuffleMask();
  if (RHSShuffle && newRHS != RHS)
    RHSMask = RHSShuffle->getShuffleMask();

  unsigned newLHSWidth = (newLHS != LHS) ? LHSOp0Width : LHSWidth;
  SmallVector<int, 16> newMask;
  bool isSplat = true;
  int SplatElt = -1;
  // Create a new mask for the new ShuffleVectorInst so that the new
  // ShuffleVectorInst is equivalent to the original one.
  for (unsigned i = 0; i < VWidth; ++i) {
    int eltMask;
    if (Mask[i] == -1) {
      // This element is an undef value.
      eltMask = -1;
    } else if (Mask[i] < (int)LHSWidth) {
      // This element is from left hand side vector operand.
      // 
      // If LHS is going to be replaced (case 1, 2, or 4), calculate the
      // new mask value for the element.
      if (newLHS != LHS) {
        eltMask = LHSMask[Mask[i]];
        // If the value selected is an undef value, explicitly specify it
        // with a -1 mask value.
        if (eltMask >= (int)LHSOp0Width && isa<UndefValue>(LHSOp1))
          eltMask = -1;
      }
      else
        eltMask = Mask[i];
    } else {
      // This element is from right hand side vector operand
      //
      // If the value selected is an undef value, explicitly specify it
      // with a -1 mask value. (case 1)
      if (isa<UndefValue>(RHS))
        eltMask = -1;
      // If RHS is going to be replaced (case 3 or 4), calculate the
      // new mask value for the element.
      else if (newRHS != RHS) {
        eltMask = RHSMask[Mask[i]-LHSWidth];
        // If the value selected is an undef value, explicitly specify it
        // with a -1 mask value.
        if (eltMask >= (int)RHSOp0Width) {
          assert(isa<UndefValue>(RHSShuffle->getOperand(1))
                 && "should have been check above");
          eltMask = -1;
        }
      }
      else
        eltMask = Mask[i]-LHSWidth;

      // If LHS's width is changed, shift the mask value accordingly.
      // If newRHS == NULL, i.e. LHSOp0 == RHSOp0, we want to remap any
      // references from RHSOp0 to LHSOp0, so we don't need to shift the mask.
      // If newRHS == newLHS, we want to remap any references from newRHS to
      // newLHS so that we can properly identify splats that may occur due to
      // obfuscation accross the two vectors.
      if (eltMask >= 0 && newRHS != NULL && newLHS != newRHS)
        eltMask += newLHSWidth;
    }

    // Check if this could still be a splat.
    if (eltMask >= 0) {
      if (SplatElt >= 0 && SplatElt != eltMask)
        isSplat = false;
      SplatElt = eltMask;
    }

    newMask.push_back(eltMask);
  }

  // If the result mask is equal to one of the original shuffle masks,
  // or is a splat, do the replacement.
  if (isSplat || newMask == LHSMask || newMask == RHSMask || newMask == Mask) {
    SmallVector<Constant*, 16> Elts;
    Type *Int32Ty = Type::getInt32Ty(SVI.getContext());
    for (unsigned i = 0, e = newMask.size(); i != e; ++i) {
      if (newMask[i] < 0) {
        Elts.push_back(UndefValue::get(Int32Ty));
      } else {
        Elts.push_back(ConstantInt::get(Int32Ty, newMask[i]));
      }
    }
    if (newRHS == NULL)
      newRHS = UndefValue::get(newLHS->getType());
    return new ShuffleVectorInst(newLHS, newRHS, ConstantVector::get(Elts));
  }

  return MadeChange ? &SVI : 0;
}

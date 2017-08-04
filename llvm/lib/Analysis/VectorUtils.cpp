//===----------- VectorUtils.cpp - Vectorizer utility functions -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines vectorizer utilities.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/VectorUtils.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/Analysis/DemandedBits.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Value.h"

using namespace llvm;
using namespace llvm::PatternMatch;

/// \brief Identify if the intrinsic is trivially vectorizable.
/// This method returns true if the intrinsic's argument types are all
/// scalars for the scalar form of the intrinsic and all vectors for
/// the vector form of the intrinsic.
bool llvm::isTriviallyVectorizable(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::sqrt:
  case Intrinsic::sin:
  case Intrinsic::cos:
  case Intrinsic::exp:
  case Intrinsic::exp2:
  case Intrinsic::log:
  case Intrinsic::log10:
  case Intrinsic::log2:
  case Intrinsic::fabs:
  case Intrinsic::minnum:
  case Intrinsic::maxnum:
  case Intrinsic::copysign:
  case Intrinsic::floor:
  case Intrinsic::ceil:
  case Intrinsic::trunc:
  case Intrinsic::rint:
  case Intrinsic::nearbyint:
  case Intrinsic::round:
  case Intrinsic::bswap:
  case Intrinsic::bitreverse:
  case Intrinsic::ctpop:
  case Intrinsic::pow:
  case Intrinsic::fma:
  case Intrinsic::fmuladd:
  case Intrinsic::ctlz:
  case Intrinsic::cttz:
  case Intrinsic::powi:
    return true;
  default:
    return false;
  }
}

/// \brief Identifies if the intrinsic has a scalar operand. It check for
/// ctlz,cttz and powi special intrinsics whose argument is scalar.
bool llvm::hasVectorInstrinsicScalarOpd(Intrinsic::ID ID,
                                        unsigned ScalarOpdIdx) {
  switch (ID) {
  case Intrinsic::ctlz:
  case Intrinsic::cttz:
  case Intrinsic::powi:
    return (ScalarOpdIdx == 1);
  default:
    return false;
  }
}

/// \brief Returns intrinsic ID for call.
/// For the input call instruction it finds mapping intrinsic and returns
/// its ID, in case it does not found it return not_intrinsic.
Intrinsic::ID llvm::getVectorIntrinsicIDForCall(const CallInst *CI,
                                                const TargetLibraryInfo *TLI) {
  Intrinsic::ID ID = getIntrinsicForCallSite(CI, TLI);
  if (ID == Intrinsic::not_intrinsic)
    return Intrinsic::not_intrinsic;

  if (isTriviallyVectorizable(ID) || ID == Intrinsic::lifetime_start ||
      ID == Intrinsic::lifetime_end || ID == Intrinsic::assume)
    return ID;
  return Intrinsic::not_intrinsic;
}

/// \brief Find the operand of the GEP that should be checked for consecutive
/// stores. This ignores trailing indices that have no effect on the final
/// pointer.
unsigned llvm::getGEPInductionOperand(const GetElementPtrInst *Gep) {
  const DataLayout &DL = Gep->getModule()->getDataLayout();
  unsigned LastOperand = Gep->getNumOperands() - 1;
  unsigned GEPAllocSize = DL.getTypeAllocSize(Gep->getResultElementType());

  // Walk backwards and try to peel off zeros.
  while (LastOperand > 1 && match(Gep->getOperand(LastOperand), m_Zero())) {
    // Find the type we're currently indexing into.
    gep_type_iterator GEPTI = gep_type_begin(Gep);
    std::advance(GEPTI, LastOperand - 2);

    // If it's a type with the same allocation size as the result of the GEP we
    // can peel off the zero index.
    if (DL.getTypeAllocSize(GEPTI.getIndexedType()) != GEPAllocSize)
      break;
    --LastOperand;
  }

  return LastOperand;
}

/// \brief If the argument is a GEP, then returns the operand identified by
/// getGEPInductionOperand. However, if there is some other non-loop-invariant
/// operand, it returns that instead.
Value *llvm::stripGetElementPtr(Value *Ptr, ScalarEvolution *SE, Loop *Lp) {
  GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Ptr);
  if (!GEP)
    return Ptr;

  unsigned InductionOperand = getGEPInductionOperand(GEP);

  // Check that all of the gep indices are uniform except for our induction
  // operand.
  for (unsigned i = 0, e = GEP->getNumOperands(); i != e; ++i)
    if (i != InductionOperand &&
        !SE->isLoopInvariant(SE->getSCEV(GEP->getOperand(i)), Lp))
      return Ptr;
  return GEP->getOperand(InductionOperand);
}

/// \brief If a value has only one user that is a CastInst, return it.
Value *llvm::getUniqueCastUse(Value *Ptr, Loop *Lp, Type *Ty) {
  Value *UniqueCast = nullptr;
  for (User *U : Ptr->users()) {
    CastInst *CI = dyn_cast<CastInst>(U);
    if (CI && CI->getType() == Ty) {
      if (!UniqueCast)
        UniqueCast = CI;
      else
        return nullptr;
    }
  }
  return UniqueCast;
}

/// \brief Get the stride of a pointer access in a loop. Looks for symbolic
/// strides "a[i*stride]". Returns the symbolic stride, or null otherwise.
Value *llvm::getStrideFromPointer(Value *Ptr, ScalarEvolution *SE, Loop *Lp) {
  auto *PtrTy = dyn_cast<PointerType>(Ptr->getType());
  if (!PtrTy || PtrTy->isAggregateType())
    return nullptr;

  // Try to remove a gep instruction to make the pointer (actually index at this
  // point) easier analyzable. If OrigPtr is equal to Ptr we are analzying the
  // pointer, otherwise, we are analyzing the index.
  Value *OrigPtr = Ptr;

  // The size of the pointer access.
  int64_t PtrAccessSize = 1;

  Ptr = stripGetElementPtr(Ptr, SE, Lp);
  const SCEV *V = SE->getSCEV(Ptr);

  if (Ptr != OrigPtr)
    // Strip off casts.
    while (const SCEVCastExpr *C = dyn_cast<SCEVCastExpr>(V))
      V = C->getOperand();

  const SCEVAddRecExpr *S = dyn_cast<SCEVAddRecExpr>(V);
  if (!S)
    return nullptr;

  V = S->getStepRecurrence(*SE);
  if (!V)
    return nullptr;

  // Strip off the size of access multiplication if we are still analyzing the
  // pointer.
  if (OrigPtr == Ptr) {
    if (const SCEVMulExpr *M = dyn_cast<SCEVMulExpr>(V)) {
      if (M->getOperand(0)->getSCEVType() != scConstant)
        return nullptr;

      const APInt &APStepVal = cast<SCEVConstant>(M->getOperand(0))->getAPInt();

      // Huge step value - give up.
      if (APStepVal.getBitWidth() > 64)
        return nullptr;

      int64_t StepVal = APStepVal.getSExtValue();
      if (PtrAccessSize != StepVal)
        return nullptr;
      V = M->getOperand(1);
    }
  }

  // Strip off casts.
  Type *StripedOffRecurrenceCast = nullptr;
  if (const SCEVCastExpr *C = dyn_cast<SCEVCastExpr>(V)) {
    StripedOffRecurrenceCast = C->getType();
    V = C->getOperand();
  }

  // Look for the loop invariant symbolic value.
  const SCEVUnknown *U = dyn_cast<SCEVUnknown>(V);
  if (!U)
    return nullptr;

  Value *Stride = U->getValue();
  if (!Lp->isLoopInvariant(Stride))
    return nullptr;

  // If we have stripped off the recurrence cast we have to make sure that we
  // return the value that is used in this loop so that we can replace it later.
  if (StripedOffRecurrenceCast)
    Stride = getUniqueCastUse(Stride, Lp, StripedOffRecurrenceCast);

  return Stride;
}

/// \brief Given a vector and an element number, see if the scalar value is
/// already around as a register, for example if it were inserted then extracted
/// from the vector.
Value *llvm::findScalarElement(Value *V, unsigned EltNo) {
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
      return nullptr;
    unsigned IIElt = cast<ConstantInt>(III->getOperand(2))->getZExtValue();

    // If this is an insert to the element we are looking for, return the
    // inserted value.
    if (EltNo == IIElt)
      return III->getOperand(1);

    // Otherwise, the insertelement doesn't modify the value, recurse on its
    // vector input.
    return findScalarElement(III->getOperand(0), EltNo);
  }

  if (ShuffleVectorInst *SVI = dyn_cast<ShuffleVectorInst>(V)) {
    unsigned LHSWidth = SVI->getOperand(0)->getType()->getVectorNumElements();
    int InEl = SVI->getMaskValue(EltNo);
    if (InEl < 0)
      return UndefValue::get(VTy->getElementType());
    if (InEl < (int)LHSWidth)
      return findScalarElement(SVI->getOperand(0), InEl);
    return findScalarElement(SVI->getOperand(1), InEl - LHSWidth);
  }

  // Extract a value from a vector add operation with a constant zero.
  Value *Val = nullptr; Constant *Con = nullptr;
  if (match(V, m_Add(m_Value(Val), m_Constant(Con))))
    if (Constant *Elt = Con->getAggregateElement(EltNo))
      if (Elt->isNullValue())
        return findScalarElement(Val, EltNo);

  // Otherwise, we don't know.
  return nullptr;
}

/// \brief Get splat value if the input is a splat vector or return nullptr.
/// This function is not fully general. It checks only 2 cases:
/// the input value is (1) a splat constants vector or (2) a sequence
/// of instructions that broadcast a single value into a vector.
///
const llvm::Value *llvm::getSplatValue(const Value *V) {

  if (auto *C = dyn_cast<Constant>(V))
    if (isa<VectorType>(V->getType()))
      return C->getSplatValue();

  auto *ShuffleInst = dyn_cast<ShuffleVectorInst>(V);
  if (!ShuffleInst)
    return nullptr;
  // All-zero (or undef) shuffle mask elements.
  for (int MaskElt : ShuffleInst->getShuffleMask())
    if (MaskElt != 0 && MaskElt != -1)
      return nullptr;
  // The first shuffle source is 'insertelement' with index 0.
  auto *InsertEltInst =
    dyn_cast<InsertElementInst>(ShuffleInst->getOperand(0));
  if (!InsertEltInst || !isa<ConstantInt>(InsertEltInst->getOperand(2)) ||
      !cast<ConstantInt>(InsertEltInst->getOperand(2))->isZero())
    return nullptr;

  return InsertEltInst->getOperand(1);
}

MapVector<Instruction *, uint64_t>
llvm::computeMinimumValueSizes(ArrayRef<BasicBlock *> Blocks, DemandedBits &DB,
                               const TargetTransformInfo *TTI) {

  // DemandedBits will give us every value's live-out bits. But we want
  // to ensure no extra casts would need to be inserted, so every DAG
  // of connected values must have the same minimum bitwidth.
  EquivalenceClasses<Value *> ECs;
  SmallVector<Value *, 16> Worklist;
  SmallPtrSet<Value *, 4> Roots;
  SmallPtrSet<Value *, 16> Visited;
  DenseMap<Value *, uint64_t> DBits;
  SmallPtrSet<Instruction *, 4> InstructionSet;
  MapVector<Instruction *, uint64_t> MinBWs;

  // Determine the roots. We work bottom-up, from truncs or icmps.
  bool SeenExtFromIllegalType = false;
  for (auto *BB : Blocks)
    for (auto &I : *BB) {
      InstructionSet.insert(&I);

      if (TTI && (isa<ZExtInst>(&I) || isa<SExtInst>(&I)) &&
          !TTI->isTypeLegal(I.getOperand(0)->getType()))
        SeenExtFromIllegalType = true;

      // Only deal with non-vector integers up to 64-bits wide.
      if ((isa<TruncInst>(&I) || isa<ICmpInst>(&I)) &&
          !I.getType()->isVectorTy() &&
          I.getOperand(0)->getType()->getScalarSizeInBits() <= 64) {
        // Don't make work for ourselves. If we know the loaded type is legal,
        // don't add it to the worklist.
        if (TTI && isa<TruncInst>(&I) && TTI->isTypeLegal(I.getType()))
          continue;

        Worklist.push_back(&I);
        Roots.insert(&I);
      }
    }
  // Early exit.
  if (Worklist.empty() || (TTI && !SeenExtFromIllegalType))
    return MinBWs;

  // Now proceed breadth-first, unioning values together.
  while (!Worklist.empty()) {
    Value *Val = Worklist.pop_back_val();
    Value *Leader = ECs.getOrInsertLeaderValue(Val);

    if (Visited.count(Val))
      continue;
    Visited.insert(Val);

    // Non-instructions terminate a chain successfully.
    if (!isa<Instruction>(Val))
      continue;
    Instruction *I = cast<Instruction>(Val);

    // If we encounter a type that is larger than 64 bits, we can't represent
    // it so bail out.
    if (DB.getDemandedBits(I).getBitWidth() > 64)
      return MapVector<Instruction *, uint64_t>();

    uint64_t V = DB.getDemandedBits(I).getZExtValue();
    DBits[Leader] |= V;
    DBits[I] = V;

    // Casts, loads and instructions outside of our range terminate a chain
    // successfully.
    if (isa<SExtInst>(I) || isa<ZExtInst>(I) || isa<LoadInst>(I) ||
        !InstructionSet.count(I))
      continue;

    // Unsafe casts terminate a chain unsuccessfully. We can't do anything
    // useful with bitcasts, ptrtoints or inttoptrs and it'd be unsafe to
    // transform anything that relies on them.
    if (isa<BitCastInst>(I) || isa<PtrToIntInst>(I) || isa<IntToPtrInst>(I) ||
        !I->getType()->isIntegerTy()) {
      DBits[Leader] |= ~0ULL;
      continue;
    }

    // We don't modify the types of PHIs. Reductions will already have been
    // truncated if possible, and inductions' sizes will have been chosen by
    // indvars.
    if (isa<PHINode>(I))
      continue;

    if (DBits[Leader] == ~0ULL)
      // All bits demanded, no point continuing.
      continue;

    for (Value *O : cast<User>(I)->operands()) {
      ECs.unionSets(Leader, O);
      Worklist.push_back(O);
    }
  }

  // Now we've discovered all values, walk them to see if there are
  // any users we didn't see. If there are, we can't optimize that
  // chain.
  for (auto &I : DBits)
    for (auto *U : I.first->users())
      if (U->getType()->isIntegerTy() && DBits.count(U) == 0)
        DBits[ECs.getOrInsertLeaderValue(I.first)] |= ~0ULL;

  for (auto I = ECs.begin(), E = ECs.end(); I != E; ++I) {
    uint64_t LeaderDemandedBits = 0;
    for (auto MI = ECs.member_begin(I), ME = ECs.member_end(); MI != ME; ++MI)
      LeaderDemandedBits |= DBits[*MI];

    uint64_t MinBW = (sizeof(LeaderDemandedBits) * 8) -
                     llvm::countLeadingZeros(LeaderDemandedBits);
    // Round up to a power of 2
    if (!isPowerOf2_64((uint64_t)MinBW))
      MinBW = NextPowerOf2(MinBW);

    // We don't modify the types of PHIs. Reductions will already have been
    // truncated if possible, and inductions' sizes will have been chosen by
    // indvars.
    // If we are required to shrink a PHI, abandon this entire equivalence class.
    bool Abort = false;
    for (auto MI = ECs.member_begin(I), ME = ECs.member_end(); MI != ME; ++MI)
      if (isa<PHINode>(*MI) && MinBW < (*MI)->getType()->getScalarSizeInBits()) {
        Abort = true;
        break;
      }
    if (Abort)
      continue;

    for (auto MI = ECs.member_begin(I), ME = ECs.member_end(); MI != ME; ++MI) {
      if (!isa<Instruction>(*MI))
        continue;
      Type *Ty = (*MI)->getType();
      if (Roots.count(*MI))
        Ty = cast<Instruction>(*MI)->getOperand(0)->getType();
      if (MinBW < Ty->getScalarSizeInBits())
        MinBWs[cast<Instruction>(*MI)] = MinBW;
    }
  }

  return MinBWs;
}

/// \returns \p I after propagating metadata from \p VL.
Instruction *llvm::propagateMetadata(Instruction *Inst, ArrayRef<Value *> VL) {
  Instruction *I0 = cast<Instruction>(VL[0]);
  SmallVector<std::pair<unsigned, MDNode *>, 4> Metadata;
  I0->getAllMetadataOtherThanDebugLoc(Metadata);

  for (auto Kind :
       {LLVMContext::MD_tbaa, LLVMContext::MD_alias_scope,
        LLVMContext::MD_noalias, LLVMContext::MD_fpmath,
        LLVMContext::MD_nontemporal, LLVMContext::MD_invariant_load}) {
    MDNode *MD = I0->getMetadata(Kind);

    for (int J = 1, E = VL.size(); MD && J != E; ++J) {
      const Instruction *IJ = cast<Instruction>(VL[J]);
      MDNode *IMD = IJ->getMetadata(Kind);
      switch (Kind) {
      case LLVMContext::MD_tbaa:
        MD = MDNode::getMostGenericTBAA(MD, IMD);
        break;
      case LLVMContext::MD_alias_scope:
        MD = MDNode::getMostGenericAliasScope(MD, IMD);
        break;
      case LLVMContext::MD_fpmath:
        MD = MDNode::getMostGenericFPMath(MD, IMD);
        break;
      case LLVMContext::MD_noalias:
      case LLVMContext::MD_nontemporal:
      case LLVMContext::MD_invariant_load:
        MD = MDNode::intersect(MD, IMD);
        break;
      default:
        llvm_unreachable("unhandled metadata");
      }
    }

    Inst->setMetadata(Kind, MD);
  }

  return Inst;
}

Constant *llvm::createInterleaveMask(IRBuilder<> &Builder, unsigned VF,
                                     unsigned NumVecs) {
  SmallVector<Constant *, 16> Mask;
  for (unsigned i = 0; i < VF; i++)
    for (unsigned j = 0; j < NumVecs; j++)
      Mask.push_back(Builder.getInt32(j * VF + i));

  return ConstantVector::get(Mask);
}

Constant *llvm::createStrideMask(IRBuilder<> &Builder, unsigned Start,
                                 unsigned Stride, unsigned VF) {
  SmallVector<Constant *, 16> Mask;
  for (unsigned i = 0; i < VF; i++)
    Mask.push_back(Builder.getInt32(Start + i * Stride));

  return ConstantVector::get(Mask);
}

Constant *llvm::createSequentialMask(IRBuilder<> &Builder, unsigned Start,
                                     unsigned NumInts, unsigned NumUndefs) {
  SmallVector<Constant *, 16> Mask;
  for (unsigned i = 0; i < NumInts; i++)
    Mask.push_back(Builder.getInt32(Start + i));

  Constant *Undef = UndefValue::get(Builder.getInt32Ty());
  for (unsigned i = 0; i < NumUndefs; i++)
    Mask.push_back(Undef);

  return ConstantVector::get(Mask);
}

/// A helper function for concatenating vectors. This function concatenates two
/// vectors having the same element type. If the second vector has fewer
/// elements than the first, it is padded with undefs.
static Value *concatenateTwoVectors(IRBuilder<> &Builder, Value *V1,
                                    Value *V2) {
  VectorType *VecTy1 = dyn_cast<VectorType>(V1->getType());
  VectorType *VecTy2 = dyn_cast<VectorType>(V2->getType());
  assert(VecTy1 && VecTy2 &&
         VecTy1->getScalarType() == VecTy2->getScalarType() &&
         "Expect two vectors with the same element type");

  unsigned NumElts1 = VecTy1->getNumElements();
  unsigned NumElts2 = VecTy2->getNumElements();
  assert(NumElts1 >= NumElts2 && "Unexpect the first vector has less elements");

  if (NumElts1 > NumElts2) {
    // Extend with UNDEFs.
    Constant *ExtMask =
        createSequentialMask(Builder, 0, NumElts2, NumElts1 - NumElts2);
    V2 = Builder.CreateShuffleVector(V2, UndefValue::get(VecTy2), ExtMask);
  }

  Constant *Mask = createSequentialMask(Builder, 0, NumElts1 + NumElts2, 0);
  return Builder.CreateShuffleVector(V1, V2, Mask);
}

Value *llvm::concatenateVectors(IRBuilder<> &Builder, ArrayRef<Value *> Vecs) {
  unsigned NumVecs = Vecs.size();
  assert(NumVecs > 1 && "Should be at least two vectors");

  SmallVector<Value *, 8> ResList;
  ResList.append(Vecs.begin(), Vecs.end());
  do {
    SmallVector<Value *, 8> TmpList;
    for (unsigned i = 0; i < NumVecs - 1; i += 2) {
      Value *V0 = ResList[i], *V1 = ResList[i + 1];
      assert((V0->getType() == V1->getType() || i == NumVecs - 2) &&
             "Only the last vector may have a different type");

      TmpList.push_back(concatenateTwoVectors(Builder, V0, V1));
    }

    // Push the last vector if the total number of vectors is odd.
    if (NumVecs % 2 != 0)
      TmpList.push_back(ResList[NumVecs - 1]);

    ResList = TmpList;
    NumVecs = ResList.size();
  } while (NumVecs > 1);

  return ResList[0];
}

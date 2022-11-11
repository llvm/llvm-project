//===- RISCVGatherScatterLowering.cpp - Gather/Scatter lowering -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass custom lowers llvm.gather and llvm.scatter instructions to
// RISCV intrinsics.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVTargetMachine.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "riscv-gather-scatter-lowering"

namespace {

class RISCVGatherScatterLowering : public FunctionPass {
  const RISCVSubtarget *ST = nullptr;
  const RISCVTargetLowering *TLI = nullptr;
  LoopInfo *LI = nullptr;
  const DataLayout *DL = nullptr;

  SmallVector<WeakTrackingVH> MaybeDeadPHIs;

  // Cache of the BasePtr and Stride determined from this GEP. When a GEP is
  // used by multiple gathers/scatters, this allow us to reuse the scalar
  // instructions we created for the first gather/scatter for the others.
  DenseMap<GetElementPtrInst *, std::pair<Value *, Value *>> StridedAddrs;

public:
  static char ID; // Pass identification, replacement for typeid

  RISCVGatherScatterLowering() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<LoopInfoWrapperPass>();
  }

  StringRef getPassName() const override {
    return "RISCV gather/scatter lowering";
  }

private:
  bool isLegalTypeAndAlignment(Type *DataType, Value *AlignOp);

  bool tryCreateStridedLoadStore(IntrinsicInst *II, Type *DataType, Value *Ptr,
                                 Value *AlignOp);

  std::pair<Value *, Value *> determineBaseAndStride(GetElementPtrInst *GEP,
                                                     IRBuilder<> &Builder);

  bool matchStridedRecurrence(Value *Index, Loop *L, Value *&Stride,
                              PHINode *&BasePtr, BinaryOperator *&Inc,
                              IRBuilder<> &Builder);
};

} // end anonymous namespace

char RISCVGatherScatterLowering::ID = 0;

INITIALIZE_PASS(RISCVGatherScatterLowering, DEBUG_TYPE,
                "RISCV gather/scatter lowering pass", false, false)

FunctionPass *llvm::createRISCVGatherScatterLoweringPass() {
  return new RISCVGatherScatterLowering();
}

bool RISCVGatherScatterLowering::isLegalTypeAndAlignment(Type *DataType,
                                                         Value *AlignOp) {
  Type *ScalarType = DataType->getScalarType();
  if (!TLI->isLegalElementTypeForRVV(ScalarType))
    return false;

  MaybeAlign MA = cast<ConstantInt>(AlignOp)->getMaybeAlignValue();
  if (MA && MA->value() < DL->getTypeStoreSize(ScalarType).getFixedSize())
    return false;

  // FIXME: Let the backend type legalize by splitting/widening?
  EVT DataVT = TLI->getValueType(*DL, DataType);
  if (!TLI->isTypeLegal(DataVT))
    return false;

  return true;
}

// TODO: Should we consider the mask when looking for a stride?
static std::pair<Value *, Value *> matchStridedConstant(Constant *StartC) {
  unsigned NumElts = cast<FixedVectorType>(StartC->getType())->getNumElements();

  // Check that the start value is a strided constant.
  auto *StartVal =
      dyn_cast_or_null<ConstantInt>(StartC->getAggregateElement((unsigned)0));
  if (!StartVal)
    return std::make_pair(nullptr, nullptr);
  APInt StrideVal(StartVal->getValue().getBitWidth(), 0);
  ConstantInt *Prev = StartVal;
  for (unsigned i = 1; i != NumElts; ++i) {
    auto *C = dyn_cast_or_null<ConstantInt>(StartC->getAggregateElement(i));
    if (!C)
      return std::make_pair(nullptr, nullptr);

    APInt LocalStride = C->getValue() - Prev->getValue();
    if (i == 1)
      StrideVal = LocalStride;
    else if (StrideVal != LocalStride)
      return std::make_pair(nullptr, nullptr);

    Prev = C;
  }

  Value *Stride = ConstantInt::get(StartVal->getType(), StrideVal);

  return std::make_pair(StartVal, Stride);
}

static std::pair<Value *, Value *> matchStridedStart(Value *Start,
                                                     IRBuilder<> &Builder) {
  // Base case, start is a strided constant.
  auto *StartC = dyn_cast<Constant>(Start);
  if (StartC)
    return matchStridedConstant(StartC);

  // Base case, start is a stepvector
  if (match(Start, m_Intrinsic<Intrinsic::experimental_stepvector>())) {
    auto *Ty = Start->getType()->getScalarType();
    return std::make_pair(ConstantInt::get(Ty, 0), ConstantInt::get(Ty, 1));
  }

  // Not a constant, maybe it's a strided constant with a splat added to it.
  auto *BO = dyn_cast<BinaryOperator>(Start);
  if (!BO || BO->getOpcode() != Instruction::Add)
    return std::make_pair(nullptr, nullptr);

  // Look for an operand that is splatted.
  unsigned OtherIndex = 1;
  Value *Splat = getSplatValue(BO->getOperand(0));
  if (!Splat) {
    Splat = getSplatValue(BO->getOperand(1));
    OtherIndex = 0;
  }
  if (!Splat)
    return std::make_pair(nullptr, nullptr);

  Value *Stride;
  std::tie(Start, Stride) = matchStridedStart(BO->getOperand(OtherIndex),
                                              Builder);
  if (!Start)
    return std::make_pair(nullptr, nullptr);

  // Add the splat value to the start.
  Builder.SetInsertPoint(BO);
  Builder.SetCurrentDebugLocation(DebugLoc());
  Start = Builder.CreateAdd(Start, Splat);
  return std::make_pair(Start, Stride);
}

// Recursively, walk about the use-def chain until we find a Phi with a strided
// start value. Build and update a scalar recurrence as we unwind the recursion.
// We also update the Stride as we unwind. Our goal is to move all of the
// arithmetic out of the loop.
bool RISCVGatherScatterLowering::matchStridedRecurrence(Value *Index, Loop *L,
                                                        Value *&Stride,
                                                        PHINode *&BasePtr,
                                                        BinaryOperator *&Inc,
                                                        IRBuilder<> &Builder) {
  // Our base case is a Phi.
  if (auto *Phi = dyn_cast<PHINode>(Index)) {
    // A phi node we want to perform this function on should be from the
    // loop header.
    if (Phi->getParent() != L->getHeader())
      return false;

    Value *Step, *Start;
    if (!matchSimpleRecurrence(Phi, Inc, Start, Step) ||
        Inc->getOpcode() != Instruction::Add)
      return false;
    assert(Phi->getNumIncomingValues() == 2 && "Expected 2 operand phi.");
    unsigned IncrementingBlock = Phi->getIncomingValue(0) == Inc ? 0 : 1;
    assert(Phi->getIncomingValue(IncrementingBlock) == Inc &&
           "Expected one operand of phi to be Inc");

    // Only proceed if the step is loop invariant.
    if (!L->isLoopInvariant(Step))
      return false;

    // Step should be a splat.
    Step = getSplatValue(Step);
    if (!Step)
      return false;

    std::tie(Start, Stride) = matchStridedStart(Start, Builder);
    if (!Start)
      return false;
    assert(Stride != nullptr);

    // Build scalar phi and increment.
    BasePtr =
        PHINode::Create(Start->getType(), 2, Phi->getName() + ".scalar", Phi);
    Inc = BinaryOperator::CreateAdd(BasePtr, Step, Inc->getName() + ".scalar",
                                    Inc);
    BasePtr->addIncoming(Start, Phi->getIncomingBlock(1 - IncrementingBlock));
    BasePtr->addIncoming(Inc, Phi->getIncomingBlock(IncrementingBlock));

    // Note that this Phi might be eligible for removal.
    MaybeDeadPHIs.push_back(Phi);
    return true;
  }

  // Otherwise look for binary operator.
  auto *BO = dyn_cast<BinaryOperator>(Index);
  if (!BO)
    return false;

  if (BO->getOpcode() != Instruction::Add &&
      BO->getOpcode() != Instruction::Or &&
      BO->getOpcode() != Instruction::Mul &&
      BO->getOpcode() != Instruction::Shl)
    return false;

  // Only support shift by constant.
  if (BO->getOpcode() == Instruction::Shl && !isa<Constant>(BO->getOperand(1)))
    return false;

  // We need to be able to treat Or as Add.
  if (BO->getOpcode() == Instruction::Or &&
      !haveNoCommonBitsSet(BO->getOperand(0), BO->getOperand(1), *DL))
    return false;

  // We should have one operand in the loop and one splat.
  Value *OtherOp;
  if (isa<Instruction>(BO->getOperand(0)) &&
      L->contains(cast<Instruction>(BO->getOperand(0)))) {
    Index = cast<Instruction>(BO->getOperand(0));
    OtherOp = BO->getOperand(1);
  } else if (isa<Instruction>(BO->getOperand(1)) &&
             L->contains(cast<Instruction>(BO->getOperand(1)))) {
    Index = cast<Instruction>(BO->getOperand(1));
    OtherOp = BO->getOperand(0);
  } else {
    return false;
  }

  // Make sure other op is loop invariant.
  if (!L->isLoopInvariant(OtherOp))
    return false;

  // Make sure we have a splat.
  Value *SplatOp = getSplatValue(OtherOp);
  if (!SplatOp)
    return false;

  // Recurse up the use-def chain.
  if (!matchStridedRecurrence(Index, L, Stride, BasePtr, Inc, Builder))
    return false;

  // Locate the Step and Start values from the recurrence.
  unsigned StepIndex = Inc->getOperand(0) == BasePtr ? 1 : 0;
  unsigned StartBlock = BasePtr->getOperand(0) == Inc ? 1 : 0;
  Value *Step = Inc->getOperand(StepIndex);
  Value *Start = BasePtr->getOperand(StartBlock);

  // We need to adjust the start value in the preheader.
  Builder.SetInsertPoint(
      BasePtr->getIncomingBlock(StartBlock)->getTerminator());
  Builder.SetCurrentDebugLocation(DebugLoc());

  switch (BO->getOpcode()) {
  default:
    llvm_unreachable("Unexpected opcode!");
  case Instruction::Add:
  case Instruction::Or: {
    // An add only affects the start value. It's ok to do this for Or because
    // we already checked that there are no common set bits.

    // If the start value is Zero, just take the SplatOp.
    if (isa<ConstantInt>(Start) && cast<ConstantInt>(Start)->isZero())
      Start = SplatOp;
    else
      Start = Builder.CreateAdd(Start, SplatOp, "start");
    BasePtr->setIncomingValue(StartBlock, Start);
    break;
  }
  case Instruction::Mul: {
    // If the start is zero we don't need to multiply.
    if (!isa<ConstantInt>(Start) || !cast<ConstantInt>(Start)->isZero())
      Start = Builder.CreateMul(Start, SplatOp, "start");

    Step = Builder.CreateMul(Step, SplatOp, "step");

    // If the Stride is 1 just take the SplatOpt.
    if (isa<ConstantInt>(Stride) && cast<ConstantInt>(Stride)->isOne())
      Stride = SplatOp;
    else
      Stride = Builder.CreateMul(Stride, SplatOp, "stride");
    Inc->setOperand(StepIndex, Step);
    BasePtr->setIncomingValue(StartBlock, Start);
    break;
  }
  case Instruction::Shl: {
    // If the start is zero we don't need to shift.
    if (!isa<ConstantInt>(Start) || !cast<ConstantInt>(Start)->isZero())
      Start = Builder.CreateShl(Start, SplatOp, "start");
    Step = Builder.CreateShl(Step, SplatOp, "step");
    Stride = Builder.CreateShl(Stride, SplatOp, "stride");
    Inc->setOperand(StepIndex, Step);
    BasePtr->setIncomingValue(StartBlock, Start);
    break;
  }
  }

  return true;
}

std::pair<Value *, Value *>
RISCVGatherScatterLowering::determineBaseAndStride(GetElementPtrInst *GEP,
                                                   IRBuilder<> &Builder) {

  auto I = StridedAddrs.find(GEP);
  if (I != StridedAddrs.end())
    return I->second;

  SmallVector<Value *, 2> Ops(GEP->operands());

  // Base pointer needs to be a scalar.
  if (Ops[0]->getType()->isVectorTy())
    return std::make_pair(nullptr, nullptr);

  Optional<unsigned> VecOperand;
  unsigned TypeScale = 0;

  // Look for a vector operand and scale.
  gep_type_iterator GTI = gep_type_begin(GEP);
  for (unsigned i = 1, e = GEP->getNumOperands(); i != e; ++i, ++GTI) {
    if (!Ops[i]->getType()->isVectorTy())
      continue;

    if (VecOperand)
      return std::make_pair(nullptr, nullptr);

    VecOperand = i;

    TypeSize TS = DL->getTypeAllocSize(GTI.getIndexedType());
    if (TS.isScalable())
      return std::make_pair(nullptr, nullptr);

    TypeScale = TS.getFixedSize();
  }

  // We need to find a vector index to simplify.
  if (!VecOperand)
    return std::make_pair(nullptr, nullptr);

  // We can't extract the stride if the arithmetic is done at a different size
  // than the pointer type. Adding the stride later may not wrap correctly.
  // Technically we could handle wider indices, but I don't expect that in
  // practice.
  Value *VecIndex = Ops[*VecOperand];
  Type *VecIntPtrTy = DL->getIntPtrType(GEP->getType());
  if (VecIndex->getType() != VecIntPtrTy)
    return std::make_pair(nullptr, nullptr);

  // Handle the non-recursive case.  This is what we see if the vectorizer
  // decides to use a scalar IV + vid on demand instead of a vector IV.
  auto [Start, Stride] = matchStridedStart(VecIndex, Builder);
  if (Start) {
    assert(Stride);
    Builder.SetInsertPoint(GEP);

    // Replace the vector index with the scalar start and build a scalar GEP.
    Ops[*VecOperand] = Start;
    Type *SourceTy = GEP->getSourceElementType();
    Value *BasePtr =
      Builder.CreateGEP(SourceTy, Ops[0], makeArrayRef(Ops).drop_front());

    // Convert stride to pointer size if needed.
    Type *IntPtrTy = DL->getIntPtrType(BasePtr->getType());
    assert(Stride->getType() == IntPtrTy && "Unexpected type");

    // Scale the stride by the size of the indexed type.
    if (TypeScale != 1)
      Stride = Builder.CreateMul(Stride, ConstantInt::get(IntPtrTy, TypeScale));

    auto P = std::make_pair(BasePtr, Stride);
    StridedAddrs[GEP] = P;
    return P;
  }

  // Make sure we're in a loop and that has a pre-header and a single latch.
  Loop *L = LI->getLoopFor(GEP->getParent());
  if (!L || !L->getLoopPreheader() || !L->getLoopLatch())
    return std::make_pair(nullptr, nullptr);

  BinaryOperator *Inc;
  PHINode *BasePhi;
  if (!matchStridedRecurrence(VecIndex, L, Stride, BasePhi, Inc, Builder))
    return std::make_pair(nullptr, nullptr);

  assert(BasePhi->getNumIncomingValues() == 2 && "Expected 2 operand phi.");
  unsigned IncrementingBlock = BasePhi->getOperand(0) == Inc ? 0 : 1;
  assert(BasePhi->getIncomingValue(IncrementingBlock) == Inc &&
         "Expected one operand of phi to be Inc");

  Builder.SetInsertPoint(GEP);

  // Replace the vector index with the scalar phi and build a scalar GEP.
  Ops[*VecOperand] = BasePhi;
  Type *SourceTy = GEP->getSourceElementType();
  Value *BasePtr =
      Builder.CreateGEP(SourceTy, Ops[0], makeArrayRef(Ops).drop_front());

  // Final adjustments to stride should go in the start block.
  Builder.SetInsertPoint(
      BasePhi->getIncomingBlock(1 - IncrementingBlock)->getTerminator());

  // Convert stride to pointer size if needed.
  Type *IntPtrTy = DL->getIntPtrType(BasePtr->getType());
  assert(Stride->getType() == IntPtrTy && "Unexpected type");

  // Scale the stride by the size of the indexed type.
  if (TypeScale != 1)
    Stride = Builder.CreateMul(Stride, ConstantInt::get(IntPtrTy, TypeScale));

  auto P = std::make_pair(BasePtr, Stride);
  StridedAddrs[GEP] = P;
  return P;
}

bool RISCVGatherScatterLowering::tryCreateStridedLoadStore(IntrinsicInst *II,
                                                           Type *DataType,
                                                           Value *Ptr,
                                                           Value *AlignOp) {
  // Make sure the operation will be supported by the backend.
  if (!isLegalTypeAndAlignment(DataType, AlignOp))
    return false;

  // Pointer should be a GEP.
  auto *GEP = dyn_cast<GetElementPtrInst>(Ptr);
  if (!GEP)
    return false;

  IRBuilder<> Builder(GEP);

  Value *BasePtr, *Stride;
  std::tie(BasePtr, Stride) = determineBaseAndStride(GEP, Builder);
  if (!BasePtr)
    return false;
  assert(Stride != nullptr);

  Builder.SetInsertPoint(II);

  CallInst *Call;
  if (II->getIntrinsicID() == Intrinsic::masked_gather)
    Call = Builder.CreateIntrinsic(
        Intrinsic::riscv_masked_strided_load,
        {DataType, BasePtr->getType(), Stride->getType()},
        {II->getArgOperand(3), BasePtr, Stride, II->getArgOperand(2)});
  else
    Call = Builder.CreateIntrinsic(
        Intrinsic::riscv_masked_strided_store,
        {DataType, BasePtr->getType(), Stride->getType()},
        {II->getArgOperand(0), BasePtr, Stride, II->getArgOperand(3)});

  Call->takeName(II);
  II->replaceAllUsesWith(Call);
  II->eraseFromParent();

  if (GEP->use_empty())
    RecursivelyDeleteTriviallyDeadInstructions(GEP);

  return true;
}

bool RISCVGatherScatterLowering::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  auto &TPC = getAnalysis<TargetPassConfig>();
  auto &TM = TPC.getTM<RISCVTargetMachine>();
  ST = &TM.getSubtarget<RISCVSubtarget>(F);
  if (!ST->hasVInstructions() || !ST->useRVVForFixedLengthVectors())
    return false;

  TLI = ST->getTargetLowering();
  DL = &F.getParent()->getDataLayout();
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

  StridedAddrs.clear();

  SmallVector<IntrinsicInst *, 4> Gathers;
  SmallVector<IntrinsicInst *, 4> Scatters;

  bool Changed = false;

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I);
      if (II && II->getIntrinsicID() == Intrinsic::masked_gather) {
        Gathers.push_back(II);
      } else if (II && II->getIntrinsicID() == Intrinsic::masked_scatter) {
        Scatters.push_back(II);
      }
    }
  }

  // Rewrite gather/scatter to form strided load/store if possible.
  for (auto *II : Gathers)
    Changed |= tryCreateStridedLoadStore(
        II, II->getType(), II->getArgOperand(0), II->getArgOperand(1));
  for (auto *II : Scatters)
    Changed |=
        tryCreateStridedLoadStore(II, II->getArgOperand(0)->getType(),
                                  II->getArgOperand(1), II->getArgOperand(2));

  // Remove any dead phis.
  while (!MaybeDeadPHIs.empty()) {
    if (auto *Phi = dyn_cast_or_null<PHINode>(MaybeDeadPHIs.pop_back_val()))
      RecursivelyDeleteDeadPHINode(Phi);
  }

  return Changed;
}

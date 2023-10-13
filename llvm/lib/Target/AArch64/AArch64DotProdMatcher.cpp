//===- AArch64DotProdMatcher - Matches instruction sequences to *DOT ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass recognizes and transforms IR to make use of two relatively simple
// cases that can be implemented by the SDOT and UDOT instructions on AArch64
// in order to increase vector unit bandwidth.
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Local.h"
#include "Utils/AArch64BaseInfo.h"
#include <deque>
#include <optional>
#include <tuple>
#include <utility>

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "aarch64-dot-product-matcher"

#define DOT_ACCUMULATOR_DEPTH (4)

STATISTIC(NumDOTInstrs, "Number of DOT Instructions generated.");
STATISTIC(NumSimpleDOTReplacements, "Num of simple dot patterns replaced.");
STATISTIC(NumLoopDOTReplacements, "Num of loop dot patterns replaced.");

struct LoopAccumulate {
  Value *RVal;
  PHINode *Phi;
  Value *IterVals;
  Value *Predicate;
  Value *Mul;
  Value *ValA;
  Value *ValB;
  VectorType *VTy;
  Type *AccTy;
  BasicBlock *LoopBlock;
  BasicBlock *PHBlock;
  bool IsSExt;

  LoopAccumulate(Value *RVal, PHINode *Phi, Value *IterVals, Value *Predicate,
                 Value *Mul, Value *ValA, Value *ValB, VectorType *VTy,
                 Type *AccTy, BasicBlock *LoopBlock, BasicBlock *PHBlock,
                 bool IsSExt)
    : RVal(RVal), Phi(Phi), IterVals(IterVals), Predicate(Predicate),
    Mul(Mul), ValA(ValA), ValB(ValB), VTy(VTy), AccTy(AccTy), LoopBlock(LoopBlock),
    PHBlock(PHBlock), IsSExt(IsSExt) {}
};

// Returns true if the instruction in question is an vector integer add
// reduction intrinsic.
static bool isScalableIntegerSumReduction(Instruction &I) {
  auto *II = dyn_cast<IntrinsicInst>(&I);
  return II &&
         II->getIntrinsicID() == Intrinsic::vector_reduce_add &&
         isa<ScalableVectorType>(II->getOperand(0)->getType());
}

// Returns a vector type for a dot product accumulator if the element type and
// extended element type are suitable, or a nullptr if not.
static Type *getAccumulatorType(Type *EltTy, Type *ExtEltTy, ElementCount EC) {
  Type *AccEltTy = nullptr;
  if (EltTy->isIntegerTy(8) && ExtEltTy->getPrimitiveSizeInBits() <= 32)
    AccEltTy = Type::getInt32Ty(EltTy->getContext());
  else if (EltTy->isIntegerTy(16) && ExtEltTy->getPrimitiveSizeInBits() <= 64)
    AccEltTy = Type::getInt64Ty(EltTy->getContext());

  if (AccEltTy)
    return VectorType::get(AccEltTy, EC);

  return nullptr;
}

// Returns either a pair of basic block pointers corresponding to the expected
// two incoming values for the phi, or None if one of the checks failed.
static std::optional<std::pair<BasicBlock*, BasicBlock*>>
getPHIIncomingBlocks(PHINode *Phi) {
  // Check PHI; we're only expecting the incoming value from within the loop
  // and one incoming value from a preheader.
  if (Phi->getNumIncomingValues() != 2)
    return std::nullopt;

  BasicBlock *PHBlock = Phi->getIncomingBlock(0);
  BasicBlock *LoopBlock = Phi->getIncomingBlock(1);
  // If this isn't a loop, or if it's a loop with multiple blocks, we bail
  // out for now. If needed we can improve this pass later.
  if (Phi->getParent() != LoopBlock && Phi->getParent() != PHBlock)
    return std::nullopt;

  // Make sure we know which incoming value belongs to the loop
  if (PHBlock == Phi->getParent())
    std::swap(LoopBlock, PHBlock);

  // If there's a non-null incoming value from the preheader, bail out for now.
  // We may be able to do better in future.
  Constant *Const = dyn_cast<Constant>(Phi->getIncomingValueForBlock(PHBlock));
  if (LoopBlock != Phi->getParent() || !Const || !Const->isNullValue())
    return std::nullopt;

  return std::make_pair(LoopBlock, PHBlock);
}

static bool checkLoopAcc(Value *RVal, PHINode *OldPHI, Value *IterVals,
                         SmallVectorImpl<LoopAccumulate> &Accumulators) {
  // Check a possible loop accumulator.
  bool IsSExt = false;

  // We only expect the add in the loop to be used by the reduction and by
  // the PHI node.
  if (!RVal->hasNUses(2) || !is_contained(OldPHI->incoming_values(), RVal)) {
    LLVM_DEBUG(dbgs() << "Loop sum operation has more than two uses or isn't "
                         "used by the accumulating PHI node.\n");
    return false;
  }

  // Look through selects with zeroinitializer. Record the predicate so
  // we can insert selects for the base values later.
  Value *Predicate = nullptr, *Mul = nullptr;
  if (!match(IterVals, m_Select(m_Value(Predicate), m_Value(Mul), m_Zero())))
    Mul = IterVals;

  Value *ValA = nullptr, *ValB = nullptr;
  // Match the core pattern of element-wise multiplication of extended values.
  if (match(Mul, m_OneUse(m_Mul(m_SExt(m_OneUse(m_Value(ValA))),
                                m_SExt(m_OneUse(m_Value(ValB)))))))
    IsSExt = true;
  else if (!match(Mul, m_OneUse(m_Mul(m_ZExt(m_OneUse(m_Value(ValA))),
                                      m_ZExt(m_OneUse(m_Value(ValB))))))) {
    LLVM_DEBUG(dbgs() << "Couldn't match inner loop multiply: "
                      << *Mul << "\n");
    return false;
  }

  // The same extended value could be used for both operands of the multiply,
  // so we just need to check that they have a single user.
  Instruction *I = dyn_cast<Instruction>(Mul);
  if (!I->getOperand(0)->hasOneUser() || !I->getOperand(1)->hasOneUser())
    return false;

  // Check that the vector type is one packed vector's worth of data.
  // TODO: Do we want to allow multiples?
  VectorType *ValTy = cast<VectorType>(ValA->getType());
  if (ValTy->getPrimitiveSizeInBits().getKnownMinValue() !=
      AArch64::SVEBitsPerBlock) {
    LLVM_DEBUG(dbgs() << "Vector base size is not a packed representation.\n");
    return false;
  }

  // Find the accumulator element type after extension and check that it isn't
  // too large; if it is, we might lose data by converting to dot instructions.
  // The element count needs to be 1/4th that of the input data, since the
  // dot product instructions take four smaller elements and multiply/accumulate
  // them into one larger element.
  Type *AccTy = getAccumulatorType(ValTy->getElementType(),
      Mul->getType()->getScalarType(),
      ValTy->getElementCount().divideCoefficientBy(4));

  if (!AccTy) {
    LLVM_DEBUG(dbgs() << "Accumulator element type too wide.\n");
    return false;
  }

  // Validate the phi node and retrieve the incoming basic blocks for the
  // accumulating loop itself and the preheader.
  auto PhiBlocks = getPHIIncomingBlocks(OldPHI);

  if (!PhiBlocks) {
    LLVM_DEBUG(dbgs() << "Unable to match PHI node\n");
    return false;
  }

  // Everything looks in order, so add it to the list of accumulators to
  // transform.
  Accumulators.emplace_back(RVal, OldPHI, IterVals, Predicate, Mul, ValA,
                            ValB, ValTy, AccTy, PhiBlocks->first,
                            PhiBlocks->second, IsSExt);
  return true;
}

static bool findDOTAccumulatorsInLoop(Value *RVal,
                                SmallVectorImpl<LoopAccumulate> &Accumulators,
                                unsigned Depth = DOT_ACCUMULATOR_DEPTH) {
  // Don't recurse too far.
  if (Depth == 0)
    return false;

  Value *V1 = nullptr, *V2 = nullptr;

  // Try to match the expected pattern from a sum reduction in
  // a vectorized loop.
  if (match(RVal, m_Add(m_Value(V1), m_Value(V2)))) {
    if (isa<PHINode>(V1) && !isa<PHINode>(V2) &&
        V1->hasOneUse() && V2->hasOneUse())
      return checkLoopAcc(RVal, cast<PHINode>(V1), V2, Accumulators);

    if (!isa<PHINode>(V1) && isa<PHINode>(V2) &&
        V1->hasOneUse() && V2->hasOneUse())
      return checkLoopAcc(RVal, cast<PHINode>(V2), V1, Accumulators);

    // Otherwise assume this is an intermediate multi-register reduction
    // and recurse to the operands.
    return findDOTAccumulatorsInLoop(V1, Accumulators, Depth - 1) &&
           findDOTAccumulatorsInLoop(V2, Accumulators, Depth - 1);
  }

  return false;
}

namespace {

class AArch64DotProdMatcher : public FunctionPass {
public:
  static char ID;
  AArch64DotProdMatcher() : FunctionPass(ID) {
    initializeAArch64DotProdMatcherPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);

    bool Changed = false;
    SmallVector<Instruction *, 4> Reductions;
    for (BasicBlock &Block : F)
      // TODO: Support non-scalable dot instructions too.
      for (Instruction &I : make_filter_range(Block,
                                              isScalableIntegerSumReduction))
        Reductions.push_back(&I);

    for (auto *Rdx : Reductions)
      Changed |= trySimpleDotReplacement(*Rdx) || tryLoopDotReplacement(*Rdx);

    return Changed;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.setPreservesCFG();
  }

  TargetTransformInfo *TTI;

private:
  bool trySimpleDotReplacement(Instruction &I);
  bool tryLoopDotReplacement(Instruction &I);
};

} // end anonymous namespace

char AArch64DotProdMatcher::ID = 0;
INITIALIZE_PASS_BEGIN(AArch64DotProdMatcher, DEBUG_TYPE,
                "AArch64 Dot Product Instruction Matcher", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(AArch64DotProdMatcher, DEBUG_TYPE,
                "AArch64 Dot Product Instruction Matcher", false, false)

FunctionPass *llvm::createAArch64DotProdMatcherPass() {
  return new AArch64DotProdMatcher();
}

// The following method looks for a simple pattern of two values being either
// sign or zero extended, multiplied together, then summed. If the types
// match the ones used by the [s|u]dot instructions (groups of 4x8 -> 32,
// groups of 4x16 -> 64) then we can replace the extends and multiply with a
// dot instruction and swap the reduce for one using fewer elements.
//
//      +-----------+   +-----------+
//      |   ValA    |   |   ValB    |
//      +-----+-----+   +-----+-----+
//            |               |
//            |               |
//      +-----v-----+   +-----v-----+
//      | [S|Z]Ext  |   | [S|Z]Ext  |
//      +-----+-----+   +-----+-----+
//            |               |
//            +--+         +--+
//               |         |
//              +v---------v+
//              |    Mul    |
//              +-----+-----+
//                    |
//                    |
//              +-----v-----+
//              | Reduce(+) |
//              +-----------+
bool AArch64DotProdMatcher::trySimpleDotReplacement(Instruction &I) {
  LLVM_DEBUG(dbgs() << "Looking for simple dot reduction: " << I << "\n");
  Value *RVal = I.getOperand(0);
  Value *ValA = nullptr, *ValB = nullptr;
  bool IsSExt = false;

  if (match(RVal, m_Mul(m_SExt(m_Value(ValA)), m_SExt(m_Value(ValB)))))
    IsSExt = true;
  else if (!match(RVal, m_Mul(m_ZExt(m_Value(ValA)), m_ZExt(m_Value(ValB))))) {
    LLVM_DEBUG(dbgs() << "Unable to match simple dot pattern\n");
    return false;
  }

  VectorType *ATy = cast<VectorType>(ValA->getType());
  VectorType *BTy = cast<VectorType>(ValB->getType());
  VectorType *MTy = cast<VectorType>(RVal->getType());
  if (ATy != BTy || !((ATy->getScalarType()->isIntegerTy(8) &&
                       MTy->getScalarType()->isIntegerTy(32)) ||
                      (ATy->getScalarType()->isIntegerTy(16) &&
                       MTy->getScalarType()->isIntegerTy(64)))) {
    LLVM_DEBUG(dbgs() << "Unable to match types for simple dot pattern\n");
    return false;
  }

  if (TTI->getRegisterBitWidth(TargetTransformInfo::RGK_ScalableVector) !=
      ATy->getPrimitiveSizeInBits())
    return false;

  // All conditions met, proceed with replacement.
  IRBuilder<> Builder(cast<Instruction>(RVal));

  // Need a new accumulator type.
  Type *AccTy = VectorType::get(MTy->getScalarType(),
                                MTy->getElementCount().divideCoefficientBy(4));
  Value *Zeroes = ConstantAggregateZero::get(AccTy);

  Intrinsic::ID IntID = IsSExt ? Intrinsic::aarch64_sve_sdot :
                                 Intrinsic::aarch64_sve_udot;
  Value *DotProd = Builder.CreateIntrinsic(IntID, {AccTy},
                                           {Zeroes, ValA, ValB});
  Builder.SetInsertPoint(&I);
  Value *Reduce = Builder.CreateAddReduce(DotProd);
  I.replaceAllUsesWith(Reduce);
  NumDOTInstrs++;
  NumSimpleDOTReplacements++;
  return true;
}

// This method looks for the following pattern: It starts from a sum
// reduction, but expects to find a vector add operation inside a loop with one
// of the operands being a PHI. The other operand can either be a select
// between zeroes and a multiply, or just the multiply directly. The rest of
// the pattern is the same as the simpler case -- multiply of extends of some
// values.
//
// Replacing this is a little tricky, since we need to replace the PHI node
// and accumulator as well, and potentially add in new selects earlier, but if
// everything checks out then the extend -> multiply -> inner loop add operation
// is replaced by the [s|u]dot instruction.
//
//                                     +-----------+
//                                     |   Zero    |
//                                     +-+---------+
//  +-------+      +---------------------+   |
//  |       |      |                         |
//  |    +--v------v-+                       |
//  |    |  OldPHI   |                       |
//  |    +--+--------+                       |
//  |       |                                |
//  |       |   +-----------+   +-----------+|
//  |       |   |   ValA    |   |   ValB    ||
//  |       |   +-----+-----+   +-----+-----+|
//  |       |         |               |      |
//  |       |         |               |      |
//  |       |   +-----v-----+   +-----v-----+|
//  |       |   | [S|Z]Ext  |   | [S|Z]Ext  ||
//  |       |   +-----+-----+   +-----+-----+|
//  |       |         |               |      |
//  |       |         +--+         +--+      |
//  |       |            |         |         |
//  |       |           +v---------v+        |
//  |       |           |    Mul    |        |
//  |       |           +-+---------+        |
//  |       |             |       +----------+
//  |       |             |       |
//  |       |           +-v-------v-+
//  |       |           |  Select   |
//  |       |           +--+--------+
//  |       |              |
//  |       |              |
//  |       |              |
//  |    +--v--------------v---+
//  |    |         Add         |
//  |    +--+-------+----------+
//  |       |       |
//  +-------+       |
//                  |
//            +-----v-----+
//            | Reduce(+) |
//            +-----------+
bool AArch64DotProdMatcher::tryLoopDotReplacement(Instruction &I) {
  LLVM_DEBUG(dbgs() << "Looking for Loop DOT Reduction: " << I << "\n");
  Value *RVal = I.getOperand(0);
  SmallVector<LoopAccumulate, 4> Accumulators;
  std::deque<Value *> RdxVals;
  IRBuilder<> Builder(&I);

  // If the loop was interleaved, we may have some intermediate add
  // instructions first before we get to the accumulators inside the
  // loop. Gather those first then process them.
  if (!findDOTAccumulatorsInLoop(RVal, Accumulators)) {
    LLVM_DEBUG(dbgs() << "Couldn't find DOT accumulators in the loop\n");
    return false;
  }

  // All conditions met, proceed with replacement.
  for (auto &Acc : Accumulators) {
    Builder.SetInsertPoint(Acc.Phi);

    // Plant new PHI node.
    PHINode *DotAcc = Builder.CreatePHI(Acc.AccTy, 2, "dot.accumulate");
    Value *Zeroes = ConstantAggregateZero::get(Acc.AccTy);
    DotAcc->addIncoming(Zeroes, Acc.PHBlock);

    // Move to the dot insertion point.
    Builder.SetInsertPoint(cast<Instruction>(Acc.RVal));

    // Need to generate selects for ValA and ValB if there was one before the
    // accumulate before.
    // Hopefully we can fold away some extra selects (e.g. if the data originally
    // came from masked loads with the same predicate).
    if (Acc.Predicate) {
      Value *Zeroes = ConstantAggregateZero::get(Acc.VTy);
      Acc.ValA = Builder.CreateSelect(Acc.Predicate, Acc.ValA, Zeroes);
      Acc.ValB = Builder.CreateSelect(Acc.Predicate, Acc.ValB, Zeroes);
    }

    // Now plant the dot instruction.
    Intrinsic::ID IntID = Acc.IsSExt ? Intrinsic::aarch64_sve_sdot :
                                          Intrinsic::aarch64_sve_udot;
    Value *DotProd = Builder.CreateIntrinsic(IntID, {Acc.AccTy},
                                             {DotAcc, Acc.ValA, Acc.ValB});
    DotAcc->addIncoming(DotProd, Acc.LoopBlock);

    RdxVals.push_back(DotProd);

    NumDOTInstrs++;
  }

  assert(!RdxVals.empty() &&
         "We found accumulators but generated no RdxVals");


  Builder.SetInsertPoint(cast<Instruction>(RVal));

  while (RdxVals.size() > 1) {
    RdxVals.push_back(Builder.CreateAdd(RdxVals[0], RdxVals[1]));
    // Drop the two RdxVals we just reduced. Sadly, there's no SmallDeque
    // with a pop_front_val() convenience method yet.
    RdxVals.pop_front();
    RdxVals.pop_front();
  }

  // Plant new reduction.
  Builder.SetInsertPoint(&I);
  Value *Reduce = Builder.CreateAddReduce(RdxVals.front());
  Value *Trunc = Builder.CreateTrunc(Reduce, I.getType(), "dot.trunc");
  I.replaceAllUsesWith(Trunc);


  // Delete the original reduction, since it's no longer required
  RecursivelyDeleteTriviallyDeadInstructions(&I);
  NumLoopDOTReplacements++;
  return true;
}


//===-------- LoopReduceMotion.cpp - Loop Reduce Motion Optimization ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This pass is designed to sink `ReduceCall` operations out of loops to reduce
// the number of instructions within the loop body.
//
// Below is the target pattern to be matched and the resulting pattern
// after the transformation.
//
// before                    | after
// ------                    | ------
// loop:                     | loop:
//   ...                     |   ...
//   d = reduce_add v        |   vsum = vadd vsum, v
//   sum = add sum, d        |   ...
//   ...                     |   ...
// exit:                     | exit:
//   value = sum             |   d = reduce_add vsum
//   ...                     |   value = d
//   ...                     |   ...
//   ret                     |   ret
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/LoopReduceMotion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

#define DEBUG_TYPE "loop-reduce-motion"

using namespace llvm;
InstructionCost LoopReduceMotionPass::getReductionPatternCost(
    Instruction *I, ElementCount VF, Type *Ty, TargetTransformInfo &TTI) {
  using namespace llvm::PatternMatch;
  auto *VectorTy = dyn_cast<VectorType>(Ty);
  if (!VectorTy)
    return InstructionCost::getInvalid();

  TTI::TargetCostKind CostKind = TargetTransformInfo::TCK_RecipThroughput;

  // Base cost: arithmetic reduction
  InstructionCost BaseCost = TTI.getArithmeticReductionCost(
      Instruction::Add, VectorTy, std::nullopt, CostKind);

  Value *RedOpVal = I->getOperand(0);
  Instruction *RedOp = dyn_cast<Instruction>(RedOpVal);
  if (!RedOp)
    return BaseCost;

  // Case 1: reduce(ext(A))
  if (match(RedOp, m_ZExtOrSExt(m_Value()))) {
    bool IsUnsigned = isa<ZExtInst>(RedOp);
    Type *SrcTy = RedOp->getOperand(0)->getType();
    auto *ExtType = VectorTy;

    InstructionCost ExtCost =
        TTI.getCastInstrCost(RedOp->getOpcode(), ExtType, SrcTy,
                             TTI::CastContextHint::None, CostKind, RedOp);

    InstructionCost RedCost = TTI.getExtendedReductionCost(
        Instruction::Add, IsUnsigned, Ty->getScalarType(),
        cast<VectorType>(SrcTy), std::nullopt, CostKind);

    if (RedCost.isValid() && RedCost < BaseCost + ExtCost)
      return RedCost;
  }

  // Case 2 & 3: reduce(mul(...))
  Value *Op0Val = nullptr;
  Value *Op1Val = nullptr;
  if (match(RedOp, m_Mul(m_Value(Op0Val), m_Value(Op1Val)))) {
    Instruction *Op0 = dyn_cast<Instruction>(Op0Val);
    Instruction *Op1 = dyn_cast<Instruction>(Op1Val);

    if (Op0 && Op1) {
      // Case 2: reduce(mul(ext(A), ext(B)))
      if (match(Op0, m_ZExtOrSExt(m_Value())) &&
          match(Op1, m_ZExtOrSExt(m_Value())) &&
          Op0->getOpcode() == Op1->getOpcode()) {
        bool IsUnsigned = isa<ZExtInst>(Op0);
        Type *Op0SrcTy = Op0->getOperand(0)->getType();
        Type *Op1SrcTy = Op1->getOperand(0)->getType();

        if (Op0SrcTy == Op1SrcTy) {
          InstructionCost ExtCost0 =
              TTI.getCastInstrCost(Op0->getOpcode(), VectorTy, Op0SrcTy,
                                   TTI::CastContextHint::None, CostKind, Op0);
          InstructionCost ExtCost1 =
              TTI.getCastInstrCost(Op1->getOpcode(), VectorTy, Op1SrcTy,
                                   TTI::CastContextHint::None, CostKind, Op1);
          InstructionCost MulCost =
              TTI.getArithmeticInstrCost(Instruction::Mul, VectorTy, CostKind);
          InstructionCost RedCost = TTI.getMulAccReductionCost(
              IsUnsigned, Instruction::Add, Ty->getScalarType(),
              cast<VectorType>(Op0SrcTy), CostKind);

          if (RedCost.isValid() &&
              RedCost < BaseCost + MulCost + ExtCost0 + ExtCost1)
            return RedCost;
        }
      }

      // Case 3: reduce(mul(A, B))
      InstructionCost MulCost =
          TTI.getArithmeticInstrCost(Instruction::Mul, VectorTy, CostKind);
      InstructionCost RedCost = TTI.getMulAccReductionCost(
          true, Instruction::Add, Ty->getScalarType(), VectorTy, CostKind);

      if (RedCost.isValid() && RedCost < BaseCost + MulCost)
        return RedCost;
    }
  }

  return BaseCost;
}

bool LoopReduceMotionPass::compareCost(LoopStandardAnalysisResults &AR, Loop &L,
                                       VectorType *VecTy,
                                       Instruction *ReduceInst) {
  auto TTI = &AR.TTI;
  auto SE = &AR.SE;

  TTI::TargetCostKind CostKind = TargetTransformInfo::TCK_RecipThroughput;

  InstructionCost VectorAddCost =
      TTI->getArithmeticInstrCost(Instruction::Add, VecTy, CostKind);
  InstructionCost ScalarAddCost = TTI->getArithmeticInstrCost(
      Instruction::Add, VecTy->getElementType(), CostKind);
  InstructionCost ReduceAddCost = getReductionPatternCost(
      ReduceInst, VecTy->getElementCount(), VecTy, *TTI);

  uint64_t FixedTripCount = SE->getSmallConstantTripCount(&L);
  if (FixedTripCount > 0) {
    InstructionCost beforeCost =
        FixedTripCount * (ReduceAddCost + ScalarAddCost);
    InstructionCost afterCost = FixedTripCount * VectorAddCost + ReduceAddCost;

    return afterCost < beforeCost;
  }

  if (VectorAddCost < ReduceAddCost && ScalarAddCost < ReduceAddCost / 2) {
    return true;
  }

  return false;
}

PreservedAnalyses LoopReduceMotionPass::run(Loop &L, LoopAnalysisManager &LAM,
                                            LoopStandardAnalysisResults &AR,
                                            LPMUpdater &Updater) {
  bool Changed = matchAndTransform(AR, L, &AR.DT, &AR.LI);

  if (!Changed)
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}

bool LoopReduceMotionPass::matchAndTransform(LoopStandardAnalysisResults &AR,
                                             Loop &L, DominatorTree *DT,
                                             LoopInfo *LI) {
  BasicBlock *Header = L.getHeader();
  BasicBlock *Latch = L.getLoopLatch();
  BasicBlock *ExitBlock = L.getExitBlock();
  BasicBlock *ExitingBlock = L.getExitingBlock();
  BasicBlock *LandingPad = nullptr;
  if (!Header || !Latch || !ExitBlock) {
    LLVM_DEBUG(dbgs() << "LRM: Skipping loop " << Header->getName()
                      << " because it is not a valid loop.\n");
    return false;
  }
  BasicBlock *Preheader = L.getLoopPreheader();
  if (!Preheader) {
    Preheader = InsertPreheaderForLoop(&L, DT, LI, nullptr, false);
    if (!Preheader) {
      LLVM_DEBUG(dbgs() << "LRM: Failed to create a preheader for loop "
                        << Header->getName() << ".\n");
      return false;
    }
  }

  bool Changed = false;
  SmallVector<Instruction *, 8> StackRecur;
  SmallVector<PHINode *, 8> Stack;
  int phiCount = 0;
  for (PHINode &PN : Header->phis()) {
    Stack.push_back(&PN);
    phiCount++;
    if (phiCount >= 8)
      return false;
  }

  while (!Stack.empty()) {
    PHINode *PN = Stack.pop_back_val();

    if (!PN->getType()->isIntegerTy())
      continue;

    auto SE = &AR.SE;
    RecurrenceDescriptor RecDesc;
    if (!RecurrenceDescriptor::isReductionPHI(PN, &L, RecDesc, nullptr, nullptr,
                                              nullptr, SE))
      continue;

    if (RecDesc.getRecurrenceKind() != RecurKind::Add)
      continue;

    Value *RecurrenceValueFromPHI = PN->getIncomingValueForBlock(Latch);
    Instruction *RecurrenceInst = dyn_cast<Instruction>(RecurrenceValueFromPHI);
    if (!RecurrenceInst || RecurrenceInst->getNumOperands() != 2)
      continue;

    // Don't match if the Recurrence Value has other use in loop
    bool hasOtherUse =
        llvm::any_of(RecurrenceValueFromPHI->users(), [&](User *U) {
          auto *Inst = dyn_cast<Instruction>(U);
          return Inst && Inst != PN && L.contains(Inst->getParent());
        });
    if (hasOtherUse)
      continue;

    Value *RecurrenceValue = RecurrenceInst->getOperand(0) == PN
                                 ? RecurrenceInst->getOperand(1)
                                 : RecurrenceInst->getOperand(0);
    Value *ReduceOperand;
    if (!llvm::PatternMatch::match(
            RecurrenceValue,
            llvm::PatternMatch::m_Intrinsic<Intrinsic::vector_reduce_add>(
                llvm::PatternMatch::m_Value(ReduceOperand))))
      continue;

    CallInst *ReduceCall = dyn_cast<CallInst>(RecurrenceValue);
    Instruction *VecIn = dyn_cast<Instruction>(ReduceOperand);
    VectorType *VecTy = cast<VectorType>(VecIn->getType());

    TargetTransformInfo &TTI = AR.TTI;
    if (TTI.preferInLoopReduction(RecurKind::Add, RecDesc.getRecurrenceType()))
      continue;
    if (!compareCost(AR, L, VecTy, ReduceCall))
      continue;
    // pattern match success
    LLVM_DEBUG(dbgs() << "Found pattern to optimize in loop "
                      << Header->getName() << "!\n");

    Value *VecZero = ConstantInt::get(VecTy, 0);

    // build new Vector Add to replace Scalar Add
    IRBuilder<> HeaderBuilder(Header, Header->getFirstNonPHIIt());
    PHINode *VecSumPhi = HeaderBuilder.CreatePHI(VecTy, 2, "vec.sum.phi");
    VecSumPhi->addIncoming(VecZero, Preheader);
    IRBuilder<> BodyBuilder(RecurrenceInst);
    Value *NewVecAdd = BodyBuilder.CreateAdd(VecSumPhi, VecIn, "vec.sum.next");
    VecSumPhi->addIncoming(NewVecAdd, Latch);

    // build landingPad for reduce add out of loop
    if (!LandingPad) {
      LandingPad = SplitEdge(ExitingBlock, ExitBlock, DT, LI);
      LandingPad->setName("loop.exit.landing");
    }
    IRBuilder<> LandingPadBuilder(LandingPad);
    LandingPadBuilder.SetInsertPoint(LandingPad->begin());

    // Create PHI node in LandingPad to maintain LCSSA form
    PHINode *VecSumExitPhi =
        LandingPadBuilder.CreatePHI(VecTy, 1, "vec.sum.exit.phi");
    VecSumExitPhi->addIncoming(NewVecAdd, ExitingBlock);

    LandingPadBuilder.SetInsertPoint(LandingPad->getTerminator());
    Value *ScalarTotalSum = LandingPadBuilder.CreateCall(
        ReduceCall->getCalledFunction(), VecSumExitPhi, "scalar.total.sum");

    Value *PreheaderValue = PN->getIncomingValueForBlock(Preheader);
    Value *LastAdd =
        PreheaderValue
            ? LandingPadBuilder.CreateAdd(PreheaderValue, ScalarTotalSum)
            : ScalarTotalSum;
    // replace the use of Recurrence Node and delete the dead Node
    Instruction *FinalNode = dyn_cast<Instruction>(LastAdd);
    if (!FinalNode)
      continue;

    // delete the dead PHI Node
    if (!PN->use_empty())
      PN->replaceAllUsesWith(PoisonValue::get(PN->getType()));
    RecursivelyDeleteDeadPHINode(PN);

    if (!RecurrenceInst->use_empty()) {
      for (User *U : make_early_inc_range(RecurrenceInst->users())) {
        auto *phi = dyn_cast<llvm::PHINode>(U);
        if (phi && phi->getParent() == LandingPad && !phi->use_empty()) {
          phi->replaceAllUsesWith(FinalNode);
          llvm::RecursivelyDeleteDeadPHINode(phi);
          Changed = true;
        }
      }
    }
  }

  return Changed;
}

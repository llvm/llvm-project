//===-- VPlanConstruction.cpp - Transforms for initial VPlan construction -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements transforms for initial VPlan construction.
///
//===----------------------------------------------------------------------===//

#include "LoopVectorizationPlanner.h"
#include "VPlan.h"
#include "VPlanCFG.h"
#include "VPlanDominatorTree.h"
#include "VPlanTransforms.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"

using namespace llvm;

/// Checks if \p HeaderVPB is a loop header block in the plain CFG; that is, it
/// has exactly 2 predecessors (preheader and latch), where the block
/// dominates the latch and the preheader dominates the block. If it is a
/// header block, returns a pair with the corresponding preheader and latch
/// blocks. Otherwise return std::nullopt.
static std::optional<std::pair<VPBasicBlock *, VPBasicBlock *>>
getPreheaderAndLatch(VPBlockBase *HeaderVPB, const VPDominatorTree &VPDT) {
  ArrayRef<VPBlockBase *> Preds = HeaderVPB->getPredecessors();
  if (Preds.size() != 2)
    return std::nullopt;

  auto *PreheaderVPBB = cast<VPBasicBlock>(Preds[0]);
  auto *LatchVPBB = cast<VPBasicBlock>(Preds[1]);
  if (VPDT.dominates(PreheaderVPBB, HeaderVPB) &&
      VPDT.dominates(HeaderVPB, LatchVPBB))
    return {std::make_pair(PreheaderVPBB, LatchVPBB)};

  std::swap(PreheaderVPBB, LatchVPBB);
  if (VPDT.dominates(PreheaderVPBB, HeaderVPB) &&
      VPDT.dominates(HeaderVPB, LatchVPBB))
    return {std::make_pair(PreheaderVPBB, LatchVPBB)};

  return std::nullopt;
}

/// Try to create a new VPRegionBlock if there is a loop starting at \p
/// HeaderVPB.
static void tryToCreateLoopRegion(VPlan &Plan, VPBlockBase *HeaderVPB,
                                  VPDominatorTree &VPDT) {
  auto Res = getPreheaderAndLatch(HeaderVPB, VPDT);
  if (!Res)
    return;

  const auto &[PreheaderVPBB, LatchVPBB] = *Res;

  // Swap the operands of header phis if needed. After creating the region, the
  // incoming value from the preheader must be the first operand and the one
  // from the latch must be the second operand.
  if (HeaderVPB->getPredecessors()[0] != PreheaderVPBB) {
    for (VPRecipeBase &R : cast<VPBasicBlock>(HeaderVPB)->phis()) {
      VPValue *Inc0 = R.getOperand(0);
      R.setOperand(0, R.getOperand(1));
      R.setOperand(1, Inc0);
    }
  }
  VPBlockUtils::disconnectBlocks(PreheaderVPBB, HeaderVPB);
  VPBlockUtils::disconnectBlocks(LatchVPBB, HeaderVPB);
  VPBlockBase *Succ = LatchVPBB->getSingleSuccessor();
  assert(LatchVPBB->getNumSuccessors() <= 1 &&
         "Latch has more than one successor");
  if (Succ)
    VPBlockUtils::disconnectBlocks(LatchVPBB, Succ);

  auto *R = Plan.createVPRegionBlock(HeaderVPB, LatchVPBB, "",
                                     false /*isReplicator*/);
  R->setParent(HeaderVPB->getParent());
  // All VPBB's reachable shallowly from HeaderVPB belong to top level loop,
  // because VPlan is expected to end at top level latch disconnected above.
  for (VPBlockBase *VPBB : vp_depth_first_shallow(HeaderVPB))
    VPBB->setParent(R);

  VPBlockUtils::insertBlockAfter(R, PreheaderVPBB);
  if (Succ)
    VPBlockUtils::connectBlocks(R, Succ);
}

void VPlanTransforms::createLoopRegions(VPlan &Plan, Type *InductionTy,
                                        PredicatedScalarEvolution &PSE,
                                        bool RequiresScalarEpilogueCheck,
                                        bool TailFolded, Loop *TheLoop) {
  VPDominatorTree VPDT;
  VPDT.recalculate(Plan);
  for (VPBlockBase *HeaderVPB : vp_depth_first_shallow(Plan.getEntry()))
    tryToCreateLoopRegion(Plan, HeaderVPB, VPDT);

  VPRegionBlock *TopRegion = Plan.getVectorLoopRegion();
  auto *OrigExiting = TopRegion->getExiting();
  VPBasicBlock *LatchVPBB = Plan.createVPBasicBlock("vector.latch");
  VPBlockUtils::insertBlockAfter(LatchVPBB, OrigExiting);
  TopRegion->setExiting(LatchVPBB);
  TopRegion->setName("vector loop");
  TopRegion->getEntryBasicBlock()->setName("vector.body");

  // Create SCEV and VPValue for the trip count.
  // We use the symbolic max backedge-taken-count, which works also when
  // vectorizing loops with uncountable early exits.
  const SCEV *BackedgeTakenCountSCEV = PSE.getSymbolicMaxBackedgeTakenCount();
  assert(!isa<SCEVCouldNotCompute>(BackedgeTakenCountSCEV) &&
         "Invalid loop count");
  ScalarEvolution &SE = *PSE.getSE();
  const SCEV *TripCount = SE.getTripCountFromExitCount(BackedgeTakenCountSCEV,
                                                       InductionTy, TheLoop);
  Plan.setTripCount(
      vputils::getOrCreateVPValueForSCEVExpr(Plan, TripCount, SE));

  VPBasicBlock *VecPreheader = Plan.createVPBasicBlock("vector.ph");
  VPBlockUtils::insertBlockAfter(VecPreheader, Plan.getEntry());

  VPBasicBlock *MiddleVPBB = Plan.createVPBasicBlock("middle.block");
  VPBlockUtils::insertBlockAfter(MiddleVPBB, TopRegion);

  VPBasicBlock *ScalarPH = Plan.createVPBasicBlock("scalar.ph");
  VPBlockUtils::connectBlocks(ScalarPH, Plan.getScalarHeader());
  if (!RequiresScalarEpilogueCheck) {
    VPBlockUtils::connectBlocks(MiddleVPBB, ScalarPH);
    return;
  }

  // If needed, add a check in the middle block to see if we have completed
  // all of the iterations in the first vector loop.  Three cases:
  // 1) If (N - N%VF) == N, then we *don't* need to run the remainder.
  //    Thus if tail is to be folded, we know we don't need to run the
  //    remainder and we can set the condition to true.
  // 2) If we require a scalar epilogue, there is no conditional branch as
  //    we unconditionally branch to the scalar preheader.  Do nothing.
  // 3) Otherwise, construct a runtime check.
  BasicBlock *IRExitBlock = TheLoop->getUniqueLatchExitBlock();
  auto *VPExitBlock = Plan.getExitBlock(IRExitBlock);
  // The connection order corresponds to the operands of the conditional branch.
  VPBlockUtils::insertBlockAfter(VPExitBlock, MiddleVPBB);
  VPBlockUtils::connectBlocks(MiddleVPBB, ScalarPH);

  auto *ScalarLatchTerm = TheLoop->getLoopLatch()->getTerminator();
  // Here we use the same DebugLoc as the scalar loop latch terminator instead
  // of the corresponding compare because they may have ended up with
  // different line numbers and we want to avoid awkward line stepping while
  // debugging. Eg. if the compare has got a line number inside the loop.
  VPBuilder Builder(MiddleVPBB);
  VPValue *Cmp =
      TailFolded
          ? Plan.getOrAddLiveIn(ConstantInt::getTrue(
                IntegerType::getInt1Ty(TripCount->getType()->getContext())))
          : Builder.createICmp(CmpInst::ICMP_EQ, Plan.getTripCount(),
                               &Plan.getVectorTripCount(),
                               ScalarLatchTerm->getDebugLoc(), "cmp.n");
  Builder.createNaryOp(VPInstruction::BranchOnCond, {Cmp},
                       ScalarLatchTerm->getDebugLoc());
}

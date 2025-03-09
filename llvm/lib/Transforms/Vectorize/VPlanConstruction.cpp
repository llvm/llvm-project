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
#include "VPlanTransforms.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"

using namespace llvm;

void VPlanTransforms::introduceTopLevelVectorLoopRegion(
    VPlan &Plan, Type *InductionTy, PredicatedScalarEvolution &PSE,
    bool RequiresScalarEpilogueCheck, bool TailFolded, Loop *TheLoop) {
  // TODO: Generalize to introduce all loop regions.
  auto *HeaderVPBB = cast<VPBasicBlock>(Plan.getEntry()->getSingleSuccessor());
  VPBlockUtils::disconnectBlocks(Plan.getEntry(), HeaderVPBB);

  VPBasicBlock *OriginalLatch =
      cast<VPBasicBlock>(HeaderVPBB->getSinglePredecessor());
  VPBlockUtils::disconnectBlocks(OriginalLatch, HeaderVPBB);
  VPBasicBlock *VecPreheader = Plan.createVPBasicBlock("vector.ph");
  VPBlockUtils::connectBlocks(Plan.getEntry(), VecPreheader);
  assert(OriginalLatch->getNumSuccessors() == 0 &&
         "Plan should end at top level latch");

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

  // Create VPRegionBlock, with existing header and new empty latch block, to be
  // filled.
  VPBasicBlock *LatchVPBB = Plan.createVPBasicBlock("vector.latch");
  VPBlockUtils::insertBlockAfter(LatchVPBB, OriginalLatch);
  auto *TopRegion = Plan.createVPRegionBlock(
      HeaderVPBB, LatchVPBB, "vector loop", false /*isReplicator*/);
  // All VPBB's reachable shallowly from HeaderVPBB belong to top level loop,
  // because VPlan is expected to end at top level latch.
  for (VPBlockBase *VPBB : vp_depth_first_shallow(HeaderVPBB))
    VPBB->setParent(TopRegion);

  VPBlockUtils::insertBlockAfter(TopRegion, VecPreheader);
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

//===-- VPlanPredicator.cpp - VPlan predicator ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements predication for VPlans.
///
//===----------------------------------------------------------------------===//

#include "VPRecipeBuilder.h"
#include "VPlan.h"
#include "VPlanCFG.h"
#include "VPlanTransforms.h"
#include "VPlanUtils.h"
#include "llvm/ADT/PostOrderIterator.h"

using namespace llvm;

struct VPPredicator {
  /// When we if-convert we need to create edge masks. We have to cache values
  /// so that we don't end up with exponential recursion/IR. Note that
  /// if-conversion currently takes place during VPlan-construction, so these
  /// caches are only used at that stage.
  using EdgeMaskCacheTy =
      DenseMap<std::pair<VPBasicBlock *, VPBasicBlock *>, VPValue *>;
  using BlockMaskCacheTy = DenseMap<VPBasicBlock *, VPValue *>;

  VPPredicator(VPRecipeBuilder &RecipeBuilder) : RecipeBuilder(RecipeBuilder) {}

  VPRecipeBuilder &RecipeBuilder;

  VPBuilder Builder;
  VPValue *createEdgeMask(VPBasicBlock *Src, VPBasicBlock *Dst) {
    assert(is_contained(Dst->getPredecessors(), Src) && "Invalid edge");

    // Look for cached value.
    VPValue *EdgeMask = RecipeBuilder.getEdgeMask(Src, Dst);
    if (EdgeMask)
      return EdgeMask;

    VPValue *SrcMask = RecipeBuilder.getBlockInMask(Src);

    // The terminator has to be a branch inst!
    if (Src->empty() || Src->getNumSuccessors() == 1) {
      RecipeBuilder.setEdgeMask(Src, Dst, SrcMask);
      return SrcMask;
    }

    auto *Term = cast<VPInstruction>(Src->getTerminator());
    if (Term->getOpcode() == Instruction::Switch) {
      createSwitchEdgeMasks(Term);
      return RecipeBuilder.getEdgeMask(Src, Dst);
    }

    auto *BI = cast<VPInstruction>(Src->getTerminator());
    assert(BI->getOpcode() == VPInstruction::BranchOnCond);
    if (Src->getSuccessors()[0] == Src->getSuccessors()[1]) {
      RecipeBuilder.setEdgeMask(Src, Dst, SrcMask);
      return SrcMask;
    }

    EdgeMask = BI->getOperand(0);
    assert(EdgeMask && "No Edge Mask found for condition");

    if (Src->getSuccessors()[0] != Dst)
      EdgeMask = Builder.createNot(EdgeMask, BI->getDebugLoc());

    if (SrcMask) { // Otherwise block in-mask is all-one, no need to AND.
      // The bitwise 'And' of SrcMask and EdgeMask introduces new UB if SrcMask
      // is false and EdgeMask is poison. Avoid that by using 'LogicalAnd'
      // instead which generates 'select i1 SrcMask, i1 EdgeMask, i1 false'.
      EdgeMask = Builder.createLogicalAnd(SrcMask, EdgeMask, BI->getDebugLoc());
    }

    RecipeBuilder.setEdgeMask(Src, Dst, EdgeMask);
    return EdgeMask;
  }

  VPValue *createBlockInMask(VPBasicBlock *VPBB) {
    Builder.setInsertPoint(VPBB, VPBB->begin());
    // All-one mask is modelled as no-mask following the convention for masked
    // load/store/gather/scatter. Initialize BlockMask to no-mask.
    VPValue *BlockMask = nullptr;
    // This is the block mask. We OR all unique incoming edges.
    for (auto *Predecessor : SetVector<VPBlockBase *>(
             VPBB->getPredecessors().begin(), VPBB->getPredecessors().end())) {
      VPValue *EdgeMask = createEdgeMask(cast<VPBasicBlock>(Predecessor), VPBB);
      if (!EdgeMask) { // Mask of predecessor is all-one so mask of block is
                       // too.
        RecipeBuilder.setBlockInMask(VPBB, EdgeMask);
        return EdgeMask;
      }

      if (!BlockMask) { // BlockMask has its initialized nullptr value.
        BlockMask = EdgeMask;
        continue;
      }

      BlockMask = Builder.createOr(BlockMask, EdgeMask, {});
    }

    RecipeBuilder.setBlockInMask(VPBB, BlockMask);
    return BlockMask;
  }

  void createHeaderMask(VPBasicBlock *HeaderVPBB, bool FoldTail) {
    if (!FoldTail) {
      RecipeBuilder.setBlockInMask(HeaderVPBB, nullptr);
      return;
    }

    // Introduce the early-exit compare IV <= BTC to form header block mask.
    // This is used instead of IV < TC because TC may wrap, unlike BTC. Start by
    // constructing the desired canonical IV in the header block as its first
    // non-phi instructions.

    auto NewInsertionPoint = HeaderVPBB->getFirstNonPhi();
    auto &Plan = *HeaderVPBB->getPlan();
    auto *IV = new VPWidenCanonicalIVRecipe(Plan.getCanonicalIV());
    HeaderVPBB->insert(IV, NewInsertionPoint);

    VPBuilder::InsertPointGuard Guard(Builder);
    Builder.setInsertPoint(HeaderVPBB, NewInsertionPoint);
    VPValue *BlockMask = nullptr;
    VPValue *BTC = Plan.getOrCreateBackedgeTakenCount();
    BlockMask = Builder.createICmp(CmpInst::ICMP_ULE, IV, BTC);
    RecipeBuilder.setBlockInMask(HeaderVPBB, BlockMask);
  }

  void createSwitchEdgeMasks(VPInstruction *SI) {
    VPBasicBlock *Src = SI->getParent();

    // Create masks where the terminator in Src is a switch. We create mask for
    // all edges at the same time. This is more efficient, as we can create and
    // collect compares for all cases once.
    VPValue *Cond = SI->getOperand(0);
    VPBasicBlock *DefaultDst = cast<VPBasicBlock>(Src->getSuccessors()[0]);
    MapVector<VPBasicBlock *, SmallVector<VPValue *>> Dst2Compares;
    for (const auto &[Idx, Succ] :
         enumerate(ArrayRef(Src->getSuccessors()).drop_front())) {
      VPBasicBlock *Dst = cast<VPBasicBlock>(Succ);
      // assert(!EdgeMaskCache.contains({Src, Dst}) && "Edge masks already
      // created");
      //  Cases whose destination is the same as default are redundant and can
      //  be ignored - they will get there anyhow.
      if (Dst == DefaultDst)
        continue;
      auto &Compares = Dst2Compares[Dst];
      VPValue *V = SI->getOperand(Idx + 1);
      Compares.push_back(Builder.createICmp(CmpInst::ICMP_EQ, Cond, V));
    }

    // We need to handle 2 separate cases below for all entries in Dst2Compares,
    // which excludes destinations matching the default destination.
    VPValue *SrcMask = RecipeBuilder.getBlockInMask(Src);
    VPValue *DefaultMask = nullptr;
    for (const auto &[Dst, Conds] : Dst2Compares) {
      // 1. Dst is not the default destination. Dst is reached if any of the
      // cases with destination == Dst are taken. Join the conditions for each
      // case whose destination == Dst using an OR.
      VPValue *Mask = Conds[0];
      for (VPValue *V : ArrayRef<VPValue *>(Conds).drop_front())
        Mask = Builder.createOr(Mask, V);
      if (SrcMask)
        Mask = Builder.createLogicalAnd(SrcMask, Mask);
      RecipeBuilder.setEdgeMask(Src, Dst, Mask);

      // 2. Create the mask for the default destination, which is reached if
      // none of the cases with destination != default destination are taken.
      // Join the conditions for each case where the destination is != Dst using
      // an OR and negate it.
      DefaultMask = DefaultMask ? Builder.createOr(DefaultMask, Mask) : Mask;
    }

    if (DefaultMask) {
      DefaultMask = Builder.createNot(DefaultMask);
      if (SrcMask)
        DefaultMask = Builder.createLogicalAnd(SrcMask, DefaultMask);
    }
    RecipeBuilder.setEdgeMask(Src, DefaultDst, DefaultMask);
  }
};

void VPlanTransforms::predicateAndLinearize(VPlan &Plan, bool FoldTail,
                                            VPRecipeBuilder &RecipeBuilder) {
  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  // Scan the body of the loop in a topological order to visit each basic block
  // after having visited its predecessor basic blocks.
  VPBasicBlock *Header = LoopRegion->getEntryBasicBlock();
  ReversePostOrderTraversal<VPBlockShallowTraversalWrapper<VPBlockBase *>> RPOT(
      Header);
  VPPredicator Predicator(RecipeBuilder);
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(RPOT)) {
    // Handle VPBBs down to the latch.
    if (VPBB == LoopRegion->getExiting())
      break;

    if (VPBB == Header) {
      Predicator.createHeaderMask(Header, FoldTail);
      continue;
    }
    SmallVector<VPWidenPHIRecipe *> Phis;
    for (VPRecipeBase &R : VPBB->phis())
      Phis.push_back(cast<VPWidenPHIRecipe>(&R));

    Predicator.createBlockInMask(VPBB);

    for (VPWidenPHIRecipe *Phi : Phis) {
      PHINode *IRPhi = cast<PHINode>(Phi->getUnderlyingValue());

      unsigned NumIncoming = IRPhi->getNumIncomingValues();

      // We know that all PHIs in non-header blocks are converted into selects,
      // so we don't have to worry about the insertion order and we can just use
      // the builder. At this point we generate the predication tree. There may
      // be duplications since this is a simple recursive scan, but future
      // optimizations will clean it up.

      // Map incoming IR BasicBlocks to incoming VPValues, for lookup below.
      // TODO: Add operands and masks in order from the VPlan predecessors.
      DenseMap<BasicBlock *, VPValue *> VPIncomingValues;
      DenseMap<BasicBlock *, VPBasicBlock *> VPIncomingBlocks;
      for (const auto &[Idx, Pred] :
           enumerate(predecessors(IRPhi->getParent()))) {
        VPIncomingValues[Pred] = Phi->getOperand(Idx);
        VPIncomingBlocks[Pred] =
            cast<VPBasicBlock>(VPBB->getPredecessors()[Idx]);
      }

      SmallVector<VPValue *, 2> OperandsWithMask;
      for (unsigned In = 0; In < NumIncoming; In++) {
        BasicBlock *Pred = IRPhi->getIncomingBlock(In);
        OperandsWithMask.push_back(VPIncomingValues.lookup(Pred));
        VPValue *EdgeMask =
            RecipeBuilder.getEdgeMask(VPIncomingBlocks.lookup(Pred), VPBB);
        if (!EdgeMask) {
          assert(In == 0 && "Both null and non-null edge masks found");
          assert(all_equal(Phi->operands()) &&
                 "Distinct incoming values with one having a full mask");
          break;
        }
        OperandsWithMask.push_back(EdgeMask);
      }
      auto *Blend = new VPBlendRecipe(IRPhi, OperandsWithMask);
      Blend->insertBefore(Phi);
      Phi->replaceAllUsesWith(Blend);
      Phi->eraseFromParent();
      RecipeBuilder.setRecipe(IRPhi, Blend);
    }
  }
}

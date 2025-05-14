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

namespace {
struct VPPredicator {
  using BlockMaskCacheTy = DenseMap<VPBasicBlock *, VPValue *>;
  VPPredicator(BlockMaskCacheTy &BlockMaskCache)
      : BlockMaskCache(BlockMaskCache) {}

  /// Builder to construct recipes to compute masks.
  VPBuilder Builder;

  /// When we if-convert we need to create edge masks. We have to cache values
  /// so that we don't end up with exponential recursion/IR.
  using EdgeMaskCacheTy =
      DenseMap<std::pair<const VPBasicBlock *, const VPBasicBlock *>,
               VPValue *>;
  EdgeMaskCacheTy EdgeMaskCache;

  BlockMaskCacheTy &BlockMaskCache;

  /// Returns the previously computed predicate of the edge between \p Src and
  /// \p Dst.
  VPValue *getEdgeMask(const VPBasicBlock *Src, const VPBasicBlock *Dst) const {
    return EdgeMaskCache.lookup({Src, Dst});
  }

  /// Returns the *entry* mask for \p VPBB.
  VPValue *getBlockInMask(VPBasicBlock *VPBB) const {
    return BlockMaskCache.lookup(VPBB);
  }
  void setBlockInMask(VPBasicBlock *VPBB, VPValue *Mask) {
    // TODO: Include the masks as operands in the predicated VPlan directly to
    // remove the need to keep a map of masks beyond the predication transform.
    assert(!BlockMaskCache.contains(VPBB) && "Mask already set");
    BlockMaskCache[VPBB] = Mask;
  }

  /// Compute and return the mask for the vector loop header block.
  void createHeaderMask(VPBasicBlock *HeaderVPBB, bool FoldTail);

  /// Compute and return the predicate of \p VPBB, assuming that the header
  /// block of the loop is set to True or the loop mask when tail folding.
  VPValue *createBlockInMask(VPBasicBlock *VPBB);

  /// Computes and return the predicate of the edge between \p Src and \p Dst.
  VPValue *createEdgeMask(VPBasicBlock *Src, VPBasicBlock *Dst);

  /// Create an edge mask for every destination of cases and/or default.
  void createSwitchEdgeMasks(VPInstruction *SI);
};
} // namespace

VPValue *VPPredicator::createEdgeMask(VPBasicBlock *Src, VPBasicBlock *Dst) {
  assert(is_contained(Dst->getPredecessors(), Src) && "Invalid edge");

  // Look for cached value.
  VPValue *EdgeMask = getEdgeMask(Src, Dst);
  if (EdgeMask)
    return EdgeMask;

  VPValue *SrcMask = getBlockInMask(Src);

  // The terminator has to be a branch inst!
  if (Src->empty() || Src->getNumSuccessors() == 1) {
    EdgeMaskCache[{Src, Dst}] = SrcMask;
    return SrcMask;
  }

  auto *Term = cast<VPInstruction>(Src->getTerminator());
  if (Term->getOpcode() == Instruction::Switch) {
    createSwitchEdgeMasks(Term);
    return getEdgeMask(Src, Dst);
  }

  auto *BI = cast<VPInstruction>(Src->getTerminator());
  assert(BI->getOpcode() == VPInstruction::BranchOnCond);
  if (Src->getSuccessors()[0] == Src->getSuccessors()[1]) {
    EdgeMaskCache[{Src, Dst}] = SrcMask;
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

  EdgeMaskCache[{Src, Dst}] = EdgeMask;
  return EdgeMask;
}

VPValue *VPPredicator::createBlockInMask(VPBasicBlock *VPBB) {
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
      setBlockInMask(VPBB, EdgeMask);
      return EdgeMask;
    }

    if (!BlockMask) { // BlockMask has its initialized nullptr value.
      BlockMask = EdgeMask;
      continue;
    }

    BlockMask = Builder.createOr(BlockMask, EdgeMask, {});
  }

  setBlockInMask(VPBB, BlockMask);
  return BlockMask;
}

void VPPredicator::createHeaderMask(VPBasicBlock *HeaderVPBB, bool FoldTail) {
  if (!FoldTail) {
    setBlockInMask(HeaderVPBB, nullptr);
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
  setBlockInMask(HeaderVPBB, BlockMask);
}

void VPPredicator::createSwitchEdgeMasks(VPInstruction *SI) {
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
    assert(!EdgeMaskCache.contains({Src, Dst}) && "Edge masks already created");
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
  VPValue *SrcMask = getBlockInMask(Src);
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
    EdgeMaskCache[{Src, Dst}] = Mask;

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
  EdgeMaskCache[{Src, DefaultDst}] = DefaultMask;
}

void VPlanTransforms::predicateAndLinearize(
    VPlan &Plan, bool FoldTail,
    DenseMap<VPBasicBlock *, VPValue *> &BlockMaskCache) {
  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  // Scan the body of the loop in a topological order to visit each basic block
  // after having visited its predecessor basic blocks.
  VPBasicBlock *Header = LoopRegion->getEntryBasicBlock();
  ReversePostOrderTraversal<VPBlockShallowTraversalWrapper<VPBlockBase *>> RPOT(
      Header);
  VPPredicator Predicator(BlockMaskCache);
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(RPOT)) {
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

      SmallVector<VPValue *, 2> OperandsWithMask;
      for (unsigned In = 0; In < NumIncoming; In++) {
        const VPBasicBlock *Pred = Phi->getIncomingBlock(In);
        OperandsWithMask.push_back(Phi->getIncomingValue(In));
        VPValue *EdgeMask = Predicator.getEdgeMask(Pred, VPBB);
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
    }
  }

  VPBlockBase *PrevVPBB = nullptr;
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(RPOT)) {
    // Handle VPBBs down to the latch.
    if (PrevVPBB && VPBB == LoopRegion->getExiting()) {
      VPBlockUtils::connectBlocks(PrevVPBB, VPBB);
      break;
    }

    auto Successors = to_vector(VPBB->getSuccessors());
    if (Successors.size() > 1)
      VPBB->getTerminator()->eraseFromParent();

    // Flatten the CFG in the loop. Masks for blocks have already been
    // generated and added to recipes as needed. To do so, first disconnect
    // VPBB from its successors. Then connect VPBB to the previously visited
    // VPBB.
    for (auto *Succ : Successors)
      VPBlockUtils::disconnectBlocks(VPBB, Succ);
    if (PrevVPBB)
      VPBlockUtils::connectBlocks(PrevVPBB, VPBB);

    PrevVPBB = VPBB;
  }
}

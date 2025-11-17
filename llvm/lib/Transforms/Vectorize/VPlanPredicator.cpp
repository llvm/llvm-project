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
#include "VPlanPatternMatch.h"
#include "VPlanTransforms.h"
#include "VPlanUtils.h"
#include "llvm/ADT/PostOrderIterator.h"

using namespace llvm;
using namespace VPlanPatternMatch;

namespace {
class VPPredicator {
  /// Builder to construct recipes to compute masks.
  VPBuilder Builder;

  /// When we if-convert we need to create edge masks. We have to cache values
  /// so that we don't end up with exponential recursion/IR.
  using EdgeMaskCacheTy =
      DenseMap<std::pair<const VPBasicBlock *, const VPBasicBlock *>,
               VPValue *>;
  using BlockMaskCacheTy = DenseMap<VPBasicBlock *, VPValue *>;
  EdgeMaskCacheTy EdgeMaskCache;

  BlockMaskCacheTy BlockMaskCache;

  /// Create an edge mask for every destination of cases and/or default.
  void createSwitchEdgeMasks(VPInstruction *SI);

  /// Computes and return the predicate of the edge between \p Src and \p Dst,
  /// possibly inserting new recipes at \p Dst (using Builder's insertion point)
  VPValue *createEdgeMask(VPBasicBlock *Src, VPBasicBlock *Dst);

  /// Returns the *entry* mask for \p VPBB.
  VPValue *getBlockInMask(VPBasicBlock *VPBB) const {
    return BlockMaskCache.lookup(VPBB);
  }

  /// Record \p Mask as the *entry* mask of \p VPBB, which is expected to not
  /// already have a mask.
  void setBlockInMask(VPBasicBlock *VPBB, VPValue *Mask) {
    // TODO: Include the masks as operands in the predicated VPlan directly to
    // avoid keeping the map of masks beyond the predication transform.
    assert(!getBlockInMask(VPBB) && "Mask already set");
    BlockMaskCache[VPBB] = Mask;
  }

  /// Record \p Mask as the mask of the edge from \p Src to \p Dst. The edge is
  /// expected to not have a mask already.
  VPValue *setEdgeMask(const VPBasicBlock *Src, const VPBasicBlock *Dst,
                       VPValue *Mask) {
    assert(Src != Dst && "Src and Dst must be different");
    assert(!getEdgeMask(Src, Dst) && "Mask already set");
    return EdgeMaskCache[{Src, Dst}] = Mask;
  }

public:
  /// Returns the precomputed predicate of the edge from \p Src to \p Dst.
  VPValue *getEdgeMask(const VPBasicBlock *Src, const VPBasicBlock *Dst) const {
    return EdgeMaskCache.lookup({Src, Dst});
  }

  /// Compute and return the mask for the vector loop header block.
  void createHeaderMask(VPBasicBlock *HeaderVPBB, bool FoldTail);

  /// Compute and return the predicate of \p VPBB, assuming that the header
  /// block of the loop is set to True, or to the loop mask when tail folding.
  VPValue *createBlockInMask(VPBasicBlock *VPBB);

  /// Convert phi recipes in \p VPBB to VPBlendRecipes.
  void convertPhisToBlends(VPBasicBlock *VPBB);

  const BlockMaskCacheTy getBlockMaskCache() const { return BlockMaskCache; }
};
} // namespace

VPValue *VPPredicator::createEdgeMask(VPBasicBlock *Src, VPBasicBlock *Dst) {
  assert(is_contained(Dst->getPredecessors(), Src) && "Invalid edge");

  // Look for cached value.
  VPValue *EdgeMask = getEdgeMask(Src, Dst);
  if (EdgeMask)
    return EdgeMask;

  VPValue *SrcMask = getBlockInMask(Src);

  // If there's a single successor, there's no terminator recipe.
  if (Src->getNumSuccessors() == 1)
    return setEdgeMask(Src, Dst, SrcMask);

  auto *Term = cast<VPInstruction>(Src->getTerminator());
  if (Term->getOpcode() == Instruction::Switch) {
    createSwitchEdgeMasks(Term);
    return getEdgeMask(Src, Dst);
  }

  assert(Term->getOpcode() == VPInstruction::BranchOnCond &&
         "Unsupported terminator");
  if (Src->getSuccessors()[0] == Src->getSuccessors()[1])
    return setEdgeMask(Src, Dst, SrcMask);

  EdgeMask = Term->getOperand(0);
  assert(EdgeMask && "No Edge Mask found for condition");

  if (Src->getSuccessors()[0] != Dst)
    EdgeMask = Builder.createNot(EdgeMask, Term->getDebugLoc());

  if (SrcMask) { // Otherwise block in-mask is all-one, no need to AND.
    // The bitwise 'And' of SrcMask and EdgeMask introduces new UB if SrcMask
    // is false and EdgeMask is poison. Avoid that by using 'LogicalAnd'
    // instead which generates 'select i1 SrcMask, i1 EdgeMask, i1 false'.
    EdgeMask = Builder.createLogicalAnd(SrcMask, EdgeMask, Term->getDebugLoc());
  }

  return setEdgeMask(Src, Dst, EdgeMask);
}

VPValue *VPPredicator::createBlockInMask(VPBasicBlock *VPBB) {
  // Start inserting after the block's phis, which be replaced by blends later.
  Builder.setInsertPoint(VPBB, VPBB->getFirstNonPhi());
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

    if (!BlockMask) { // BlockMask has its initial nullptr value.
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

  auto &Plan = *HeaderVPBB->getPlan();
  auto *IV =
      new VPWidenCanonicalIVRecipe(HeaderVPBB->getParent()->getCanonicalIV());
  Builder.setInsertPoint(HeaderVPBB, HeaderVPBB->getFirstNonPhi());
  Builder.insert(IV);

  VPValue *BTC = Plan.getOrCreateBackedgeTakenCount();
  VPValue *BlockMask = Builder.createICmp(CmpInst::ICMP_ULE, IV, BTC);
  setBlockInMask(HeaderVPBB, BlockMask);
}

void VPPredicator::createSwitchEdgeMasks(VPInstruction *SI) {
  VPBasicBlock *Src = SI->getParent();

  // Create masks where SI is a switch. We create masks for all edges from SI's
  // parent block at the same time. This is more efficient, as we can create and
  // collect compares for all cases once.
  VPValue *Cond = SI->getOperand(0);
  VPBasicBlock *DefaultDst = cast<VPBasicBlock>(Src->getSuccessors()[0]);
  MapVector<VPBasicBlock *, SmallVector<VPValue *>> Dst2Compares;
  for (const auto &[Idx, Succ] : enumerate(drop_begin(Src->getSuccessors()))) {
    VPBasicBlock *Dst = cast<VPBasicBlock>(Succ);
    assert(!getEdgeMask(Src, Dst) && "Edge masks already created");
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
    for (VPValue *V : drop_begin(Conds))
      Mask = Builder.createOr(Mask, V);
    if (SrcMask)
      Mask = Builder.createLogicalAnd(SrcMask, Mask);
    setEdgeMask(Src, Dst, Mask);

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
  setEdgeMask(Src, DefaultDst, DefaultMask);
}

void VPPredicator::convertPhisToBlends(VPBasicBlock *VPBB) {
  SmallVector<VPPhi *> Phis;
  for (VPRecipeBase &R : VPBB->phis())
    Phis.push_back(cast<VPPhi>(&R));
  for (VPPhi *PhiR : Phis) {
    // The non-header Phi is converted into a Blend recipe below,
    // so we don't have to worry about the insertion order and we can just use
    // the builder. At this point we generate the predication tree. There may
    // be duplications since this is a simple recursive scan, but future
    // optimizations will clean it up.

    SmallVector<VPValue *, 2> OperandsWithMask;
    for (const auto &[InVPV, InVPBB] : PhiR->incoming_values_and_blocks()) {
      OperandsWithMask.push_back(InVPV);
      VPValue *EdgeMask = getEdgeMask(InVPBB, VPBB);
      if (!EdgeMask) {
        assert(all_equal(PhiR->incoming_values()) &&
               "Distinct incoming values with one having a full mask");
        break;
      }

      OperandsWithMask.push_back(EdgeMask);
    }
    PHINode *IRPhi = cast_or_null<PHINode>(PhiR->getUnderlyingValue());
    auto *Blend =
        new VPBlendRecipe(IRPhi, OperandsWithMask, PhiR->getDebugLoc());
    Builder.insert(Blend);
    PhiR->replaceAllUsesWith(Blend);
    PhiR->eraseFromParent();
  }
}

DenseMap<VPBasicBlock *, VPValue *>
VPlanTransforms::introduceMasksAndLinearize(VPlan &Plan, bool FoldTail) {
  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  // Scan the body of the loop in a topological order to visit each basic block
  // after having visited its predecessor basic blocks.
  VPBasicBlock *Header = LoopRegion->getEntryBasicBlock();
  ReversePostOrderTraversal<VPBlockShallowTraversalWrapper<VPBlockBase *>> RPOT(
      Header);
  VPPredicator Predicator;
  for (VPBlockBase *VPB : RPOT) {
    // Non-outer regions with VPBBs only are supported at the moment.
    auto *VPBB = cast<VPBasicBlock>(VPB);
    // Introduce the mask for VPBB, which may introduce needed edge masks, and
    // convert all phi recipes of VPBB to blend recipes unless VPBB is the
    // header.
    if (VPBB == Header) {
      Predicator.createHeaderMask(Header, FoldTail);
      continue;
    }

    Predicator.createBlockInMask(VPBB);
    Predicator.convertPhisToBlends(VPBB);
  }

  // Linearize the blocks of the loop into one serial chain.
  VPBlockBase *PrevVPBB = nullptr;
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(RPOT)) {
    auto Successors = to_vector(VPBB->getSuccessors());
    if (Successors.size() > 1)
      VPBB->getTerminator()->eraseFromParent();

    // Flatten the CFG in the loop. To do so, first disconnect VPBB from its
    // successors. Then connect VPBB to the previously visited VPBB.
    for (auto *Succ : Successors)
      VPBlockUtils::disconnectBlocks(VPBB, Succ);
    if (PrevVPBB)
      VPBlockUtils::connectBlocks(PrevVPBB, VPBB);

    PrevVPBB = VPBB;
  }
  return Predicator.getBlockMaskCache();
}

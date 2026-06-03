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
#include "VPlanDominatorTree.h"
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

  /// Dominator tree for the VPlan.
  VPDominatorTree VPDT;

  /// Post-dominator tree for the VPlan.
  VPPostDominatorTree VPPDT;

  /// Post-dominator frontier for the VPlan.
  VPPostDominanceFrontier VPPDF;

  /// When we if-convert we need to create edge masks. We have to cache values
  /// so that we don't end up with exponential recursion/IR.
  using EdgeMaskCacheTy =
      DenseMap<std::pair<const VPBasicBlock *, const VPBasicBlock *>,
               VPValue *>;
  using BlockMaskCacheTy = DenseMap<const VPBasicBlock *, VPValue *>;
  EdgeMaskCacheTy EdgeMaskCache;

  BlockMaskCacheTy BlockMaskCache;

  /// Create an edge mask for every destination of cases and/or default.
  void createSwitchEdgeMasks(const VPInstruction *SI);

  /// Computes and return the predicate of the edge between \p Src and \p Dst,
  /// possibly inserting new recipes at \p Dst (using Builder's insertion point)
  VPValue *createEdgeMask(const VPBasicBlock *Src, const VPBasicBlock *Dst);

  /// Record \p Mask as the *entry* mask of \p VPBB, which is expected to not
  /// already have a mask.
  void setBlockInMask(const VPBasicBlock *VPBB, VPValue *Mask) {
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

  /// Returns where to insert new masks in \p VPBB.
  VPBasicBlock::iterator getMaskInsertPoint(VPBasicBlock *VPBB) {
    if (VPValue *Mask = getBlockInMask(VPBB))
      if (VPRecipeBase *MaskR = Mask->getDefiningRecipe())
        if (MaskR->getParent() == VPBB) // In-mask may be the IDom's.
          return std::next(MaskR->getIterator());
    return VPBB->getFirstNonPhi();
  }

  using EdgeTy = std::pair<const VPBasicBlock *, const VPBasicBlock *>;

  /// Compute the "furthest up" set of edges for each incoming value of \Phi.
  MapVector<EdgeTy, VPValue *> computeBlendEdges(VPPhi *Phi);

  /// Given a set of \p Edges that lead to \p VPBB, return the OR of all edges
  /// or an equivalent block in-mask.
  VPValue *createMaskDisjunction(ArrayRef<EdgeTy> Edges, VPBasicBlock *VPBB);

public:
  VPPredicator(VPlan &Plan) : VPDT(Plan), VPPDT(Plan), VPPDF(VPPDT) {}

  /// Returns the *entry* mask for \p VPBB.
  VPValue *getBlockInMask(const VPBasicBlock *VPBB) const {
    return BlockMaskCache.lookup(VPBB);
  }

  /// Returns the precomputed predicate of the edge from \p Src to \p Dst.
  VPValue *getEdgeMask(const VPBasicBlock *Src, const VPBasicBlock *Dst) const {
    return EdgeMaskCache.lookup({Src, Dst});
  }

  /// Compute the predicate of \p VPBB.
  void createBlockInMask(VPBasicBlock *VPBB);

  /// Convert phi recipes in \p VPBB to VPBlendRecipes.
  void convertPhisToBlends(VPBasicBlock *VPBB);
};
} // namespace

VPValue *VPPredicator::createEdgeMask(const VPBasicBlock *Src,
                                      const VPBasicBlock *Dst) {
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

void VPPredicator::createBlockInMask(VPBasicBlock *VPBB) {
  // Start inserting after the block's phis, which be replaced by blends later.
  Builder.setInsertPoint(VPBB, VPBB->getFirstNonPhi());

  // Reuse the mask of the immediate dominator if the VPBB post-dominates the
  // immediate dominator.
  auto *IDom = VPDT.getNode(VPBB)->getIDom();
  assert(IDom && "Block in loop must have immediate dominator");
  auto *IDomBB = cast<VPBasicBlock>(IDom->getBlock());
  if (VPPDT.properlyDominates(VPBB, IDomBB)) {
    setBlockInMask(VPBB, getBlockInMask(IDomBB));
    return;
  }
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
      return;
    }

    if (!BlockMask) { // BlockMask has its initial nullptr value.
      BlockMask = EdgeMask;
      continue;
    }

    BlockMask = Builder.createOr(BlockMask, EdgeMask, {});
  }

  setBlockInMask(VPBB, BlockMask);
}

void VPPredicator::createSwitchEdgeMasks(const VPInstruction *SI) {
  const VPBasicBlock *Src = SI->getParent();

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
  } else {
    // There are no destinations other than the default destination, so this is
    // an unconditional branch.
    DefaultMask = SrcMask;
  }
  setEdgeMask(Src, DefaultDst, DefaultMask);
}

// Compute the "furthest up" set of edges for each incoming value of a phi.
//
// Start by keeping track of what edges lead to which value. Then see if any
// node has the same value for all outgoing edges. If so then propagate that
// value up to every node it postdominates.
MapVector<VPPredicator::EdgeTy, VPValue *>
VPPredicator::computeBlendEdges(VPPhi *Phi) {
  MapVector<EdgeTy, VPValue *> Edges;

  // Mark the given edge as providing the value \p V.
  auto AddEdge = [&Edges](const VPBlockBase *From, const VPBlockBase *To,
                          VPValue *V) {
    EdgeTy Edge = {cast<VPBasicBlock>(From), cast<VPBasicBlock>(To)};
    assert((!Edges.contains(Edge) || Edges.lookup(Edge) == V) &&
           "Clobbering an edge?");
    Edges[Edge] = V;
  };

  for (auto [InVal, InVPBB] : Phi->incoming_values_and_blocks())
    AddEdge(InVPBB, Phi->getParent(), InVal);

  // The root phi must postdominate every incoming block. Also don't touch
  // phis in a reduction chain since they need to be in a specific structure
  // for handle*Reductions.
  for (auto [InVal, InVPBB] : Phi->incoming_values_and_blocks())
    if (!VPPDT.dominates(Phi->getParent(), InVPBB) ||
        isa<VPReductionPHIRecipe>(InVal))
      return Edges;

  // Given a list of edges, check if they all have the same value and return it.
  auto GetAllEqual = [&Edges](ArrayRef<EdgeTy> OutEdges) -> VPValue * {
    VPValue *Common = nullptr;
    for (EdgeTy E : OutEdges) {
      VPValue *V = Edges.lookup(E);
      if (!V)
        return nullptr;
      if (match(V, m_Poison()))
        continue;
      if (!Common)
        Common = V;
      else if (Common != V)
        return nullptr;
    }
    return Common;
  };

  SetVector<const VPBlockBase *> Worklist(from_range, Phi->incoming_blocks());
  while (!Worklist.empty()) {
    auto *VPBB = cast<VPBasicBlock>(Worklist.pop_back_val());

    // Check that all outgoing edges from VPBB have the same value.
    SmallVector<EdgeTy> OutEdges;
    for (const VPBlockBase *Succ : VPBB->getSuccessors())
      OutEdges.emplace_back(VPBB, cast<VPBasicBlock>(Succ));
    VPValue *Common = GetAllEqual(OutEdges);
    if (!Common)
      continue;

    // They have the same value: we can move the edges up
    for (EdgeTy Edge : OutEdges)
      Edges.erase(Edge);

    // Peek through phis that are postdominated by VPBB
    if (auto *Phi = dyn_cast<VPPhi>(Common))
      if (VPPDT.dominates(VPBB, Phi->getParent())) {
        for (auto [InV, InVPBB] : Phi->incoming_values_and_blocks()) {
          AddEdge(InVPBB, Phi->getParent(), InV);
          Worklist.insert(InVPBB);
        }
        continue;
      }

    // Iterate up through the post dominance frontier
    for (const VPBlockBase *Frontier : VPPDF.find(VPBB)->second) {
      for (const VPBlockBase *FrontierSucc : Frontier->getSuccessors())
        if (VPPDT.dominates(VPBB, FrontierSucc))
          AddEdge(Frontier, FrontierSucc, Common);
      Worklist.insert(cast<VPBasicBlock>(Frontier));
    }
  }

  return Edges;
}

VPValue *VPPredicator::createMaskDisjunction(ArrayRef<EdgeTy> Edges,
                                             VPBasicBlock *VPBB) {
  auto Dsts = map_range(Edges, [](auto E) { return E.second; });
  const VPBasicBlock *PostDom = *Dsts.begin();
  for (const VPBasicBlock *VPBB : drop_begin(Dsts))
    PostDom =
        cast<VPBasicBlock>(VPPDT.findNearestCommonDominator(PostDom, VPBB));
  assert(VPPDT.dominates(VPBB, PostDom) && "Edges don't postdominate VPBB");
  if (PostDom != VPBB)
    return getBlockInMask(PostDom);

  VPValue *Mask = nullptr;
  for (auto [Src, ConstDst] : Edges) {
    auto *Dst = const_cast<VPBasicBlock *>(ConstDst);
    VPValue *EdgeMask;
    {
      VPBuilder::InsertPointGuard Guard(Builder);
      Builder.setInsertPoint(Dst, getMaskInsertPoint(Dst));
      EdgeMask = createEdgeMask(Src, Dst);
    }
    Mask = Mask ? Builder.createOr(Mask, EdgeMask) : EdgeMask;
  }
  return Mask;
}

void VPPredicator::convertPhisToBlends(VPBasicBlock *VPBB) {
  Builder.setInsertPoint(VPBB, getMaskInsertPoint(VPBB));

  SmallVector<VPPhi *> Phis;
  for (VPRecipeBase &R : VPBB->phis())
    Phis.push_back(cast<VPPhi>(&R));
  for (VPPhi *PhiR : Phis) {
    // The non-header Phi is converted into a Blend recipe below,
    // so we don't have to worry about the insertion order and we can just use
    // the builder. At this point we generate the predication tree. There may
    // be duplications since this is a simple recursive scan, but future
    // optimizations will clean it up.

    auto NotPoison = make_filter_range(PhiR->incoming_values(), [](VPValue *V) {
      return !match(V, m_Poison());
    });
    if (all_equal(NotPoison)) {
      PhiR->replaceAllUsesWith(NotPoison.empty() ? PhiR->getIncomingValue(0)
                                                 : *NotPoison.begin());
      PhiR->eraseFromParent();
      continue;
    }

    MapVector<VPValue *, SmallVector<EdgeTy>> InValEdgesMap;
    for (auto [Edge, Val] : computeBlendEdges(PhiR))
      InValEdgesMap[Val].push_back(Edge);
    auto InValEdges = InValEdgesMap.takeVector();

    if (InValEdges.size() == 1) {
      PhiR->replaceAllUsesWith(InValEdges[0].first);
      PhiR->eraseFromParent();
      continue;
    }

    // Sort the incoming value order to match PhiR as much as possible.
    llvm::stable_sort(InValEdges, [&PhiR](auto &L, auto &R) {
      auto InVs = PhiR->incoming_values();
      return std::distance(InVs.begin(), find(InVs, L.first)) <
             std::distance(InVs.begin(), find(InVs, R.first));
    });

    SmallVector<VPValue *, 2> OperandsWithMask;
    for (const auto &[InVPV, Edges] : InValEdges) {
      if (match(InVPV, m_Poison()))
        continue;
      OperandsWithMask.push_back(InVPV);
      OperandsWithMask.push_back(createMaskDisjunction(Edges, VPBB));
    }
    PHINode *IRPhi = cast_or_null<PHINode>(PhiR->getUnderlyingValue());
    auto *Blend =
        new VPBlendRecipe(IRPhi, OperandsWithMask, *PhiR, PhiR->getDebugLoc());
    Builder.insert(Blend);
    PhiR->replaceAllUsesWith(Blend);
    PhiR->eraseFromParent();
  }
}

void VPlanTransforms::introduceMasksAndLinearize(VPlan &Plan) {
  // Nested loop regions (outer-loop vectorization) are not supported yet.
  if (Plan.isOuterLoop())
    return;
  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  // Scan the body of the loop in a topological order to visit each basic block
  // after having visited its predecessor basic blocks.
  VPBasicBlock *Header = LoopRegion->getEntryBasicBlock();
  ReversePostOrderTraversal<VPBlockShallowTraversalWrapper<VPBlockBase *>> RPOT(
      Header);
  VPPredicator Predicator(Plan);
  for (VPBlockBase *VPB : RPOT) {
    // Non-outer regions with VPBBs only are supported at the moment.
    auto *VPBB = cast<VPBasicBlock>(VPB);
    // Introduce the mask for VPBB, which may introduce needed edge masks, and
    // convert all phi recipes of VPBB to blend recipes unless VPBB is the
    // header.
    if (VPBB != Header)
      Predicator.createBlockInMask(VPBB);

    VPValue *BlockMask = Predicator.getBlockInMask(VPBB);
    if (!BlockMask)
      continue;

    // Mask all VPInstructions in the block.
    for (VPRecipeBase &R : *VPBB) {
      if (auto *VPI = dyn_cast<VPInstruction>(&R))
        VPI->addMask(BlockMask);
    }
  }

  for (VPBlockBase *VPBB : reverse(RPOT))
    if (VPBB != Header)
      Predicator.convertPhisToBlends(cast<VPBasicBlock>(VPBB));

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
}

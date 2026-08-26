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

#define DEBUG_TYPE "vplan-predicator"

using namespace llvm;
using namespace VPlanPatternMatch;

namespace {
class CompactRPOT {
  SmallVector<VPBlockBase *> Blocks;
  DenseMap<const VPBlockBase *, unsigned> BlockIndex;

  template <typename rpo_iterator>
  void scheduleDomRegion(VPBlockBase *VPBB, rpo_iterator It, rpo_iterator End,
                         unsigned &NextIndex, const VPDominatorTree &VPDT) {
    BlockIndex[VPBB] = NextIndex++;
    for (; It != End; ++It) {
      auto *DTNode = VPDT.getNode(*It);
      if (DTNode->getIDom()->getBlock() != VPBB)
        continue;
      scheduleDomRegion(*It, It, End, NextIndex, VPDT);
    }
  }

public:
  CompactRPOT(VPBasicBlock *Header, const VPDominatorTree &VPDT) {
    copy(post_order(VPBlockShallowTraversalWrapper<VPBlockBase *>{Header}),
         std::back_inserter(Blocks));
    unsigned Index = 0;
    // Blocks are in post-order (not reversed), so use reverse iterators while
    // compacting.
    scheduleDomRegion(*Blocks.rbegin(), Blocks.rbegin(), Blocks.rend(), Index,
                      VPDT);
    sort(Blocks, [&](VPBlockBase *A, VPBlockBase *B) {
      return BlockIndex[A] < BlockIndex[B];
    });

    LLVM_DEBUG({
      dbgs() << "Compact RPOT: ";
      for (VPBlockBase *VPBB : Blocks) {
        dbgs() << " " << VPBB->getName();
      }
      dbgs() << "\n";
    });
  }

  auto begin() const { return Blocks.begin(); }
  auto end() const { return Blocks.end(); }
  unsigned size() const { return Blocks.size(); }
  unsigned getIndex(const VPBlockBase *BB) const { return BlockIndex.at(BB); }
};

class VPPredicator {
  VPlan &Plan;

  /// Builder to construct recipes to compute masks.
  VPBuilder Builder;

  /// Dominator tree for the VPlan.
  VPDominatorTree VPDT;

  /// Post-dominator tree for the VPlan.
  VPPostDominatorTree VPPDT;

  // Scan the body of the loop in a topological order to visit each basic
  // block after having visited its predecessor basic blocks.
  CompactRPOT BlocksInCompactRPOTOrder;

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

  /// Create a logical-and of a source mask and edge condition, keeping the
  /// header mask outermost when present. E.g. createMaskAnd("H && M", "C")
  /// would result in "H && (M && C)". The edge condition must not contain the
  /// header mask.
  VPValue *createMaskAnd(VPValue *SrcMask, VPValue *EdgeCond, DebugLoc DL);

  /// Create a logical-or, factoring out a common header mask if present.
  VPValue *createMaskOr(VPValue *LHS, VPValue *RHS, DebugLoc DL);

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

  using BlendTermTy = std::pair<VPValue *, VPBasicBlock *>;

  /// Pre-linearization blend terms, indexed by block's index in
  /// BlocksInCompactRPOTOrder and phi.
  SmallVector<DenseMap<VPPhi *, SmallVector<BlendTermTy>>> BlendTerms;

  /// Return true if every path starting at \p Root reaches one of \p Blocks.
  /// All blocks in \p Blocks are expected to be dominated by \p Root.
  bool
  isJointlyPostDominated(const VPBasicBlock *Root,
                         const SmallPtrSetImpl<VPBasicBlock *> &Blocks) const;

  /// Compute an ordered sequence of incoming values and the blocks whose
  /// in-masks select them. Consecutive terms with the same value are combined
  /// when their masks can be represented by a common dominator's in-mask.
  SmallVector<BlendTermTy> computeBlendTerms(VPPhi *Phi) const;

public:
  VPPredicator(VPlan &Plan)
      : Plan(Plan), VPDT(Plan), VPPDT(Plan),
        BlocksInCompactRPOTOrder(
            Plan.getVectorLoopRegion()->getEntryBasicBlock(), VPDT),
        BlendTerms(BlocksInCompactRPOTOrder.size()) {}

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

  /// Predicate and linearize the plan.
  void run();
};
} // namespace

VPValue *VPPredicator::createMaskAnd(VPValue *SrcMask, VPValue *EdgeCond,
                                     DebugLoc DL) {
  VPValue *HeaderMask = Plan.getVectorLoopRegion()->getHeaderMask();
  VPValue *Remainder = nullptr;
  if (!HeaderMask || !match(SrcMask, m_RemoveMask(HeaderMask, Remainder)))
    return Builder.createLogicalAnd(SrcMask, EdgeCond, DL);

  [[maybe_unused]] VPValue *EdgeRemainder = nullptr;
  assert(!match(EdgeCond, m_RemoveMask(HeaderMask, EdgeRemainder)) &&
         "Edge condition must not contain the header mask");

  if (!Remainder)
    return Builder.createLogicalAnd(HeaderMask, EdgeCond, DL);
  return Builder.createLogicalAnd(
      HeaderMask, Builder.createLogicalAnd(Remainder, EdgeCond, DL), DL);
}

VPValue *VPPredicator::createMaskOr(VPValue *LHS, VPValue *RHS, DebugLoc DL) {
  VPValue *HeaderMask = Plan.getVectorLoopRegion()->getHeaderMask();
  VPValue *LHSRemainder = nullptr;
  VPValue *RHSRemainder = nullptr;
  if (!HeaderMask || !match(LHS, m_RemoveMask(HeaderMask, LHSRemainder)) ||
      !match(RHS, m_RemoveMask(HeaderMask, RHSRemainder)))
    return Builder.createOr(LHS, RHS, DL);

  if (!LHSRemainder || !RHSRemainder)
    return HeaderMask;
  return Builder.createLogicalAnd(
      HeaderMask, Builder.createOr(LHSRemainder, RHSRemainder, DL), DL);
}

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
    EdgeMask = createMaskAnd(SrcMask, EdgeMask, Term->getDebugLoc());
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

    BlockMask = createMaskOr(BlockMask, EdgeMask, {});
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
      Mask = createMaskAnd(SrcMask, Mask, {});
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
      DefaultMask = createMaskAnd(SrcMask, DefaultMask, {});
  } else {
    // There are no destinations other than the default destination, so this is
    // an unconditional branch.
    DefaultMask = SrcMask;
  }
  setEdgeMask(Src, DefaultDst, DefaultMask);
}

bool VPPredicator::isJointlyPostDominated(
    const VPBasicBlock *Root,
    const SmallPtrSetImpl<VPBasicBlock *> &Blocks) const {
  assert(
      all_of(Blocks,
             [&](VPBasicBlock *VPBB) { return VPDT.dominates(Root, VPBB); }) &&
      "Root must dominate all blocks");

  SmallPtrSet<const VPBasicBlock *, 16> Visited;
  SmallVector<const VPBasicBlock *> Worklist(1, Root);
  while (!Worklist.empty()) {
    const VPBasicBlock *VPBB = Worklist.pop_back_val();
    if (!Visited.insert(VPBB).second || Blocks.contains(VPBB))
      continue;
    if (VPBB->getNumSuccessors() == 0)
      return false;
    for (const VPBlockBase *Succ : VPBB->getSuccessors())
      Worklist.push_back(cast<VPBasicBlock>(Succ));
  }
  return true;
}

SmallVector<VPPredicator::BlendTermTy>
VPPredicator::computeBlendTerms(VPPhi *Phi) const {
  SmallVector<BlendTermTy> Terms;
  for (auto [V, VPBB] : Phi->incoming_values_and_blocks())
    Terms.emplace_back(V, const_cast<VPBasicBlock *>(cast<VPBasicBlock>(VPBB)));

  sort(Terms, [this](const BlendTermTy &L, const BlendTermTy &R) {
    return BlocksInCompactRPOTOrder.getIndex(L.second) <
           BlocksInCompactRPOTOrder.getIndex(R.second);
  });
  assert(all_of(zip(Terms, drop_begin(Terms)),
                [](const auto &Pair) {
                  const auto &[L, R] = Pair;
                  return L.second != R.second || L.first == R.first;
                }) &&
         "Different values provided by the same block");

  // If a group of consecutive terms have the same value, and the blocks' common
  // dominator is jointly post-dominated by those blocks, replace the entire
  // group of these terms with a sinle term using that common dominator's mask.
  SmallVector<BlendTermTy> Combined;
  for (ArrayRef<BlendTermTy> RemainingTerms = Terms; !RemainingTerms.empty();) {
    ArrayRef<BlendTermTy> ConsequtiveTermsUsingSameValue =
        RemainingTerms.take_while([&](const BlendTermTy &Term) {
          return Term.first == RemainingTerms.front().first;
        });
    SmallPtrSet<VPBasicBlock *, 8> Blocks(
        from_range, make_second_range(ConsequtiveTermsUsingSameValue));
    auto *CommonDom = cast<VPBasicBlock>(
        VPDT.findNearestCommonDominator(iterator_range(Blocks)));

    if (isJointlyPostDominated(CommonDom, Blocks)) {
      Combined.emplace_back(ConsequtiveTermsUsingSameValue.front().first,
                            CommonDom);
    } else {
      Combined.append(ConsequtiveTermsUsingSameValue.begin(),
                      ConsequtiveTermsUsingSameValue.end());
    }
    RemainingTerms =
        RemainingTerms.drop_front(ConsequtiveTermsUsingSameValue.size());
  }
  return Combined;
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

    unsigned BlockIdx = BlocksInCompactRPOTOrder.getIndex(VPBB);
    const auto &Terms = BlendTerms[BlockIdx].at(PhiR);

    // The in-mask of the common dominator is true on all paths from an
    // incoming block to the phi. The dominator tree still represents the
    // pre-linearization CFG.
    VPBasicBlock *CommonIncomingDom = cast<VPBasicBlock>(
        VPDT.findNearestCommonDominator(make_second_range(Terms)));
    VPValue *CommonIncomingMask = getBlockInMask(CommonIncomingDom);

    SmallVector<VPValue *, 2> OperandsWithMask;
    for (auto [V, MaskBlock] : Terms) {
      VPValue *Mask = getBlockInMask(MaskBlock);
      VPValue *RemainingMask = nullptr;
      bool RemovedCommonMask =
          CommonIncomingMask && Mask &&
          match(Mask, m_RemoveMask(CommonIncomingMask, RemainingMask));
      VPValue *BlendMask = RemovedCommonMask ? RemainingMask : Mask;
      OperandsWithMask.append({V, BlendMask ? BlendMask : Plan.getTrue()});
    }

    PHINode *IRPhi = cast_or_null<PHINode>(PhiR->getUnderlyingValue());
    auto *Blend =
        new VPBlendRecipe(IRPhi, OperandsWithMask, *PhiR, PhiR->getDebugLoc());
    Builder.insert(Blend);
    PhiR->replaceAllUsesWith(Blend);
    PhiR->eraseFromParent();
  }
}

void VPPredicator::run() {
  VPBasicBlock *Header = Plan.getVectorLoopRegion()->getEntryBasicBlock();
  for (VPBlockBase *VPB : BlocksInCompactRPOTOrder) {
    // Non-outer regions with VPBBs only are supported at the moment.
    auto *VPBB = cast<VPBasicBlock>(VPB);
    // Introduce the mask for VPBB, which may introduce needed edge masks.
    if (VPBB != Header)
      createBlockInMask(VPBB);

    VPValue *BlockMask = getBlockInMask(VPBB);
    if (!BlockMask)
      continue;

    // Mask all VPInstructions in the block.
    for (VPRecipeBase &R : *VPBB) {
      if (auto *VPI = dyn_cast<VPInstruction>(&R))
        VPI->addMask(BlockMask);
    }
  }

  // Cache blend terms before linearization. Computing them requires the
  // original phi predecessor mappings and CFG successor relation, both of
  // which are rewritten below. Skip the header which is the first block.
  for (VPBasicBlock *VPBB :
       drop_begin(VPBlockUtils::blocksOnly<VPBasicBlock>(
           BlocksInCompactRPOTOrder))) {
    for (VPRecipeBase &R : VPBB->phis()) {
      auto *PhiR = cast<VPPhi>(&R);
      unsigned BlockIdx = BlocksInCompactRPOTOrder.getIndex(VPBB);
      BlendTerms[BlockIdx][PhiR] = computeBlendTerms(PhiR);
    }
  }

  // Linearize the blocks of the loop into one serial chain.
  VPBlockBase *PrevVPBB = nullptr;
  for (VPBasicBlock *VPBB :
       VPBlockUtils::blocksOnly<VPBasicBlock>(BlocksInCompactRPOTOrder)) {
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

  for (VPBlockBase *VPBB : reverse(BlocksInCompactRPOTOrder))
    if (VPBB != Header)
      convertPhisToBlends(cast<VPBasicBlock>(VPBB));
}

void VPlanTransforms::introduceMasksAndLinearize(VPlan &Plan) {
  // Nested loop regions (outer-loop vectorization) are not supported yet.
  if (Plan.isOuterLoop())
    return;
  VPPredicator(Plan).run();
}

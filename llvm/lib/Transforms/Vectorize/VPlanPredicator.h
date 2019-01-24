//===-- VPlanPredicator.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the VPlanPredicator class which contains the public
/// interfaces to predicate and linearize the VPlan region.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_VPLAN_PREDICATOR_H
#define LLVM_TRANSFORMS_VECTORIZE_VPLAN_PREDICATOR_H

#include "LoopVectorizationPlanner.h"
#include "VPlan.h"
#include "VPlanDominatorTree.h"

namespace llvm {

class VPlanPredicator {
private:
  enum class EdgeType {
    TRUE_EDGE,
    FALSE_EDGE,
  };

  // VPlan being predicated.
  VPlan &Plan;

  // VPLoopInfo for Plan's HCFG.
  VPLoopInfo *VPLI;

  // Dominator tree for Plan's HCFG.
  VPDominatorTree VPDomTree;

  // VPlan builder used to generate VPInstructions for block predicates.
  VPBuilder Builder;

  /// Get the type of edge from \p FromBlock to \p ToBlock. Returns TRUE_EDGE if
  /// \p ToBlock is either the unconditional successor or the conditional true
  /// successor of \p FromBlock and FALSE_EDGE otherwise.
  EdgeType getEdgeTypeBetween(VPBlockBase *FromBlock, VPBlockBase *ToBlock);

  /// Create and return VPValue corresponding to the predicate for the edge from
  /// \p PredBB to \p CurrentBlock.
  VPValue *getOrCreateNotPredicate(VPBasicBlock *PredBB, VPBasicBlock *CurrBB);

  /// Generate and return the result of ORing all the predicate VPValues in \p
  /// Worklist.
  VPValue *genPredicateTree(std::list<VPValue *> &Worklist);

  /// Create or propagate predicate for \p CurrBlock in region \p Region using
  /// predicate(s) of its predecessor(s)
  void createOrPropagatePredicates(VPBlockBase *CurrBlock,
                                   VPRegionBlock *Region);

  /// Predicate the CFG within \p Region.
  void predicateRegionRec(VPRegionBlock *Region);

  /// Linearize the CFG within \p Region.
  void linearizeRegionRec(VPRegionBlock *Region);

public:
  VPlanPredicator(VPlan &Plan);

  /// Predicate Plan's HCFG.
  void predicate(void);
};
} // end namespace llvm
#endif // LLVM_TRANSFORMS_VECTORIZE_VPLAN_PREDICATOR_H

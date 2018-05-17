//===-- VPlanHCFGBuilder.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the VPlanHCFGBuilder class which contains the public
/// interface (buildHierarchicalCFG) to build a VPlan-based Hierarchical CFG
/// (H-CFG) for an incoming IR.
///
/// A H-CFG in VPlan is a control-flow graph whose nodes are VPBasicBlocks
/// and/or VPRegionBlocks (i.e., other H-CFGs). The outermost H-CFG of a VPlan
/// consists of a VPRegionBlock, denoted Top Region, which encloses any other
/// VPBlockBase in the H-CFG. This guarantees that any VPBlockBase in the H-CFG
/// other than the Top Region will have a parent VPRegionBlock and allows us
/// to easily add more nodes before/after the main vector loop (such as the
/// reduction epilogue).
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_VPLAN_VPLANHCFGBUILDER_H
#define LLVM_TRANSFORMS_VECTORIZE_VPLAN_VPLANHCFGBUILDER_H

#include "VPlan.h"
#include "VPlanVerifier.h"

namespace llvm {

class Loop;

/// Main class to build the VPlan H-CFG for an incoming IR.
class VPlanHCFGBuilder {
private:
  // The outermost loop of the input loop nest considered for vectorization.
  Loop *TheLoop;

  // Loop Info analysis.
  LoopInfo *LI;

  // VPlan verifier utility.
  VPlanVerifier Verifier;

public:
  VPlanHCFGBuilder(Loop *Lp, LoopInfo *LI) : TheLoop(Lp), LI(LI) {}

  /// Build H-CFG for TheLoop and update \p Plan accordingly.
  void buildHierarchicalCFG(VPlan &Plan);
};
} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_VPLAN_VPLANHCFGBUILDER_H

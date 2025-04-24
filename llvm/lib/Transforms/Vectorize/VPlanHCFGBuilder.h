//===-- VPlanHCFGBuilder.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#include "llvm/ADT/DenseMap.h"

namespace llvm {

class Loop;
class LoopInfo;
class VPlan;
class VPlanTestIRBase;
class VPBlockBase;
class BasicBlock;

/// Main class to build the VPlan H-CFG for an incoming IR.
class VPlanHCFGBuilder {
  friend VPlanTestIRBase;

private:
  // The outermost loop of the input loop nest considered for vectorization.
  Loop *TheLoop;

  // Loop Info analysis.
  LoopInfo *LI;

  // The VPlan that will contain the H-CFG we are building.
  VPlan &Plan;

  /// Map of create VP blocks to their input IR basic blocks, if they have been
  /// created for a input IR basic block.
  DenseMap<VPBlockBase *, BasicBlock *> VPB2IRBB;

public:
  VPlanHCFGBuilder(Loop *Lp, LoopInfo *LI, VPlan &P)
      : TheLoop(Lp), LI(LI), Plan(P) {}

  /// Build plain CFG for TheLoop and connects it to Plan's entry.
  void buildPlainCFG();

  /// Return the input IR BasicBlock corresponding to \p VPB. Returns nullptr if
  /// there is no such corresponding block.
  /// FIXME: This is a temporary workaround to drive the createBlockInMask.
  /// Remove once mask creation is done on VPlan.
  BasicBlock *getIRBBForVPB(const VPBlockBase *VPB) const {
    return VPB2IRBB.lookup(VPB);
  }
};
} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_VPLAN_VPLANHCFGBUILDER_H

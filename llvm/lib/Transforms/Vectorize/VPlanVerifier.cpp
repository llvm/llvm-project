//===-- VPlanVerifier.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the class VPlanVerifier, which contains utility functions
/// to check the consistency and invariants of a VPlan.
///
//===----------------------------------------------------------------------===//

#include "VPlanVerifier.h"
#include "VPlan.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "loop-vectorize"

using namespace llvm;

static cl::opt<bool> EnableHCFGVerifier("vplan-verify-hcfg", cl::init(false),
                                        cl::Hidden,
                                        cl::desc("Verify VPlan H-CFG."));

#ifndef NDEBUG
/// Utility function that checks whether \p VPBlockVec has duplicate
/// VPBlockBases.
static bool hasDuplicates(const SmallVectorImpl<VPBlockBase *> &VPBlockVec) {
  SmallDenseSet<const VPBlockBase *, 8> VPBlockSet;
  for (const auto *Block : VPBlockVec) {
    if (VPBlockSet.count(Block))
      return true;
    VPBlockSet.insert(Block);
  }
  return false;
}
#endif

/// Helper function that verifies the CFG invariants of the VPBlockBases within
/// \p Region. Checks in this function are generic for VPBlockBases. They are
/// not specific for VPBasicBlocks or VPRegionBlocks.
static void verifyBlocksInRegion(const VPRegionBlock *Region) {
  for (const VPBlockBase *VPB :
       make_range(df_iterator<const VPBlockBase *>::begin(Region->getEntry()),
                  df_iterator<const VPBlockBase *>::end(Region->getExit()))) {
    // Check block's parent.
    assert(VPB->getParent() == Region && "VPBlockBase has wrong parent");

    // Check block's condition bit.
    if (VPB->getNumSuccessors() > 1)
      assert(VPB->getCondBit() && "Missing condition bit!");
    else
      assert(!VPB->getCondBit() && "Unexpected condition bit!");

    // Check block's successors.
    const auto &Successors = VPB->getSuccessors();
    // There must be only one instance of a successor in block's successor list.
    // TODO: This won't work for switch statements.
    assert(!hasDuplicates(Successors) &&
           "Multiple instances of the same successor.");

    for (const VPBlockBase *Succ : Successors) {
      // There must be a bi-directional link between block and successor.
      const auto &SuccPreds = Succ->getPredecessors();
      assert(llvm::is_contained(SuccPreds, VPB) && "Missing predecessor link.");
      (void)SuccPreds;
    }

    // Check block's predecessors.
    const auto &Predecessors = VPB->getPredecessors();
    // There must be only one instance of a predecessor in block's predecessor
    // list.
    // TODO: This won't work for switch statements.
    assert(!hasDuplicates(Predecessors) &&
           "Multiple instances of the same predecessor.");

    for (const VPBlockBase *Pred : Predecessors) {
      // Block and predecessor must be inside the same region.
      assert(Pred->getParent() == VPB->getParent() &&
             "Predecessor is not in the same region.");

      // There must be a bi-directional link between block and predecessor.
      const auto &PredSuccs = Pred->getSuccessors();
      assert(llvm::is_contained(PredSuccs, VPB) && "Missing successor link.");
      (void)PredSuccs;
    }
  }
}

/// Verify the CFG invariants of VPRegionBlock \p Region and its nested
/// VPBlockBases. Do not recurse inside nested VPRegionBlocks.
static void verifyRegion(const VPRegionBlock *Region) {
  const VPBlockBase *Entry = Region->getEntry();
  const VPBlockBase *Exit = Region->getExit();

  // Entry and Exit shouldn't have any predecessor/successor, respectively.
  assert(!Entry->getNumPredecessors() && "Region entry has predecessors.");
  assert(!Exit->getNumSuccessors() && "Region exit has successors.");
  (void)Entry;
  (void)Exit;

  verifyBlocksInRegion(Region);
}

/// Verify the CFG invariants of VPRegionBlock \p Region and its nested
/// VPBlockBases. Recurse inside nested VPRegionBlocks.
static void verifyRegionRec(const VPRegionBlock *Region) {
  verifyRegion(Region);

  // Recurse inside nested regions.
  for (const VPBlockBase *VPB :
       make_range(df_iterator<const VPBlockBase *>::begin(Region->getEntry()),
                  df_iterator<const VPBlockBase *>::end(Region->getExit()))) {
    if (const auto *SubRegion = dyn_cast<VPRegionBlock>(VPB))
      verifyRegionRec(SubRegion);
  }
}

void VPlanVerifier::verifyHierarchicalCFG(
    const VPRegionBlock *TopRegion) const {
  if (!EnableHCFGVerifier)
    return;

  LLVM_DEBUG(dbgs() << "Verifying VPlan H-CFG.\n");
  assert(!TopRegion->getParent() && "VPlan Top Region should have no parent.");
  verifyRegionRec(TopRegion);
}

bool VPlanVerifier::verifyPlanIsValid(const VPlan &Plan) {
  auto Iter = depth_first(
      VPBlockRecursiveTraversalWrapper<const VPBlockBase *>(Plan.getEntry()));
  for (const VPBasicBlock *VPBB :
       VPBlockUtils::blocksOnly<const VPBasicBlock>(Iter)) {
    // Verify that phi-like recipes are at the beginning of the block, with no
    // other recipes in between.
    auto RecipeI = VPBB->begin();
    auto End = VPBB->end();
    while (RecipeI != End && RecipeI->isPhi())
      RecipeI++;

    while (RecipeI != End) {
      if (RecipeI->isPhi() && !isa<VPBlendRecipe>(&*RecipeI)) {
        errs() << "Found phi-like recipe after non-phi recipe";

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
        errs() << ": ";
        RecipeI->dump();
        errs() << "after\n";
        std::prev(RecipeI)->dump();
#endif
        return false;
      }
      RecipeI++;
    }
  }

  const VPRegionBlock *TopRegion = Plan.getVectorLoopRegion();
  const VPBasicBlock *Entry = dyn_cast<VPBasicBlock>(TopRegion->getEntry());
  if (!Entry) {
    errs() << "VPlan entry block is not a VPBasicBlock\n";
    return false;
  }

  if (!isa<VPCanonicalIVPHIRecipe>(&*Entry->begin())) {
    errs() << "VPlan vector loop header does not start with a "
              "VPCanonicalIVPHIRecipe\n";
    return false;
  }

  const VPBasicBlock *Exit = dyn_cast<VPBasicBlock>(TopRegion->getExit());
  if (!Exit) {
    errs() << "VPlan exit block is not a VPBasicBlock\n";
    return false;
  }

  if (Exit->empty()) {
    errs() << "VPlan vector loop exit must end with BranchOnCount "
              "VPInstruction but is empty\n";
    return false;
  }

  auto *LastInst = dyn_cast<VPInstruction>(std::prev(Exit->end()));
  if (!LastInst || LastInst->getOpcode() != VPInstruction::BranchOnCount) {
    errs() << "VPlan vector loop exit must end with BranchOnCount "
              "VPInstruction\n";
    return false;
  }

  for (const VPRegionBlock *Region :
       VPBlockUtils::blocksOnly<const VPRegionBlock>(
           depth_first(VPBlockRecursiveTraversalWrapper<const VPBlockBase *>(
               Plan.getEntry())))) {
    if (Region->getEntry()->getNumPredecessors() != 0) {
      errs() << "region entry block has predecessors\n";
      return false;
    }
    if (Region->getExit()->getNumSuccessors() != 0) {
      errs() << "region exit block has successors\n";
      return false;
    }
  }
  return true;
}

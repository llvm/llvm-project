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
#include "VPlanCFG.h"
#include "VPlanDominatorTree.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "loop-vectorize"

using namespace llvm;

// Verify that phi-like recipes are at the beginning of \p VPBB, with no
// other recipes in between. Also check that only header blocks contain
// VPHeaderPHIRecipes.
static bool verifyPhiRecipes(const VPBasicBlock *VPBB) {
  auto RecipeI = VPBB->begin();
  auto End = VPBB->end();
  unsigned NumActiveLaneMaskPhiRecipes = 0;
  const VPRegionBlock *ParentR = VPBB->getParent();
  bool IsHeaderVPBB = ParentR && !ParentR->isReplicator() &&
                      ParentR->getEntryBasicBlock() == VPBB;
  while (RecipeI != End && RecipeI->isPhi()) {
    if (isa<VPActiveLaneMaskPHIRecipe>(RecipeI))
      NumActiveLaneMaskPhiRecipes++;

    if (IsHeaderVPBB && !isa<VPHeaderPHIRecipe, VPWidenPHIRecipe>(*RecipeI)) {
      errs() << "Found non-header PHI recipe in header VPBB";
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
      errs() << ": ";
      RecipeI->dump();
#endif
      return false;
    }

    if (!IsHeaderVPBB && isa<VPHeaderPHIRecipe>(*RecipeI)) {
      errs() << "Found header PHI recipe in non-header VPBB";
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
      errs() << ": ";
      RecipeI->dump();
#endif
      return false;
    }

    RecipeI++;
  }

  if (NumActiveLaneMaskPhiRecipes > 1) {
    errs() << "There should be no more than one VPActiveLaneMaskPHIRecipe";
    return false;
  }

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
  return true;
}

static bool verifyVPBasicBlock(const VPBasicBlock *VPBB,
                               const VPDominatorTree &VPDT) {
  if (!verifyPhiRecipes(VPBB))
    return false;

  // Verify that defs in VPBB dominate all their uses. The current
  // implementation is still incomplete.
  DenseMap<const VPRecipeBase *, unsigned> RecipeNumbering;
  unsigned Cnt = 0;
  for (const VPRecipeBase &R : *VPBB)
    RecipeNumbering[&R] = Cnt++;

  // Set of recipe types along with VPInstruction Opcodes of all EVL-related
  // recipes that must appear at most once in the header block.
  DenseSet<unsigned> EVLFound;
  const VPRecipeBase *VPWidenMemRecipe = nullptr;
  const VPlan *Plan = VPBB->getPlan();
  bool IsHeader = Plan->getEntry()->getNumSuccessors() == 1 &&
                  Plan->getVectorLoopRegion()->getEntry() == VPBB;
  auto CheckEVLRecipiesInsts = [&](const VPRecipeBase *R) {
    if (isa<VPEVLBasedIVPHIRecipe>(R)) {
      if (!IsHeader) {
        errs() << "EVL PHI recipe not in entry block!\n";
        return false;
      }
      if (!EVLFound.insert(VPDef::VPEVLBasedIVPHISC).second) {
        errs() << "EVL PHI recipe inserted more than once!\n";
        return false;
      }
      return true;
    }
    if (const auto *RInst = dyn_cast<VPInstruction>(R);
        RInst && RInst->getOpcode() == VPInstruction::ExplicitVectorLength) {
      if (!IsHeader) {
        errs() << "EVL instruction not in the header block!\n";
        return false;
      }
      if (!EVLFound.insert(RInst->getOpcode() + VPDef::VPLastPHISC).second) {
        errs() << "EVL instruction inserted more than once!\n";
        return false;
      }
      if (VPWidenMemRecipe) {
        errs() << "Use of EVL instruction by widen memory recipe before "
                  "definition!\n";
        return false;
      }
      return true;
    }
    if (isa<VPWidenMemoryInstructionRecipe>(R))
      VPWidenMemRecipe = R;
    return true;
  };

  for (const VPRecipeBase &R : *VPBB) {
    if (!CheckEVLRecipiesInsts(&R))
      return false;
    for (const VPValue *V : R.definedValues()) {
      for (const VPUser *U : V->users()) {
        auto *UI = dyn_cast<VPRecipeBase>(U);
        // TODO: check dominance of incoming values for phis properly.
        if (!UI ||
            isa<VPHeaderPHIRecipe, VPWidenPHIRecipe, VPPredInstPHIRecipe>(UI))
          continue;

        // If the user is in the same block, check it comes after R in the
        // block.
        if (UI->getParent() == VPBB) {
          if (RecipeNumbering[UI] < RecipeNumbering[&R]) {
            errs() << "Use before def!\n";
            return false;
          }
          continue;
        }

        if (!VPDT.dominates(VPBB, UI->getParent())) {
          errs() << "Use before def!\n";
          return false;
        }
      }
    }
  }
  return true;
}

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

static bool verifyBlock(const VPBlockBase *VPB, const VPDominatorTree &VPDT) {
  auto *VPBB = dyn_cast<VPBasicBlock>(VPB);
  // Check block's condition bit.
  if (VPB->getNumSuccessors() > 1 ||
      (VPBB && VPBB->getParent() && VPBB->isExiting() &&
       !VPBB->getParent()->isReplicator())) {
    if (!VPBB || !VPBB->getTerminator()) {
      errs() << "Block has multiple successors but doesn't "
                "have a proper branch recipe!\n";
      return false;
    }
  } else {
    if (VPBB && VPBB->getTerminator()) {
      errs() << "Unexpected branch recipe!\n";
      return false;
    }
  }

  // Check block's successors.
  const auto &Successors = VPB->getSuccessors();
  // There must be only one instance of a successor in block's successor list.
  // TODO: This won't work for switch statements.
  if (hasDuplicates(Successors)) {
    errs() << "Multiple instances of the same successor.\n";
    return false;
  }

  for (const VPBlockBase *Succ : Successors) {
    // There must be a bi-directional link between block and successor.
    const auto &SuccPreds = Succ->getPredecessors();
    if (!is_contained(SuccPreds, VPB)) {
      errs() << "Missing predecessor link.\n";
      return false;
    }
  }

  // Check block's predecessors.
  const auto &Predecessors = VPB->getPredecessors();
  // There must be only one instance of a predecessor in block's predecessor
  // list.
  // TODO: This won't work for switch statements.
  if (hasDuplicates(Predecessors)) {
    errs() << "Multiple instances of the same predecessor.\n";
    return false;
  }

  for (const VPBlockBase *Pred : Predecessors) {
    // Block and predecessor must be inside the same region.
    if (Pred->getParent() != VPB->getParent()) {
      errs() << "Predecessor is not in the same region.\n";
      return false;
    }

    // There must be a bi-directional link between block and predecessor.
    const auto &PredSuccs = Pred->getSuccessors();
    if (!is_contained(PredSuccs, VPB)) {
      errs() << "Missing successor link.\n";
      return false;
    }
  }
  return !VPBB || verifyVPBasicBlock(VPBB, VPDT);
}

/// Helper function that verifies the CFG invariants of the VPBlockBases within
/// \p Region. Checks in this function are generic for VPBlockBases. They are
/// not specific for VPBasicBlocks or VPRegionBlocks.
static bool verifyBlocksInRegion(const VPRegionBlock *Region,
                                 const VPDominatorTree &VPDT) {
  for (const VPBlockBase *VPB : vp_depth_first_shallow(Region->getEntry())) {
    // Check block's parent.
    if (VPB->getParent() != Region) {
      errs() << "VPBlockBase has wrong parent\n";
      return false;
    }

    if (!verifyBlock(VPB, VPDT))
      return false;
  }
  return true;
}

/// Verify the CFG invariants of VPRegionBlock \p Region and its nested
/// VPBlockBases. Do not recurse inside nested VPRegionBlocks.
static bool verifyRegion(const VPRegionBlock *Region,
                         const VPDominatorTree &VPDT) {
  const VPBlockBase *Entry = Region->getEntry();
  const VPBlockBase *Exiting = Region->getExiting();

  // Entry and Exiting shouldn't have any predecessor/successor, respectively.
  if (Entry->getNumPredecessors() != 0) {
    errs() << "region entry block has predecessors\n";
    return false;
  }
  if (Exiting->getNumSuccessors() != 0) {
    errs() << "region exiting block has successors\n";
    return false;
  }

  return verifyBlocksInRegion(Region, VPDT);
}

/// Verify the CFG invariants of VPRegionBlock \p Region and its nested
/// VPBlockBases. Recurse inside nested VPRegionBlocks.
static bool verifyRegionRec(const VPRegionBlock *Region,
                            const VPDominatorTree &VPDT) {
  // Recurse inside nested regions and check all blocks inside the region.
  return verifyRegion(Region, VPDT) &&
         all_of(vp_depth_first_shallow(Region->getEntry()),
                [&VPDT](const VPBlockBase *VPB) {
                  const auto *SubRegion = dyn_cast<VPRegionBlock>(VPB);
                  return !SubRegion || verifyRegionRec(SubRegion, VPDT);
                });
}

bool llvm::verifyVPlanIsValid(const VPlan &Plan) {
  VPDominatorTree VPDT;
  VPDT.recalculate(const_cast<VPlan &>(Plan));

  if (any_of(
          vp_depth_first_shallow(Plan.getEntry()),
          [&VPDT](const VPBlockBase *VPB) { return !verifyBlock(VPB, VPDT); }))
    return false;

  const VPRegionBlock *TopRegion = Plan.getVectorLoopRegion();
  if (!verifyRegionRec(TopRegion, VPDT))
    return false;

  if (TopRegion->getParent()) {
    errs() << "VPlan Top Region should have no parent.\n";
    return false;
  }

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

  const VPBasicBlock *Exiting = dyn_cast<VPBasicBlock>(TopRegion->getExiting());
  if (!Exiting) {
    errs() << "VPlan exiting block is not a VPBasicBlock\n";
    return false;
  }

  if (Exiting->empty()) {
    errs() << "VPlan vector loop exiting block must end with BranchOnCount or "
              "BranchOnCond VPInstruction but is empty\n";
    return false;
  }

  auto *LastInst = dyn_cast<VPInstruction>(std::prev(Exiting->end()));
  if (!LastInst || (LastInst->getOpcode() != VPInstruction::BranchOnCount &&
                    LastInst->getOpcode() != VPInstruction::BranchOnCond)) {
    errs() << "VPlan vector loop exit must end with BranchOnCount or "
              "BranchOnCond VPInstruction\n";
    return false;
  }

  for (const auto &KV : Plan.getLiveOuts())
    if (KV.second->getNumOperands() != 1) {
      errs() << "live outs must have a single operand\n";
      return false;
    }

  return true;
}

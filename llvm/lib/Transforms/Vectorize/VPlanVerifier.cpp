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
#include "VPlanHelpers.h"
#include "VPlanPatternMatch.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "loop-vectorize"

using namespace llvm;

namespace {
class VPlanVerifier {
  const VPDominatorTree &VPDT;
  VPTypeAnalysis &TypeInfo;
  bool VerifyLate;

  SmallPtrSet<BasicBlock *, 8> WrappedIRBBs;

  // Verify that phi-like recipes are at the beginning of \p VPBB, with no
  // other recipes in between. Also check that only header blocks contain
  // VPHeaderPHIRecipes.
  bool verifyPhiRecipes(const VPBasicBlock *VPBB);

  /// Verify that \p EVL is used correctly. The user must be either in
  /// EVL-based recipes as a last operand or VPInstruction::Add which is
  /// incoming value into EVL's recipe.
  bool verifyEVLRecipe(const VPInstruction &EVL) const;

  bool verifyVPBasicBlock(const VPBasicBlock *VPBB);

  bool verifyBlock(const VPBlockBase *VPB);

  /// Helper function that verifies the CFG invariants of the VPBlockBases
  /// within
  /// \p Region. Checks in this function are generic for VPBlockBases. They are
  /// not specific for VPBasicBlocks or VPRegionBlocks.
  bool verifyBlocksInRegion(const VPRegionBlock *Region);

  /// Verify the CFG invariants of VPRegionBlock \p Region and its nested
  /// VPBlockBases. Do not recurse inside nested VPRegionBlocks.
  bool verifyRegion(const VPRegionBlock *Region);

  /// Verify the CFG invariants of VPRegionBlock \p Region and its nested
  /// VPBlockBases. Recurse inside nested VPRegionBlocks.
  bool verifyRegionRec(const VPRegionBlock *Region);

public:
  VPlanVerifier(VPDominatorTree &VPDT, VPTypeAnalysis &TypeInfo,
                bool VerifyLate)
      : VPDT(VPDT), TypeInfo(TypeInfo), VerifyLate(VerifyLate) {}

  bool verify(const VPlan &Plan);
};
} // namespace

bool VPlanVerifier::verifyPhiRecipes(const VPBasicBlock *VPBB) {
  auto RecipeI = VPBB->begin();
  auto End = VPBB->end();
  unsigned NumActiveLaneMaskPhiRecipes = 0;
  bool IsHeaderVPBB = VPBlockUtils::isHeader(VPBB, VPDT);
  while (RecipeI != End && RecipeI->isPhi()) {
    if (isa<VPActiveLaneMaskPHIRecipe>(RecipeI))
      NumActiveLaneMaskPhiRecipes++;

    if (IsHeaderVPBB &&
        !isa<VPHeaderPHIRecipe, VPWidenPHIRecipe, VPPhi>(*RecipeI)) {
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

    // Check if the recipe operands match the number of predecessors.
    // TODO Extend to other phi-like recipes.
    if (auto *PhiIRI = dyn_cast<VPIRPhi>(&*RecipeI)) {
      if (PhiIRI->getNumOperands() != VPBB->getNumPredecessors()) {
        errs() << "Phi-like recipe with different number of operands and "
                  "predecessors.\n";
        // TODO: Print broken recipe. At the moment printing an ill-formed
        // phi-like recipe may crash.
        return false;
      }
    }

    RecipeI++;
  }

  if (!VerifyLate && NumActiveLaneMaskPhiRecipes > 1) {
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

bool VPlanVerifier::verifyEVLRecipe(const VPInstruction &EVL) const {
  if (EVL.getOpcode() != VPInstruction::ExplicitVectorLength) {
    errs() << "verifyEVLRecipe should only be called on "
              "VPInstruction::ExplicitVectorLength\n";
    return false;
  }
  auto VerifyEVLUse = [&](const VPRecipeBase &R,
                          const unsigned ExpectedIdx) -> bool {
    SmallVector<const VPValue *> Ops(R.operands());
    unsigned UseCount = count(Ops, &EVL);
    if (UseCount != 1 || Ops[ExpectedIdx] != &EVL) {
      errs() << "EVL is used as non-last operand in EVL-based recipe\n";
      return false;
    }
    return true;
  };
  return all_of(EVL.users(), [this, &VerifyEVLUse](VPUser *U) {
    return TypeSwitch<const VPUser *, bool>(U)
        .Case<VPWidenIntrinsicRecipe>([&](const VPWidenIntrinsicRecipe *S) {
          return VerifyEVLUse(*S, S->getNumOperands() - 1);
        })
        .Case<VPWidenStoreEVLRecipe, VPReductionEVLRecipe,
              VPWidenIntOrFpInductionRecipe, VPWidenPointerInductionRecipe>(
            [&](const VPRecipeBase *S) { return VerifyEVLUse(*S, 2); })
        .Case<VPScalarIVStepsRecipe>([&](auto *R) {
          if (R->getNumOperands() != 3) {
            errs() << "Unrolling with EVL tail folding not yet supported\n";
            return false;
          }
          return VerifyEVLUse(*R, 2);
        })
        .Case<VPWidenLoadEVLRecipe, VPVectorEndPointerRecipe>(
            [&](const VPRecipeBase *R) { return VerifyEVLUse(*R, 1); })
        .Case<VPInstructionWithType>(
            [&](const VPInstructionWithType *S) { return VerifyEVLUse(*S, 0); })
        .Case<VPInstruction>([&](const VPInstruction *I) {
          if (I->getOpcode() == Instruction::PHI ||
              I->getOpcode() == Instruction::ICmp ||
              I->getOpcode() == Instruction::Sub)
            return VerifyEVLUse(*I, 1);
          switch (I->getOpcode()) {
          case Instruction::Add:
            break;
          case Instruction::UIToFP:
          case Instruction::Trunc:
          case Instruction::ZExt:
          case Instruction::Mul:
          case Instruction::FMul:
            // Opcodes above can only use EVL after wide inductions have been
            // expanded.
            if (!VerifyLate) {
              errs() << "EVL used by unexpected VPInstruction\n";
              return false;
            }
            break;
          default:
            errs() << "EVL used by unexpected VPInstruction\n";
            return false;
          }
          // EVLIVIncrement is only used by EVLIV & BranchOnCount.
          // Having more than two users is unexpected.
          if ((I->getNumUsers() != 1) &&
              (I->getNumUsers() != 2 || none_of(I->users(), [&I](VPUser *U) {
                 using namespace llvm::VPlanPatternMatch;
                 return match(U, m_BranchOnCount(m_Specific(I), m_VPValue()));
               }))) {
            errs() << "EVL is used in VPInstruction with multiple users\n";
            return false;
          }
          if (!VerifyLate && !isa<VPEVLBasedIVPHIRecipe>(*I->users().begin())) {
            errs() << "Result of VPInstruction::Add with EVL operand is "
                      "not used by VPEVLBasedIVPHIRecipe\n";
            return false;
          }
          return true;
        })
        .Default([&](const VPUser *U) {
          errs() << "EVL has unexpected user\n";
          return false;
        });
  });
}

bool VPlanVerifier::verifyVPBasicBlock(const VPBasicBlock *VPBB) {
  if (!verifyPhiRecipes(VPBB))
    return false;

  // Verify that defs in VPBB dominate all their uses.
  DenseMap<const VPRecipeBase *, unsigned> RecipeNumbering;
  unsigned Cnt = 0;
  for (const VPRecipeBase &R : *VPBB)
    RecipeNumbering[&R] = Cnt++;

  for (const VPRecipeBase &R : *VPBB) {
    if (isa<VPIRInstruction>(&R) && !isa<VPIRBasicBlock>(VPBB)) {
      errs() << "VPIRInstructions ";
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
      R.dump();
      errs() << " ";
#endif
      errs() << "not in a VPIRBasicBlock!\n";
      return false;
    }
    for (const VPValue *V : R.definedValues()) {
      // Verify that we can infer a scalar type for each defined value. With
      // assertions enabled, inferScalarType will perform some consistency
      // checks during type inference.
      if (!TypeInfo.inferScalarType(V)) {
        errs() << "Failed to infer scalar type!\n";
        return false;
      }

      for (const VPUser *U : V->users()) {
        auto *UI = cast<VPRecipeBase>(U);
        if (auto *Phi = dyn_cast<VPPhiAccessors>(UI)) {
          for (unsigned Idx = 0; Idx != Phi->getNumIncoming(); ++Idx) {
            VPValue *IncomingVPV = Phi->getIncomingValue(Idx);
            if (IncomingVPV != V)
              continue;

            const VPBasicBlock *IncomingVPBB = Phi->getIncomingBlock(Idx);
            if (VPDT.dominates(VPBB, IncomingVPBB))
              continue;

            errs() << "Incoming def at index " << Idx
                   << " does not dominate incoming block!\n";
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
            VPSlotTracker Tracker(VPBB->getPlan());
            IncomingVPV->getDefiningRecipe()->print(errs(), "  ", Tracker);
            errs() << "\n  does not dominate " << IncomingVPBB->getName()
                   << " for\n";
            UI->print(errs(), "  ", Tracker);
#endif
            return false;
          }
          continue;
        }
        // TODO: Also verify VPPredInstPHIRecipe.
        if (isa<VPPredInstPHIRecipe>(UI))
          continue;

        // If the user is in the same block, check it comes after R in the
        // block.
        if (UI->getParent() == VPBB) {
          if (RecipeNumbering[UI] >= RecipeNumbering[&R])
            continue;
        } else {
          if (VPDT.dominates(VPBB, UI->getParent()))
            continue;
        }

        errs() << "Use before def!\n";
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
        VPSlotTracker Tracker(VPBB->getPlan());
        UI->print(errs(), "  ", Tracker);
        errs() << "\n  before\n";
        R.print(errs(), "  ", Tracker);
        errs() << "\n";
#endif
        return false;
      }
    }
    if (const auto *EVL = dyn_cast<VPInstruction>(&R)) {
      if (EVL->getOpcode() == VPInstruction::ExplicitVectorLength &&
          !verifyEVLRecipe(*EVL)) {
        errs() << "EVL VPValue is not used correctly\n";
        return false;
      }
    }
  }

  auto *IRBB = dyn_cast<VPIRBasicBlock>(VPBB);
  if (!IRBB)
    return true;

  if (!WrappedIRBBs.insert(IRBB->getIRBasicBlock()).second) {
    errs() << "Same IR basic block used by multiple wrapper blocks!\n";
    return false;
  }

  return true;
}

/// Utility function that checks whether \p VPBlockVec has duplicate
/// VPBlockBases.
static bool hasDuplicates(const SmallVectorImpl<VPBlockBase *> &VPBlockVec) {
  SmallDenseSet<const VPBlockBase *, 8> VPBlockSet;
  for (const auto *Block : VPBlockVec) {
    if (!VPBlockSet.insert(Block).second)
      return true;
  }
  return false;
}

bool VPlanVerifier::verifyBlock(const VPBlockBase *VPB) {
  auto *VPBB = dyn_cast<VPBasicBlock>(VPB);
  // Check block's condition bit.
  if (!isa<VPIRBasicBlock>(VPB)) {
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
  return !VPBB || verifyVPBasicBlock(VPBB);
}

bool VPlanVerifier::verifyBlocksInRegion(const VPRegionBlock *Region) {
  for (const VPBlockBase *VPB : vp_depth_first_shallow(Region->getEntry())) {
    // Check block's parent.
    if (VPB->getParent() != Region) {
      errs() << "VPBlockBase has wrong parent\n";
      return false;
    }

    if (!verifyBlock(VPB))
      return false;
  }
  return true;
}

bool VPlanVerifier::verifyRegion(const VPRegionBlock *Region) {
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

  return verifyBlocksInRegion(Region);
}

bool VPlanVerifier::verifyRegionRec(const VPRegionBlock *Region) {
  // Recurse inside nested regions and check all blocks inside the region.
  return verifyRegion(Region) &&
         all_of(vp_depth_first_shallow(Region->getEntry()),
                [this](const VPBlockBase *VPB) {
                  const auto *SubRegion = dyn_cast<VPRegionBlock>(VPB);
                  return !SubRegion || verifyRegionRec(SubRegion);
                });
}

bool VPlanVerifier::verify(const VPlan &Plan) {
  if (any_of(vp_depth_first_shallow(Plan.getEntry()),
             [this](const VPBlockBase *VPB) { return !verifyBlock(VPB); }))
    return false;

  const VPRegionBlock *TopRegion = Plan.getVectorLoopRegion();
  // TODO: Verify all blocks using vp_depth_first_deep iterators.
  if (!TopRegion)
    return true;

  if (!verifyRegionRec(TopRegion))
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

  return true;
}

bool llvm::verifyVPlanIsValid(const VPlan &Plan, bool VerifyLate) {
  VPDominatorTree VPDT;
  VPDT.recalculate(const_cast<VPlan &>(Plan));
  VPTypeAnalysis TypeInfo(Plan);
  VPlanVerifier Verifier(VPDT, TypeInfo, VerifyLate);
  return Verifier.verify(Plan);
}

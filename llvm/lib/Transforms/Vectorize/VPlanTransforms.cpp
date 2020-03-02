//===-- VPlanTransforms.cpp - Utility VPlan to VPlan transforms -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a set of utility VPlan to VPlan transformations.
///
//===----------------------------------------------------------------------===//

#include "VPlanTransforms.h"
#include "llvm/ADT/PostOrderIterator.h"

using namespace llvm;

void VPlanTransforms::VPInstructionsToVPRecipes(
    Loop *OrigLoop, VPlanPtr &Plan,
    LoopVectorizationLegality::InductionList &Inductions,
    SmallPtrSetImpl<Instruction *> &DeadInstructions) {

  auto *TopRegion = cast<VPRegionBlock>(Plan->getEntry());
  ReversePostOrderTraversal<VPBlockBase *> RPOT(TopRegion->getEntry());

  // Condition bit VPValues get deleted during transformation to VPRecipes.
  // Create new VPValues and save away as condition bits. These will be deleted
  // after finalizing the vector IR basic blocks.
  for (VPBlockBase *Base : RPOT) {
    VPBasicBlock *VPBB = Base->getEntryBasicBlock();
    if (auto *CondBit = VPBB->getCondBit()) {
      auto *NCondBit = new VPValue(CondBit->getUnderlyingValue());
      VPBB->setCondBit(NCondBit);
      Plan->addCBV(NCondBit);
    }
  }
  for (VPBlockBase *Base : RPOT) {
    // Do not widen instructions in pre-header and exit blocks.
    if (Base->getNumPredecessors() == 0 || Base->getNumSuccessors() == 0)
      continue;

    VPBasicBlock *VPBB = Base->getEntryBasicBlock();
    VPRecipeBase *LastRecipe = nullptr;
    // Introduce each ingredient into VPlan.
    for (auto I = VPBB->begin(), E = VPBB->end(); I != E;) {
      VPRecipeBase *Ingredient = &*I++;
      // Can only handle VPInstructions.
      VPInstruction *VPInst = cast<VPInstruction>(Ingredient);
      Instruction *Inst = cast<Instruction>(VPInst->getUnderlyingValue());
      if (DeadInstructions.count(Inst)) {
        Ingredient->eraseFromParent();
        continue;
      }

      VPRecipeBase *NewRecipe = nullptr;
      // Create VPWidenMemoryInstructionRecipe for loads and stores.
      if (isa<LoadInst>(Inst) || isa<StoreInst>(Inst))
        NewRecipe = new VPWidenMemoryInstructionRecipe(
            *Inst, Plan->getOrAddVPValue(getLoadStorePointerOperand(Inst)),
            nullptr /*Mask*/);
      else if (PHINode *Phi = dyn_cast<PHINode>(Inst)) {
        InductionDescriptor II = Inductions.lookup(Phi);
        if (II.getKind() == InductionDescriptor::IK_IntInduction ||
            II.getKind() == InductionDescriptor::IK_FpInduction) {
          NewRecipe = new VPWidenIntOrFpInductionRecipe(Phi);
        } else
          NewRecipe = new VPWidenPHIRecipe(Phi);
      } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Inst)) {
        NewRecipe = new VPWidenGEPRecipe(GEP, OrigLoop);
      } else {
        // If the last recipe is a VPWidenRecipe, add Inst to it instead of
        // creating a new recipe.
        if (VPWidenRecipe *WidenRecipe =
                dyn_cast_or_null<VPWidenRecipe>(LastRecipe)) {
          WidenRecipe->appendInstruction(Inst);
          Ingredient->eraseFromParent();
          continue;
        }
        NewRecipe = new VPWidenRecipe(Inst);
      }

      NewRecipe->insertBefore(Ingredient);
      LastRecipe = NewRecipe;
      Ingredient->eraseFromParent();
    }
  }
}

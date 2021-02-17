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

  for (VPBlockBase *Base : RPOT) {
    // Do not widen instructions in pre-header and exit blocks.
    if (Base->getNumPredecessors() == 0 || Base->getNumSuccessors() == 0)
      continue;

    VPBasicBlock *VPBB = Base->getEntryBasicBlock();
    // Introduce each ingredient into VPlan.
    for (auto I = VPBB->begin(), E = VPBB->end(); I != E;) {
      VPRecipeBase *Ingredient = &*I++;
      // Can only handle VPInstructions.
      VPInstruction *VPInst = cast<VPInstruction>(Ingredient);
      Instruction *Inst = cast<Instruction>(VPInst->getUnderlyingValue());
      if (DeadInstructions.count(Inst)) {
        VPValue DummyValue;
        VPInst->replaceAllUsesWith(&DummyValue);
        Ingredient->eraseFromParent();
        continue;
      }

      VPRecipeBase *NewRecipe = nullptr;
      // Create VPWidenMemoryInstructionRecipe for loads and stores.
      if (LoadInst *Load = dyn_cast<LoadInst>(Inst))
        NewRecipe = new VPWidenMemoryInstructionRecipe(
            *Load, Plan->getOrAddVPValue(getLoadStorePointerOperand(Inst)),
            nullptr /*Mask*/);
      else if (StoreInst *Store = dyn_cast<StoreInst>(Inst))
        NewRecipe = new VPWidenMemoryInstructionRecipe(
            *Store, Plan->getOrAddVPValue(getLoadStorePointerOperand(Inst)),
            Plan->getOrAddVPValue(Store->getValueOperand()), nullptr /*Mask*/);
      else if (PHINode *Phi = dyn_cast<PHINode>(Inst)) {
        InductionDescriptor II = Inductions.lookup(Phi);
        if (II.getKind() == InductionDescriptor::IK_IntInduction ||
            II.getKind() == InductionDescriptor::IK_FpInduction) {
          VPValue *Start = Plan->getOrAddVPValue(II.getStartValue());
          NewRecipe = new VPWidenIntOrFpInductionRecipe(Phi, Start, nullptr);
        } else
          NewRecipe = new VPWidenPHIRecipe(Phi);
      } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Inst)) {
        NewRecipe = new VPWidenGEPRecipe(
            GEP, Plan->mapToVPValues(GEP->operands()), OrigLoop);
      } else
        NewRecipe =
            new VPWidenRecipe(*Inst, Plan->mapToVPValues(Inst->operands()));

      NewRecipe->insertBefore(Ingredient);
      if (NewRecipe->getNumDefinedValues() == 1)
        VPInst->replaceAllUsesWith(NewRecipe->getVPValue());
      else
        assert(NewRecipe->getNumDefinedValues() == 0 &&
               "Only recpies with zero or one defined values expected");
      Ingredient->eraseFromParent();
      Plan->removeVPValueFor(Inst);
      for (auto *Def : NewRecipe->definedValues()) {
        Plan->addVPValue(Inst, Def);
      }
    }
  }
}

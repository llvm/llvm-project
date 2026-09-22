//===- SLPVPlanCodegen.cpp - VPlan-based codegen for SLP ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SLPVPlanCodegen.h"
#include "LoopVectorizationPlanner.h"
#include "VPlan.h"
#include "VPlanHelpers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

bool slpvectorizer::isLiveInOperand(unsigned Opcode, unsigned J) {
  // Loads and stores take their pointer operand as a live-in.
  return (Opcode == Instruction::Load &&
          J == LoadInst::getPointerOperandIndex()) ||
         (Opcode == Instruction::Store &&
          J == StoreInst::getPointerOperandIndex());
}

bool slpvectorizer::isSupportedVPlanCodegenOpcode(unsigned Opcode) {
  if (Instruction::isBinaryOp(Opcode))
    return true;
  switch (Opcode) {
  case Instruction::FNeg:
  case Instruction::Load:
  case Instruction::Store:
    return true;
  default:
    return false;
  }
}

/// Returns the IR flags common to all of \p Scalars, seeded from \p MainOp.
static VPIRFlags computeIntersectedFlags(Instruction *MainOp,
                                         ArrayRef<Value *> Scalars) {
  VPIRFlags Flags(*MainOp);
  for (Value *V : Scalars) {
    // Drop all flags if a lane is not an instruction, e.g. poison.
    auto *I = dyn_cast<Instruction>(V);
    if (!I)
      return VPIRFlags();
    Flags.intersectFlags(VPIRFlags(*I));
  }
  return Flags;
}

VPValue *slpvectorizer::createRecipeForBundle(VPlan &Plan, VPBuilder &VPB,
                                              Instruction *MainOp,
                                              ArrayRef<Value *> Scalars,
                                              ArrayRef<VPValue *> Ops) {
  VPIRFlags Flags = computeIntersectedFlags(MainOp, Scalars);
  // Only instructions carry metadata, so other lanes, e.g. poison, are skipped.
  VPIRMetadata Metadata(
      *MainOp, to_vector(make_filter_range(Scalars, IsaPred<Instruction>)));
  DebugLoc DL = MainOp->getDebugLoc();

  if (auto *LI = dyn_cast<LoadInst>(MainOp))
    return VPB.createWidenLoad(
        *LI, Plan.getOrAddLiveIn(LI->getPointerOperand()),
        /*Mask=*/nullptr, /*Consecutive=*/true, Metadata, DL);
  if (auto *SI = dyn_cast<StoreInst>(MainOp)) {
    VPB.createWidenStore(*SI, Plan.getOrAddLiveIn(SI->getPointerOperand()),
                         Ops[0], /*Mask=*/nullptr, /*Consecutive=*/true,
                         Metadata, DL);
    // Stores do not define a value.
    return nullptr;
  }
  return VPB.insert(new VPWidenRecipe(*MainOp, Ops, Flags, Metadata, DL));
}

void slpvectorizer::executeSLPPlan(VPlan &Plan, VPTransformState &State) {
  for (VPRecipeBase &R : *Plan.getEntry()) {
    State.Builder.SetCurrentDebugLocation(R.getDebugLoc());
    R.execute(State);
  }
}

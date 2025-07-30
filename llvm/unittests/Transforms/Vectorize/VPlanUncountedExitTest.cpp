//===- llvm/unittests/Transforms/Vectorize/VPlanUncountedExitTest.cpp -----===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lib/Transforms/Vectorize/LoopVectorizationPlanner.h"
#include "../lib/Transforms/Vectorize/VPlan.h"
#include "../lib/Transforms/Vectorize/VPlanPatternMatch.h"
#include "../lib/Transforms/Vectorize/VPlanUtils.h"
#include "VPlanTestBase.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "gtest/gtest.h"

namespace llvm {

namespace {
class VPUncountedExitTest : public VPlanTestBase {};

TEST_F(VPUncountedExitTest, FindUncountedExitRecipes) {
  // Create CFG skeleton.
  VPlan &Plan = getPlan();
  VPBasicBlock *ScalarPH = Plan.getEntry();
  VPBasicBlock *Entry = Plan.createVPBasicBlock("entry");
  Plan.setEntry(Entry);
  VPBasicBlock *VectorPH = Plan.createVPBasicBlock("vector.ph");
  VPBasicBlock *VecBody = Plan.createVPBasicBlock("vector.body");
  VPRegionBlock *Region =
      Plan.createVPRegionBlock(VecBody, VecBody, "vector loop");
  VPBasicBlock *MiddleBlock = Plan.createVPBasicBlock("middle.block");
  VPBlockUtils::connectBlocks(Entry, ScalarPH);
  VPBlockUtils::connectBlocks(Entry, VectorPH);
  VPBlockUtils::connectBlocks(VectorPH, Region);
  VPBlockUtils::connectBlocks(Region, MiddleBlock);
  VPBlockUtils::connectBlocks(MiddleBlock, ScalarPH);

  // Live-Ins
  IntegerType *I64Ty = IntegerType::get(C, 64);
  IntegerType *I32Ty = IntegerType::get(C, 32);
  PointerType *PTy = PointerType::get(C, 0);
  VPValue *Zero = Plan.getOrAddLiveIn(ConstantInt::get(I64Ty, 0));
  VPValue *Inc = Plan.getOrAddLiveIn(ConstantInt::get(I64Ty, 1));
  VPValue *VF = &Plan.getVF();
  Plan.setTripCount(Plan.getOrAddLiveIn(ConstantInt::get(I64Ty, 64)));

  // Populate vector.body with the recipes for exiting.
  auto *IV = new VPCanonicalIVPHIRecipe(Zero, {});
  VecBody->appendRecipe(IV);
  VPBuilder Builder(VecBody, VecBody->getFirstNonPhi());
  auto *Steps = Builder.createScalarIVSteps(Instruction::Add, nullptr, IV, Inc,
                                            VF, DebugLoc());

  // Uncounted Exit; GEP -> Load -> Cmp
  auto *DummyGEP = GetElementPtrInst::Create(I32Ty, Zero->getUnderlyingValue(),
                                             {}, Twine("ee.addr"));
  auto *GEP = new VPReplicateRecipe(DummyGEP, {Zero, Steps}, true, nullptr);
  Builder.insert(GEP);
  auto *DummyLoad =
      new LoadInst(I32Ty, PoisonValue::get(PTy), "ee.load", false, Align(1));
  VPValue *Load =
      new VPWidenLoadRecipe(*DummyLoad, GEP, nullptr, true, false, {}, {});
  Builder.insert(Load->getDefiningRecipe());
  // Should really splat the zero, but we're not checking types here.
  VPValue *Cmp = new VPWidenRecipe(Instruction::ICmp, {Load, Zero},
                                   VPIRFlags(CmpInst::ICMP_EQ), {}, {});
  Builder.insert(Cmp->getDefiningRecipe());
  VPValue *AnyOf = Builder.createNaryOp(VPInstruction::AnyOf, Cmp);

  // Counted Exit; Inc IV -> Cmp
  VPValue *NextIV = Builder.createNaryOp(Instruction::Add, {IV, VF});
  VPValue *Counted =
      Builder.createICmp(CmpInst::ICMP_EQ, NextIV, Plan.getTripCount());

  // Combine, and branch.
  VPValue *Combined = Builder.createNaryOp(Instruction::Or, {AnyOf, Counted});
  Builder.createNaryOp(VPInstruction::BranchOnCond, {Combined});

  SmallVector<VPRecipeBase *, 8> Recipes;
  SmallVector<VPReplicateRecipe *, 2> GEPs;

  std::optional<VPValue *> UncountedCondition =
      vputils::getRecipesForUncountedExit(Plan, Recipes, GEPs);
  ASSERT_TRUE(UncountedCondition.has_value());
  ASSERT_EQ(*UncountedCondition, Cmp);
  ASSERT_EQ(GEPs.size(), 1ull);
  ASSERT_EQ(GEPs[0], GEP);
  ASSERT_EQ(Recipes.size(), 3ull);

  delete DummyLoad;
  delete DummyGEP;
}

} // namespace
} // namespace llvm

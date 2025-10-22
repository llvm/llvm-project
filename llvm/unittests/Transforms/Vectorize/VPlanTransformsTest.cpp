//===- llvm/unittests/Transforms/Vectorize/VPlanTransformsTest.cpp --------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lib/Transforms/Vectorize/VPlanTransforms.h"
#include "../lib/Transforms/Vectorize/VPlan.h"
#include "VPlanTestBase.h"
#include "gtest/gtest.h"

namespace llvm {

namespace {
using VPlanTransformsTest = VPlanTestBase;

// Test we create and merge replicate regions:
// VPBB1:
//   REPLICATE %Rep0 = add
//   REPLICATE %Rep1 = add, %Mask
//   REPLICATE %Rep2 = add, %Mask
//   REPLICATE %Rep3 = add, %Mask
// No successors
//
// ->
//
// <xVFxUF> pred.add: {
//   pred.add.entry:
//     BRANCH-ON-MASK %Mask
//   Successor(s): pred.add.if, pred.add.continue
//
//   pred.add.if:
//     REPLICATE %Rep0 = add
//     REPLICATE %Rep1 = add
//     REPLICATE %Rep2 = add
//     REPLICATE %Rep3 = add %Rep0
//   Successor(s): pred.add.continue
//
//   pred.add.continue:
//   No successors
// }
TEST_F(VPlanTransformsTest, createAndOptimizeReplicateRegions) {
  VPlan &Plan = getPlan();
  Plan.addVF(ElementCount::getFixed(4));

  IntegerType *Int32 = IntegerType::get(C, 32);
  auto *AI = BinaryOperator::CreateAdd(PoisonValue::get(Int32),
                                       PoisonValue::get(Int32));

  auto *VPBB0 = Plan.getEntry();
  auto *VPBB1 = Plan.createVPBasicBlock("VPBB1");
  VPRegionBlock *R1 = Plan.createVPRegionBlock(VPBB1, VPBB1, "R1");
  VPBlockUtils::connectBlocks(VPBB0, R1);
  auto *Mask = new VPInstruction(0, {});
  // Not masked, but should still be sunk into Rep3's region.
  auto *Rep0 = new VPReplicateRecipe(AI, {}, false);
  // Masked, should each create a replicate region and get merged.
  auto *Rep1 = new VPReplicateRecipe(AI, {}, false, Mask);
  auto *Rep2 = new VPReplicateRecipe(AI, {}, false, Mask);
  auto *Rep3 = new VPReplicateRecipe(AI, {Rep0}, false, Mask);
  VPBB1->appendRecipe(Rep0);
  VPBB1->appendRecipe(Rep1);
  VPBB1->appendRecipe(Rep2);
  VPBB1->appendRecipe(Rep3);

  VPlanTransforms::createAndOptimizeReplicateRegions(Plan);

  auto *Replicator = cast<VPRegionBlock>(R1->getEntry()->getSingleSuccessor());
  EXPECT_TRUE(Replicator->isReplicator());
  auto *ReplicatorEntry = cast<VPBasicBlock>(Replicator->getEntry());
  EXPECT_EQ(ReplicatorEntry->size(), 1u);
  EXPECT_TRUE(isa<VPBranchOnMaskRecipe>(ReplicatorEntry->front()));
  auto *ReplicatorIf = cast<VPBasicBlock>(ReplicatorEntry->getSuccessors()[0]);
  EXPECT_EQ(ReplicatorIf->size(), 4u);
  EXPECT_EQ(ReplicatorEntry->getSuccessors()[1],
            ReplicatorIf->getSingleSuccessor());
  EXPECT_EQ(ReplicatorIf->getSingleSuccessor(), Replicator->getExiting());
}

} // namespace
} // namespace llvm

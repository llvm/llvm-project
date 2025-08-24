//===- llvm/unittests/Transforms/Vectorize/VPlanTest.cpp - VPlan tests ----===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lib/Transforms/Vectorize/VPlan.h"
#include "../lib/Transforms/Vectorize/VPlanCFG.h"
#include "../lib/Transforms/Vectorize/VPlanHelpers.h"
#include "VPlanTestBase.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "gtest/gtest.h"
#include <string>

namespace llvm {

namespace {

#define CHECK_ITERATOR(Range1, ...)                                            \
  do {                                                                         \
    std::vector<VPInstruction *> Tmp = {__VA_ARGS__};                          \
    EXPECT_EQ((size_t)std::distance(Range1.begin(), Range1.end()),             \
              Tmp.size());                                                     \
    for (auto Pair : zip(Range1, make_range(Tmp.begin(), Tmp.end())))          \
      EXPECT_EQ(&std::get<0>(Pair), std::get<1>(Pair));                        \
  } while (0)

using VPInstructionTest = VPlanTestBase;

TEST_F(VPInstructionTest, insertBefore) {
  VPInstruction *I1 = new VPInstruction(0, {});
  VPInstruction *I2 = new VPInstruction(1, {});
  VPInstruction *I3 = new VPInstruction(2, {});

  VPBasicBlock &VPBB1 = *getPlan().createVPBasicBlock("");
  VPBB1.appendRecipe(I1);

  I2->insertBefore(I1);
  CHECK_ITERATOR(VPBB1, I2, I1);

  I3->insertBefore(I2);
  CHECK_ITERATOR(VPBB1, I3, I2, I1);
}

TEST_F(VPInstructionTest, eraseFromParent) {
  VPInstruction *I1 = new VPInstruction(0, {});
  VPInstruction *I2 = new VPInstruction(1, {});
  VPInstruction *I3 = new VPInstruction(2, {});

  VPBasicBlock &VPBB1 = *getPlan().createVPBasicBlock("");
  VPBB1.appendRecipe(I1);
  VPBB1.appendRecipe(I2);
  VPBB1.appendRecipe(I3);

  I2->eraseFromParent();
  CHECK_ITERATOR(VPBB1, I1, I3);

  I1->eraseFromParent();
  CHECK_ITERATOR(VPBB1, I3);

  I3->eraseFromParent();
  EXPECT_TRUE(VPBB1.empty());
}

TEST_F(VPInstructionTest, moveAfter) {
  VPInstruction *I1 = new VPInstruction(0, {});
  VPInstruction *I2 = new VPInstruction(1, {});
  VPInstruction *I3 = new VPInstruction(2, {});

  VPBasicBlock &VPBB1 = *getPlan().createVPBasicBlock("");
  VPBB1.appendRecipe(I1);
  VPBB1.appendRecipe(I2);
  VPBB1.appendRecipe(I3);

  I1->moveAfter(I2);

  CHECK_ITERATOR(VPBB1, I2, I1, I3);

  VPInstruction *I4 = new VPInstruction(4, {});
  VPInstruction *I5 = new VPInstruction(5, {});
  VPBasicBlock &VPBB2 = *getPlan().createVPBasicBlock("");
  VPBB2.appendRecipe(I4);
  VPBB2.appendRecipe(I5);

  I3->moveAfter(I4);

  CHECK_ITERATOR(VPBB1, I2, I1);
  CHECK_ITERATOR(VPBB2, I4, I3, I5);
  EXPECT_EQ(I3->getParent(), I4->getParent());
}

TEST_F(VPInstructionTest, moveBefore) {
  VPInstruction *I1 = new VPInstruction(0, {});
  VPInstruction *I2 = new VPInstruction(1, {});
  VPInstruction *I3 = new VPInstruction(2, {});

  VPBasicBlock &VPBB1 = *getPlan().createVPBasicBlock("");
  VPBB1.appendRecipe(I1);
  VPBB1.appendRecipe(I2);
  VPBB1.appendRecipe(I3);

  I1->moveBefore(VPBB1, I3->getIterator());

  CHECK_ITERATOR(VPBB1, I2, I1, I3);

  VPInstruction *I4 = new VPInstruction(4, {});
  VPInstruction *I5 = new VPInstruction(5, {});
  VPBasicBlock &VPBB2 = *getPlan().createVPBasicBlock("");
  VPBB2.appendRecipe(I4);
  VPBB2.appendRecipe(I5);

  I3->moveBefore(VPBB2, I4->getIterator());

  CHECK_ITERATOR(VPBB1, I2, I1);
  CHECK_ITERATOR(VPBB2, I3, I4, I5);
  EXPECT_EQ(I3->getParent(), I4->getParent());

  VPBasicBlock &VPBB3 = *getPlan().createVPBasicBlock("");

  I4->moveBefore(VPBB3, VPBB3.end());

  CHECK_ITERATOR(VPBB1, I2, I1);
  CHECK_ITERATOR(VPBB2, I3, I5);
  CHECK_ITERATOR(VPBB3, I4);
  EXPECT_EQ(&VPBB3, I4->getParent());
}

TEST_F(VPInstructionTest, setOperand) {
  IntegerType *Int32 = IntegerType::get(C, 32);
  VPValue *VPV1 = getPlan().getOrAddLiveIn(ConstantInt::get(Int32, 1));
  VPValue *VPV2 = getPlan().getOrAddLiveIn(ConstantInt::get(Int32, 2));
  VPInstruction *I1 = new VPInstruction(0, {VPV1, VPV2});
  EXPECT_EQ(1u, VPV1->getNumUsers());
  EXPECT_EQ(I1, *VPV1->user_begin());
  EXPECT_EQ(1u, VPV2->getNumUsers());
  EXPECT_EQ(I1, *VPV2->user_begin());

  // Replace operand 0 (VPV1) with VPV3.
  VPValue *VPV3 = getPlan().getOrAddLiveIn(ConstantInt::get(Int32, 3));
  I1->setOperand(0, VPV3);
  EXPECT_EQ(0u, VPV1->getNumUsers());
  EXPECT_EQ(1u, VPV2->getNumUsers());
  EXPECT_EQ(I1, *VPV2->user_begin());
  EXPECT_EQ(1u, VPV3->getNumUsers());
  EXPECT_EQ(I1, *VPV3->user_begin());

  // Replace operand 1 (VPV2) with VPV3.
  I1->setOperand(1, VPV3);
  EXPECT_EQ(0u, VPV1->getNumUsers());
  EXPECT_EQ(0u, VPV2->getNumUsers());
  EXPECT_EQ(2u, VPV3->getNumUsers());
  EXPECT_EQ(I1, *VPV3->user_begin());
  EXPECT_EQ(I1, *std::next(VPV3->user_begin()));

  // Replace operand 0 (VPV3) with VPV4.
  VPValue *VPV4 = getPlan().getOrAddLiveIn(ConstantInt::get(Int32, 4));
  I1->setOperand(0, VPV4);
  EXPECT_EQ(1u, VPV3->getNumUsers());
  EXPECT_EQ(I1, *VPV3->user_begin());
  EXPECT_EQ(I1, *VPV4->user_begin());

  // Replace operand 1 (VPV3) with VPV4.
  I1->setOperand(1, VPV4);
  EXPECT_EQ(0u, VPV3->getNumUsers());
  EXPECT_EQ(I1, *VPV4->user_begin());
  EXPECT_EQ(I1, *std::next(VPV4->user_begin()));

  delete I1;
}

TEST_F(VPInstructionTest, replaceAllUsesWith) {
  IntegerType *Int32 = IntegerType::get(C, 32);
  VPValue *VPV1 = getPlan().getOrAddLiveIn(ConstantInt::get(Int32, 1));
  VPValue *VPV2 = getPlan().getOrAddLiveIn(ConstantInt::get(Int32, 2));
  VPInstruction *I1 = new VPInstruction(0, {VPV1, VPV2});

  // Replace all uses of VPV1 with VPV3.
  VPValue *VPV3 = getPlan().getOrAddLiveIn(ConstantInt::get(Int32, 3));
  VPV1->replaceAllUsesWith(VPV3);
  EXPECT_EQ(VPV3, I1->getOperand(0));
  EXPECT_EQ(VPV2, I1->getOperand(1));
  EXPECT_EQ(0u, VPV1->getNumUsers());
  EXPECT_EQ(1u, VPV2->getNumUsers());
  EXPECT_EQ(I1, *VPV2->user_begin());
  EXPECT_EQ(1u, VPV3->getNumUsers());
  EXPECT_EQ(I1, *VPV3->user_begin());

  // Replace all uses of VPV2 with VPV3.
  VPV2->replaceAllUsesWith(VPV3);
  EXPECT_EQ(VPV3, I1->getOperand(0));
  EXPECT_EQ(VPV3, I1->getOperand(1));
  EXPECT_EQ(0u, VPV1->getNumUsers());
  EXPECT_EQ(0u, VPV2->getNumUsers());
  EXPECT_EQ(2u, VPV3->getNumUsers());
  EXPECT_EQ(I1, *VPV3->user_begin());

  // Replace all uses of VPV3 with VPV1.
  VPV3->replaceAllUsesWith(VPV1);
  EXPECT_EQ(VPV1, I1->getOperand(0));
  EXPECT_EQ(VPV1, I1->getOperand(1));
  EXPECT_EQ(2u, VPV1->getNumUsers());
  EXPECT_EQ(I1, *VPV1->user_begin());
  EXPECT_EQ(0u, VPV2->getNumUsers());
  EXPECT_EQ(0u, VPV3->getNumUsers());

  VPInstruction *I2 = new VPInstruction(0, {VPV1, VPV2});
  EXPECT_EQ(3u, VPV1->getNumUsers());
  VPV1->replaceAllUsesWith(VPV3);
  EXPECT_EQ(3u, VPV3->getNumUsers());

  delete I1;
  delete I2;
}

TEST_F(VPInstructionTest, releaseOperandsAtDeletion) {
  IntegerType *Int32 = IntegerType::get(C, 32);
  VPValue *VPV1 = getPlan().getOrAddLiveIn(ConstantInt::get(Int32, 1));
  VPValue *VPV2 = getPlan().getOrAddLiveIn(ConstantInt::get(Int32, 1));
  VPInstruction *I1 = new VPInstruction(0, {VPV1, VPV2});

  EXPECT_EQ(1u, VPV1->getNumUsers());
  EXPECT_EQ(I1, *VPV1->user_begin());
  EXPECT_EQ(1u, VPV2->getNumUsers());
  EXPECT_EQ(I1, *VPV2->user_begin());

  delete I1;

  EXPECT_EQ(0u, VPV1->getNumUsers());
  EXPECT_EQ(0u, VPV2->getNumUsers());
}

using VPBasicBlockTest = VPlanTestBase;

TEST_F(VPBasicBlockTest, getPlan) {
  {
    VPlan &Plan = getPlan();
    VPBasicBlock *VPBB1 = Plan.getEntry();
    VPBasicBlock *VPBB2 = Plan.createVPBasicBlock("");
    VPBasicBlock *VPBB3 = Plan.createVPBasicBlock("");
    VPBasicBlock *VPBB4 = Plan.createVPBasicBlock("");

    //     VPBB1
    //     /   \
    // VPBB2  VPBB3
    //    \    /
    //    VPBB4
    VPBlockUtils::connectBlocks(VPBB1, VPBB2);
    VPBlockUtils::connectBlocks(VPBB1, VPBB3);
    VPBlockUtils::connectBlocks(VPBB2, VPBB4);
    VPBlockUtils::connectBlocks(VPBB3, VPBB4);
    VPBlockUtils::connectBlocks(VPBB4, Plan.getScalarHeader());

    EXPECT_EQ(&Plan, VPBB1->getPlan());
    EXPECT_EQ(&Plan, VPBB2->getPlan());
    EXPECT_EQ(&Plan, VPBB3->getPlan());
    EXPECT_EQ(&Plan, VPBB4->getPlan());
  }

  {
    VPlan &Plan = getPlan();
    VPBasicBlock *VPBB1 = Plan.getEntry();
    // VPBasicBlock is the entry into the VPlan, followed by a region.
    VPBasicBlock *R1BB1 = Plan.createVPBasicBlock("");
    VPBasicBlock *R1BB2 = Plan.createVPBasicBlock("");
    VPRegionBlock *R1 = Plan.createVPRegionBlock(R1BB1, R1BB2, "R1");
    VPBlockUtils::connectBlocks(R1BB1, R1BB2);

    VPBlockUtils::connectBlocks(VPBB1, R1);

    VPBlockUtils::connectBlocks(R1, Plan.getScalarHeader());

    EXPECT_EQ(&Plan, VPBB1->getPlan());
    EXPECT_EQ(&Plan, R1->getPlan());
    EXPECT_EQ(&Plan, R1BB1->getPlan());
    EXPECT_EQ(&Plan, R1BB2->getPlan());
  }

  {
    VPlan &Plan = getPlan();
    VPBasicBlock *R1BB1 = Plan.createVPBasicBlock("");
    VPBasicBlock *R1BB2 = Plan.createVPBasicBlock("");
    VPRegionBlock *R1 = Plan.createVPRegionBlock(R1BB1, R1BB2, "R1");
    VPBlockUtils::connectBlocks(R1BB1, R1BB2);

    VPBasicBlock *R2BB1 = Plan.createVPBasicBlock("");
    VPBasicBlock *R2BB2 = Plan.createVPBasicBlock("");
    VPRegionBlock *R2 = Plan.createVPRegionBlock(R2BB1, R2BB2, "R2");
    VPBlockUtils::connectBlocks(R2BB1, R2BB2);

    VPBasicBlock *VPBB1 = Plan.getEntry();
    VPBlockUtils::connectBlocks(VPBB1, R1);
    VPBlockUtils::connectBlocks(VPBB1, R2);

    VPBasicBlock *VPBB2 = Plan.createVPBasicBlock("");
    VPBlockUtils::connectBlocks(R1, VPBB2);
    VPBlockUtils::connectBlocks(R2, VPBB2);

    VPBlockUtils::connectBlocks(R2, Plan.getScalarHeader());

    EXPECT_EQ(&Plan, VPBB1->getPlan());
    EXPECT_EQ(&Plan, R1->getPlan());
    EXPECT_EQ(&Plan, R1BB1->getPlan());
    EXPECT_EQ(&Plan, R1BB2->getPlan());
    EXPECT_EQ(&Plan, R2->getPlan());
    EXPECT_EQ(&Plan, R2BB1->getPlan());
    EXPECT_EQ(&Plan, R2BB2->getPlan());
    EXPECT_EQ(&Plan, VPBB2->getPlan());
  }
}

TEST_F(VPBasicBlockTest, TraversingIteratorTest) {
  {
    // VPBasicBlocks only
    //     VPBB1
    //     /   \
    // VPBB2  VPBB3
    //    \    /
    //    VPBB4
    //
    VPlan &Plan = getPlan();
    VPBasicBlock *VPBB1 = Plan.getEntry();
    VPBasicBlock *VPBB2 = Plan.createVPBasicBlock("");
    VPBasicBlock *VPBB3 = Plan.createVPBasicBlock("");
    VPBasicBlock *VPBB4 = Plan.createVPBasicBlock("");

    VPBlockUtils::connectBlocks(VPBB1, VPBB2);
    VPBlockUtils::connectBlocks(VPBB1, VPBB3);
    VPBlockUtils::connectBlocks(VPBB2, VPBB4);
    VPBlockUtils::connectBlocks(VPBB3, VPBB4);

    VPBlockDeepTraversalWrapper<const VPBlockBase *> Start(VPBB1);
    SmallVector<const VPBlockBase *> FromIterator(depth_first(Start));
    EXPECT_EQ(4u, FromIterator.size());
    EXPECT_EQ(VPBB1, FromIterator[0]);
    EXPECT_EQ(VPBB2, FromIterator[1]);

    VPBlockUtils::connectBlocks(VPBB4, Plan.getScalarHeader());
  }

  {
    // 2 consecutive regions.
    // VPBB0
    //  |
    // R1 {
    //     \
    //     R1BB1
    //    /     \   |--|
    //  R1BB2   R1BB3 -|
    //    \      /
    //     R1BB4
    //  }
    //   |
    // R2 {
    //   \
    //    R2BB1
    //      |
    //    R2BB2
    //
    VPlan &Plan = getPlan();
    VPBasicBlock *VPBB0 = Plan.getEntry();
    VPBasicBlock *R1BB1 = Plan.createVPBasicBlock("");
    VPBasicBlock *R1BB2 = Plan.createVPBasicBlock("");
    VPBasicBlock *R1BB3 = Plan.createVPBasicBlock("");
    VPBasicBlock *R1BB4 = Plan.createVPBasicBlock("");
    VPRegionBlock *R1 = Plan.createVPRegionBlock(R1BB1, R1BB4, "R1");
    R1BB2->setParent(R1);
    R1BB3->setParent(R1);
    VPBlockUtils::connectBlocks(VPBB0, R1);
    VPBlockUtils::connectBlocks(R1BB1, R1BB2);
    VPBlockUtils::connectBlocks(R1BB1, R1BB3);
    VPBlockUtils::connectBlocks(R1BB2, R1BB4);
    VPBlockUtils::connectBlocks(R1BB3, R1BB4);
    // Cycle.
    VPBlockUtils::connectBlocks(R1BB3, R1BB3);

    VPBasicBlock *R2BB1 = Plan.createVPBasicBlock("");
    VPBasicBlock *R2BB2 = Plan.createVPBasicBlock("");
    VPRegionBlock *R2 = Plan.createVPRegionBlock(R2BB1, R2BB2, "R2");
    VPBlockUtils::connectBlocks(R2BB1, R2BB2);
    VPBlockUtils::connectBlocks(R1, R2);

    // Successors of R1.
    SmallVector<const VPBlockBase *> FromIterator(
        VPAllSuccessorsIterator<VPBlockBase *>(R1),
        VPAllSuccessorsIterator<VPBlockBase *>::end(R1));
    EXPECT_EQ(1u, FromIterator.size());
    EXPECT_EQ(R1BB1, FromIterator[0]);

    // Depth-first.
    VPBlockDeepTraversalWrapper<VPBlockBase *> Start(R1);
    FromIterator.clear();
    copy(df_begin(Start), df_end(Start), std::back_inserter(FromIterator));
    EXPECT_EQ(8u, FromIterator.size());
    EXPECT_EQ(R1, FromIterator[0]);
    EXPECT_EQ(R1BB1, FromIterator[1]);
    EXPECT_EQ(R1BB2, FromIterator[2]);
    EXPECT_EQ(R1BB4, FromIterator[3]);
    EXPECT_EQ(R2, FromIterator[4]);
    EXPECT_EQ(R2BB1, FromIterator[5]);
    EXPECT_EQ(R2BB2, FromIterator[6]);
    EXPECT_EQ(R1BB3, FromIterator[7]);

    // const VPBasicBlocks only.
    FromIterator.clear();
    copy(VPBlockUtils::blocksOnly<const VPBasicBlock>(depth_first(Start)),
         std::back_inserter(FromIterator));
    EXPECT_EQ(6u, FromIterator.size());
    EXPECT_EQ(R1BB1, FromIterator[0]);
    EXPECT_EQ(R1BB2, FromIterator[1]);
    EXPECT_EQ(R1BB4, FromIterator[2]);
    EXPECT_EQ(R2BB1, FromIterator[3]);
    EXPECT_EQ(R2BB2, FromIterator[4]);
    EXPECT_EQ(R1BB3, FromIterator[5]);

    // VPRegionBlocks only.
    SmallVector<VPRegionBlock *> FromIteratorVPRegion(
        VPBlockUtils::blocksOnly<VPRegionBlock>(depth_first(Start)));
    EXPECT_EQ(2u, FromIteratorVPRegion.size());
    EXPECT_EQ(R1, FromIteratorVPRegion[0]);
    EXPECT_EQ(R2, FromIteratorVPRegion[1]);

    // Post-order.
    FromIterator.clear();
    copy(post_order(Start), std::back_inserter(FromIterator));
    EXPECT_EQ(8u, FromIterator.size());
    EXPECT_EQ(R2BB2, FromIterator[0]);
    EXPECT_EQ(R2BB1, FromIterator[1]);
    EXPECT_EQ(R2, FromIterator[2]);
    EXPECT_EQ(R1BB4, FromIterator[3]);
    EXPECT_EQ(R1BB2, FromIterator[4]);
    EXPECT_EQ(R1BB3, FromIterator[5]);
    EXPECT_EQ(R1BB1, FromIterator[6]);
    EXPECT_EQ(R1, FromIterator[7]);

    VPBlockUtils::connectBlocks(R2, Plan.getScalarHeader());
  }

  {
    // 2 nested regions.
    //  VPBB1
    //    |
    //  R1 {
    //         R1BB1
    //       /        \
    //   R2 {          |
    //     \           |
    //     R2BB1       |
    //       |   \    R1BB2
    //     R2BB2-|     |
    //        \        |
    //         R2BB3   |
    //   }            /
    //      \        /
    //        R1BB3
    //  }
    //   |
    //  VPBB2
    //
    VPlan &Plan = getPlan();
    VPBasicBlock *R1BB1 = Plan.createVPBasicBlock("R1BB1");
    VPBasicBlock *R1BB2 = Plan.createVPBasicBlock("R1BB2");
    VPBasicBlock *R1BB3 = Plan.createVPBasicBlock("R1BB3");
    VPRegionBlock *R1 = Plan.createVPRegionBlock(R1BB1, R1BB3, "R1");

    VPBasicBlock *R2BB1 = Plan.createVPBasicBlock("R2BB1");
    VPBasicBlock *R2BB2 = Plan.createVPBasicBlock("R2BB2");
    VPBasicBlock *R2BB3 = Plan.createVPBasicBlock("R2BB3");
    VPRegionBlock *R2 = Plan.createVPRegionBlock(R2BB1, R2BB3, "R2");
    R2BB2->setParent(R2);
    VPBlockUtils::connectBlocks(R2BB1, R2BB2);
    VPBlockUtils::connectBlocks(R2BB2, R2BB1);
    VPBlockUtils::connectBlocks(R2BB2, R2BB3);

    R2->setParent(R1);
    VPBlockUtils::connectBlocks(R1BB1, R2);
    R1BB2->setParent(R1);
    VPBlockUtils::connectBlocks(R1BB1, R1BB2);
    VPBlockUtils::connectBlocks(R1BB2, R1BB3);
    VPBlockUtils::connectBlocks(R2, R1BB3);

    VPBasicBlock *VPBB1 = Plan.getEntry();
    VPBlockUtils::connectBlocks(VPBB1, R1);
    VPBasicBlock *VPBB2 = Plan.createVPBasicBlock("VPBB2");
    VPBlockUtils::connectBlocks(R1, VPBB2);

    // Depth-first.
    VPBlockDeepTraversalWrapper<VPBlockBase *> Start(VPBB1);
    SmallVector<VPBlockBase *> FromIterator(depth_first(Start));
    EXPECT_EQ(10u, FromIterator.size());
    EXPECT_EQ(VPBB1, FromIterator[0]);
    EXPECT_EQ(R1, FromIterator[1]);
    EXPECT_EQ(R1BB1, FromIterator[2]);
    EXPECT_EQ(R2, FromIterator[3]);
    EXPECT_EQ(R2BB1, FromIterator[4]);
    EXPECT_EQ(R2BB2, FromIterator[5]);
    EXPECT_EQ(R2BB3, FromIterator[6]);
    EXPECT_EQ(R1BB3, FromIterator[7]);
    EXPECT_EQ(VPBB2, FromIterator[8]);
    EXPECT_EQ(R1BB2, FromIterator[9]);

    // Post-order.
    FromIterator.clear();
    FromIterator.append(po_begin(Start), po_end(Start));
    EXPECT_EQ(10u, FromIterator.size());
    EXPECT_EQ(VPBB2, FromIterator[0]);
    EXPECT_EQ(R1BB3, FromIterator[1]);
    EXPECT_EQ(R2BB3, FromIterator[2]);
    EXPECT_EQ(R2BB2, FromIterator[3]);
    EXPECT_EQ(R2BB1, FromIterator[4]);
    EXPECT_EQ(R2, FromIterator[5]);
    EXPECT_EQ(R1BB2, FromIterator[6]);
    EXPECT_EQ(R1BB1, FromIterator[7]);
    EXPECT_EQ(R1, FromIterator[8]);
    EXPECT_EQ(VPBB1, FromIterator[9]);

    VPBlockUtils::connectBlocks(VPBB2, Plan.getScalarHeader());
  }

  {
    //  VPBB1
    //    |
    //  R1 {
    //    \
    //     R2 {
    //      R2BB1
    //        |
    //      R2BB2
    //   }
    //
    VPlan &Plan = getPlan();
    VPBasicBlock *R2BB1 = Plan.createVPBasicBlock("R2BB1");
    VPBasicBlock *R2BB2 = Plan.createVPBasicBlock("R2BB2");
    VPRegionBlock *R2 = Plan.createVPRegionBlock(R2BB1, R2BB2, "R2");
    VPBlockUtils::connectBlocks(R2BB1, R2BB2);

    VPRegionBlock *R1 = Plan.createVPRegionBlock(R2, R2, "R1");
    R2->setParent(R1);

    VPBasicBlock *VPBB1 = Plan.getEntry();
    VPBlockUtils::connectBlocks(VPBB1, R1);

    // Depth-first.
    VPBlockDeepTraversalWrapper<VPBlockBase *> Start(VPBB1);
    SmallVector<VPBlockBase *> FromIterator(depth_first(Start));
    EXPECT_EQ(5u, FromIterator.size());
    EXPECT_EQ(VPBB1, FromIterator[0]);
    EXPECT_EQ(R1, FromIterator[1]);
    EXPECT_EQ(R2, FromIterator[2]);
    EXPECT_EQ(R2BB1, FromIterator[3]);
    EXPECT_EQ(R2BB2, FromIterator[4]);

    // Post-order.
    FromIterator.clear();
    FromIterator.append(po_begin(Start), po_end(Start));
    EXPECT_EQ(5u, FromIterator.size());
    EXPECT_EQ(R2BB2, FromIterator[0]);
    EXPECT_EQ(R2BB1, FromIterator[1]);
    EXPECT_EQ(R2, FromIterator[2]);
    EXPECT_EQ(R1, FromIterator[3]);
    EXPECT_EQ(VPBB1, FromIterator[4]);

    VPBlockUtils::connectBlocks(R1, Plan.getScalarHeader());
  }

  {
    //  Nested regions with both R3 and R2 being exit nodes without successors.
    //  The successors of R1 should be used.
    //
    //  VPBB1
    //    |
    //  R1 {
    //    \
    //     R2 {
    //      \
    //      R2BB1
    //        |
    //       R3 {
    //          R3BB1
    //      }
    //   }
    //   |
    //  VPBB2
    //
    VPlan &Plan = getPlan();
    VPBasicBlock *R3BB1 = Plan.createVPBasicBlock("R3BB1");
    VPRegionBlock *R3 = Plan.createVPRegionBlock(R3BB1, R3BB1, "R3");

    VPBasicBlock *R2BB1 = Plan.createVPBasicBlock("R2BB1");
    VPRegionBlock *R2 = Plan.createVPRegionBlock(R2BB1, R3, "R2");
    R3->setParent(R2);
    VPBlockUtils::connectBlocks(R2BB1, R3);

    VPRegionBlock *R1 = Plan.createVPRegionBlock(R2, R2, "R1");
    R2->setParent(R1);

    VPBasicBlock *VPBB1 = Plan.getEntry();
    VPBasicBlock *VPBB2 = Plan.createVPBasicBlock("VPBB2");
    VPBlockUtils::connectBlocks(VPBB1, R1);
    VPBlockUtils::connectBlocks(R1, VPBB2);

    // Depth-first.
    VPBlockDeepTraversalWrapper<VPBlockBase *> Start(VPBB1);
    SmallVector<VPBlockBase *> FromIterator(depth_first(Start));
    EXPECT_EQ(7u, FromIterator.size());
    EXPECT_EQ(VPBB1, FromIterator[0]);
    EXPECT_EQ(R1, FromIterator[1]);
    EXPECT_EQ(R2, FromIterator[2]);
    EXPECT_EQ(R2BB1, FromIterator[3]);
    EXPECT_EQ(R3, FromIterator[4]);
    EXPECT_EQ(R3BB1, FromIterator[5]);
    EXPECT_EQ(VPBB2, FromIterator[6]);

    SmallVector<VPBlockBase *> FromIteratorVPBB;
    copy(VPBlockUtils::blocksOnly<VPBasicBlock>(depth_first(Start)),
         std::back_inserter(FromIteratorVPBB));
    EXPECT_EQ(VPBB1, FromIteratorVPBB[0]);
    EXPECT_EQ(R2BB1, FromIteratorVPBB[1]);
    EXPECT_EQ(R3BB1, FromIteratorVPBB[2]);
    EXPECT_EQ(VPBB2, FromIteratorVPBB[3]);

    // Post-order.
    FromIterator.clear();
    copy(post_order(Start), std::back_inserter(FromIterator));
    EXPECT_EQ(7u, FromIterator.size());
    EXPECT_EQ(VPBB2, FromIterator[0]);
    EXPECT_EQ(R3BB1, FromIterator[1]);
    EXPECT_EQ(R3, FromIterator[2]);
    EXPECT_EQ(R2BB1, FromIterator[3]);
    EXPECT_EQ(R2, FromIterator[4]);
    EXPECT_EQ(R1, FromIterator[5]);
    EXPECT_EQ(VPBB1, FromIterator[6]);

    // Post-order, const VPRegionBlocks only.
    VPBlockDeepTraversalWrapper<const VPBlockBase *> StartConst(VPBB1);
    SmallVector<const VPRegionBlock *> FromIteratorVPRegion(
        VPBlockUtils::blocksOnly<const VPRegionBlock>(post_order(StartConst)));
    EXPECT_EQ(3u, FromIteratorVPRegion.size());
    EXPECT_EQ(R3, FromIteratorVPRegion[0]);
    EXPECT_EQ(R2, FromIteratorVPRegion[1]);
    EXPECT_EQ(R1, FromIteratorVPRegion[2]);

    // Post-order, VPBasicBlocks only.
    FromIterator.clear();
    copy(VPBlockUtils::blocksOnly<VPBasicBlock>(post_order(Start)),
         std::back_inserter(FromIterator));
    EXPECT_EQ(FromIterator.size(), 4u);
    EXPECT_EQ(VPBB2, FromIterator[0]);
    EXPECT_EQ(R3BB1, FromIterator[1]);
    EXPECT_EQ(R2BB1, FromIterator[2]);
    EXPECT_EQ(VPBB1, FromIterator[3]);

    VPBlockUtils::connectBlocks(VPBB2, Plan.getScalarHeader());
  }
}

TEST_F(VPBasicBlockTest, reassociateBlocks) {
  {
    // Ensure that when we reassociate a basic block, we make sure to update any
    // references to it in VPWidenPHIRecipes' incoming blocks.
    VPlan &Plan = getPlan();
    VPBasicBlock *VPBB1 = Plan.createVPBasicBlock("VPBB1");
    VPBasicBlock *VPBB2 = Plan.createVPBasicBlock("VPBB2");
    VPBlockUtils::connectBlocks(VPBB1, VPBB2);

    auto *WidenPhi = new VPWidenPHIRecipe(nullptr);
    IntegerType *Int32 = IntegerType::get(C, 32);
    VPValue *Val = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 1));
    WidenPhi->addOperand(Val);
    VPBB2->appendRecipe(WidenPhi);

    VPBasicBlock *VPBBNew = Plan.createVPBasicBlock("VPBBNew");
    VPBlockUtils::reassociateBlocks(VPBB1, VPBBNew);
    EXPECT_EQ(VPBB2->getSinglePredecessor(), VPBBNew);
    EXPECT_EQ(WidenPhi->getIncomingBlock(0), VPBBNew);
  }

  {
    // Ensure that we update VPWidenPHIRecipes that are nested inside a
    // VPRegionBlock.
    VPlan &Plan = getPlan();
    VPBasicBlock *VPBB1 = Plan.createVPBasicBlock("VPBB1");
    VPBasicBlock *VPBB2 = Plan.createVPBasicBlock("VPBB2");
    VPRegionBlock *R1 = Plan.createVPRegionBlock(VPBB2, VPBB2, "R1");
    VPBlockUtils::connectBlocks(VPBB1, R1);

    auto *WidenPhi = new VPWidenPHIRecipe(nullptr);
    IntegerType *Int32 = IntegerType::get(C, 32);
    VPValue *Val = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 1));
    WidenPhi->addOperand(Val);
    WidenPhi->addOperand(Val);
    VPBB2->appendRecipe(WidenPhi);

    VPBasicBlock *VPBBNew = Plan.createVPBasicBlock("VPBBNew");
    VPBlockUtils::reassociateBlocks(VPBB1, VPBBNew);
    EXPECT_EQ(R1->getSinglePredecessor(), VPBBNew);
    EXPECT_EQ(WidenPhi->getIncomingBlock(0), VPBBNew);
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
TEST_F(VPBasicBlockTest, print) {
  VPInstruction *TC = new VPInstruction(Instruction::PHI, {});
  VPlan &Plan = getPlan(TC);
  IntegerType *Int32 = IntegerType::get(C, 32);
  VPValue *Val = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 1));
  VPBasicBlock *VPBB0 = Plan.getEntry();
  VPBB0->appendRecipe(TC);

  VPInstruction *I1 = new VPInstruction(Instruction::Add, {Val, Val});
  VPInstruction *I2 = new VPInstruction(Instruction::Sub, {I1, Val});
  VPInstruction *I3 = new VPInstruction(Instruction::Br, {I1, I2});

  VPBasicBlock *VPBB1 = Plan.createVPBasicBlock("");
  VPBB1->appendRecipe(I1);
  VPBB1->appendRecipe(I2);
  VPBB1->appendRecipe(I3);
  VPBB1->setName("bb1");

  VPInstruction *I4 = new VPInstruction(Instruction::Mul, {I2, I1});
  VPInstruction *I5 = new VPInstruction(Instruction::Br, {I4});
  VPBasicBlock *VPBB2 = Plan.createVPBasicBlock("");
  VPBB2->appendRecipe(I4);
  VPBB2->appendRecipe(I5);
  VPBB2->setName("bb2");

  VPBlockUtils::connectBlocks(VPBB1, VPBB2);

  // Check printing an instruction without associated VPlan.
  {
    std::string I3Dump;
    raw_string_ostream OS(I3Dump);
    VPSlotTracker SlotTracker;
    I3->print(OS, "", SlotTracker);
    EXPECT_EQ("EMIT br <badref>, <badref>", I3Dump);
  }

  VPBlockUtils::connectBlocks(VPBB2, Plan.getScalarHeader());
  VPBlockUtils::connectBlocks(VPBB0, VPBB1);
  std::string FullDump;
  raw_string_ostream OS(FullDump);
  Plan.printDOT(OS);

  const char *ExpectedStr = R"(digraph VPlan {
graph [labelloc=t, fontsize=30; label="Vectorization Plan\n for UF\>=1\nvp\<%1\> = original trip-count\n"]
node [shape=rect, fontname=Courier, fontsize=30]
edge [fontname=Courier, fontsize=30]
compound=true
  N0 [label =
    "preheader:\l" +
    "  EMIT-SCALAR vp\<%1\> = phi \l" +
    "Successor(s): bb1\l"
  ]
  N0 -> N1 [ label=""]
  N1 [label =
    "bb1:\l" +
    "  EMIT vp\<%2\> = add ir\<1\>, ir\<1\>\l" +
    "  EMIT vp\<%3\> = sub vp\<%2\>, ir\<1\>\l" +
    "  EMIT br vp\<%2\>, vp\<%3\>\l" +
    "Successor(s): bb2\l"
  ]
  N1 -> N2 [ label=""]
  N2 [label =
    "bb2:\l" +
    "  EMIT vp\<%5\> = mul vp\<%3\>, vp\<%2\>\l" +
    "  EMIT br vp\<%5\>\l" +
    "Successor(s): ir-bb\<scalar.header\>\l"
  ]
  N2 -> N3 [ label=""]
  N3 [label =
    "ir-bb\<scalar.header\>:\l" +
    "No successors\l"
  ]
}
)";
  EXPECT_EQ(ExpectedStr, FullDump);

  const char *ExpectedBlock1Str = R"(bb1:
  EMIT vp<%2> = add ir<1>, ir<1>
  EMIT vp<%3> = sub vp<%2>, ir<1>
  EMIT br vp<%2>, vp<%3>
Successor(s): bb2
)";
  std::string Block1Dump;
  raw_string_ostream OS1(Block1Dump);
  VPBB1->print(OS1);
  EXPECT_EQ(ExpectedBlock1Str, Block1Dump);

  // Ensure that numbering is good when dumping the second block in isolation.
  const char *ExpectedBlock2Str = R"(bb2:
  EMIT vp<%5> = mul vp<%3>, vp<%2>
  EMIT br vp<%5>
Successor(s): ir-bb<scalar.header>
)";
  std::string Block2Dump;
  raw_string_ostream OS2(Block2Dump);
  VPBB2->print(OS2);
  EXPECT_EQ(ExpectedBlock2Str, Block2Dump);

  {
    std::string I3Dump;
    raw_string_ostream OS(I3Dump);
    VPSlotTracker SlotTracker(&Plan);
    I3->print(OS, "", SlotTracker);
    EXPECT_EQ("EMIT br vp<%2>, vp<%3>", I3Dump);
  }

  {
    std::string I4Dump;
    raw_string_ostream OS(I4Dump);
    OS << *I4;
    EXPECT_EQ("EMIT vp<%5> = mul vp<%3>, vp<%2>", I4Dump);
  }
}

TEST_F(VPBasicBlockTest, printPlanWithVFsAndUFs) {
  VPInstruction *TC = new VPInstruction(Instruction::Sub, {});
  VPlan &Plan = getPlan(TC);
  VPBasicBlock *VPBB0 = Plan.getEntry();
  VPBB0->appendRecipe(TC);

  VPInstruction *I1 = new VPInstruction(Instruction::Add, {});
  VPBasicBlock *VPBB1 = Plan.createVPBasicBlock("");
  VPBB1->appendRecipe(I1);
  VPBB1->setName("bb1");

  VPBlockUtils::connectBlocks(VPBB1, Plan.getScalarHeader());
  VPBlockUtils::connectBlocks(VPBB0, VPBB1);
  Plan.setName("TestPlan");
  Plan.addVF(ElementCount::getFixed(4));

  {
    std::string FullDump;
    raw_string_ostream OS(FullDump);
    Plan.print(OS);

    const char *ExpectedStr = R"(VPlan 'TestPlan for VF={4},UF>=1' {
vp<%1> = original trip-count

preheader:
  EMIT vp<%1> = sub 
Successor(s): bb1

bb1:
  EMIT vp<%2> = add 
Successor(s): ir-bb<scalar.header>

ir-bb<scalar.header>:
No successors
}
)";
    EXPECT_EQ(ExpectedStr, FullDump);
  }

  {
    Plan.addVF(ElementCount::getScalable(8));
    std::string FullDump;
    raw_string_ostream OS(FullDump);
    Plan.print(OS);

    const char *ExpectedStr = R"(VPlan 'TestPlan for VF={4,vscale x 8},UF>=1' {
vp<%1> = original trip-count

preheader:
  EMIT vp<%1> = sub 
Successor(s): bb1

bb1:
  EMIT vp<%2> = add 
Successor(s): ir-bb<scalar.header>

ir-bb<scalar.header>:
No successors
}
)";
    EXPECT_EQ(ExpectedStr, FullDump);
  }

  {
    Plan.setUF(4);
    std::string FullDump;
    raw_string_ostream OS(FullDump);
    Plan.print(OS);

    const char *ExpectedStr = R"(VPlan 'TestPlan for VF={4,vscale x 8},UF={4}' {
vp<%1> = original trip-count

preheader:
  EMIT vp<%1> = sub 
Successor(s): bb1

bb1:
  EMIT vp<%2> = add 
Successor(s): ir-bb<scalar.header>

ir-bb<scalar.header>:
No successors
}
)";
    EXPECT_EQ(ExpectedStr, FullDump);
  }
}

TEST_F(VPBasicBlockTest, cloneAndPrint) {
  VPlan &Plan = getPlan(nullptr);
  VPBasicBlock *VPBB0 = Plan.getEntry();

  IntegerType *Int32 = IntegerType::get(C, 32);
  VPValue *Val = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 1));

  VPInstruction *I1 = new VPInstruction(Instruction::Add, {Val, Val});
  VPInstruction *I2 = new VPInstruction(Instruction::Sub, {I1, Val});
  VPInstruction *I3 = new VPInstruction(Instruction::Store, {I1, I2});

  VPBasicBlock *VPBB1 = Plan.createVPBasicBlock("");
  VPBB1->appendRecipe(I1);
  VPBB1->appendRecipe(I2);
  VPBB1->appendRecipe(I3);
  VPBB1->setName("bb1");
  VPBlockUtils::connectBlocks(VPBB0, VPBB1);

  const char *ExpectedStr = R"(digraph VPlan {
graph [labelloc=t, fontsize=30; label="Vectorization Plan\n for UF\>=1\n"]
node [shape=rect, fontname=Courier, fontsize=30]
edge [fontname=Courier, fontsize=30]
compound=true
  N0 [label =
    "preheader:\l" +
    "Successor(s): bb1\l"
  ]
  N0 -> N1 [ label=""]
  N1 [label =
    "bb1:\l" +
    "  EMIT vp\<%1\> = add ir\<1\>, ir\<1\>\l" +
    "  EMIT vp\<%2\> = sub vp\<%1\>, ir\<1\>\l" +
    "  EMIT store vp\<%1\>, vp\<%2\>\l" +
    "No successors\l"
  ]
}
)";
  // Check that printing a cloned plan produces the same output.
  std::string FullDump;
  raw_string_ostream OS(FullDump);
  VPlan *Clone = Plan.duplicate();
  Clone->printDOT(OS);
  EXPECT_EQ(ExpectedStr, FullDump);
  delete Clone;
}
#endif

using VPRecipeTest = VPlanTestBase;
TEST_F(VPRecipeTest, CastVPInstructionToVPUser) {
  IntegerType *Int32 = IntegerType::get(C, 32);
  VPlan &Plan = getPlan();
  VPValue *Op1 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 1));
  VPValue *Op2 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 2));
  VPInstruction Recipe(Instruction::Add, {Op1, Op2});
  EXPECT_TRUE(isa<VPUser>(&Recipe));
  VPRecipeBase *BaseR = &Recipe;
  EXPECT_TRUE(isa<VPUser>(BaseR));
  EXPECT_EQ(&Recipe, BaseR);
}

TEST_F(VPRecipeTest, CastVPWidenRecipeToVPUser) {
  VPlan &Plan = getPlan();
  IntegerType *Int32 = IntegerType::get(C, 32);
  auto *AI = BinaryOperator::CreateAdd(PoisonValue::get(Int32),
                                       PoisonValue::get(Int32));
  VPValue *Op1 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 1));
  VPValue *Op2 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 2));
  SmallVector<VPValue *, 2> Args;
  Args.push_back(Op1);
  Args.push_back(Op2);
  VPWidenRecipe WidenR(*AI, make_range(Args.begin(), Args.end()));
  EXPECT_TRUE(isa<VPUser>(&WidenR));
  VPRecipeBase *WidenRBase = &WidenR;
  EXPECT_TRUE(isa<VPUser>(WidenRBase));
  EXPECT_EQ(&WidenR, WidenRBase);
  delete AI;
}

TEST_F(VPRecipeTest, CastVPWidenCallRecipeToVPUserAndVPDef) {
  VPlan &Plan = getPlan();
  IntegerType *Int32 = IntegerType::get(C, 32);
  FunctionType *FTy = FunctionType::get(Int32, false);
  Function *Fn = Function::Create(FTy, GlobalValue::ExternalLinkage, 0);
  auto *Call = CallInst::Create(FTy, Fn);
  VPValue *Op1 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 1));
  VPValue *Op2 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 2));
  VPValue *CalledFn = Plan.getOrAddLiveIn(Call->getCalledFunction());
  SmallVector<VPValue *, 2> Args;
  Args.push_back(Op1);
  Args.push_back(Op2);
  Args.push_back(CalledFn);
  VPWidenCallRecipe Recipe(Call, Fn, Args);
  EXPECT_TRUE(isa<VPUser>(&Recipe));
  VPRecipeBase *BaseR = &Recipe;
  EXPECT_TRUE(isa<VPUser>(BaseR));
  EXPECT_EQ(&Recipe, BaseR);

  VPValue *VPV = &Recipe;
  EXPECT_TRUE(VPV->getDefiningRecipe());
  EXPECT_EQ(&Recipe, VPV->getDefiningRecipe());

  delete Call;
  delete Fn;
}

TEST_F(VPRecipeTest, CastVPWidenSelectRecipeToVPUserAndVPDef) {
  VPlan &Plan = getPlan();
  IntegerType *Int1 = IntegerType::get(C, 1);
  IntegerType *Int32 = IntegerType::get(C, 32);
  auto *SelectI = SelectInst::Create(
      PoisonValue::get(Int1), PoisonValue::get(Int32), PoisonValue::get(Int32));
  VPValue *Op1 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 1));
  VPValue *Op2 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 2));
  VPValue *Op3 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 3));
  SmallVector<VPValue *, 4> Args;
  Args.push_back(Op1);
  Args.push_back(Op2);
  Args.push_back(Op3);
  VPWidenSelectRecipe WidenSelectR(*SelectI,
                                   make_range(Args.begin(), Args.end()));
  EXPECT_TRUE(isa<VPUser>(&WidenSelectR));
  VPRecipeBase *BaseR = &WidenSelectR;
  EXPECT_TRUE(isa<VPUser>(BaseR));
  EXPECT_EQ(&WidenSelectR, BaseR);

  VPValue *VPV = &WidenSelectR;
  EXPECT_TRUE(isa<VPRecipeBase>(VPV->getDefiningRecipe()));
  EXPECT_EQ(&WidenSelectR, VPV->getDefiningRecipe());

  delete SelectI;
}

TEST_F(VPRecipeTest, CastVPWidenGEPRecipeToVPUserAndVPDef) {
  VPlan &Plan = getPlan();
  IntegerType *Int32 = IntegerType::get(C, 32);
  PointerType *Int32Ptr = PointerType::get(C, 0);
  auto *GEP = GetElementPtrInst::Create(Int32, PoisonValue::get(Int32Ptr),
                                        PoisonValue::get(Int32));
  VPValue *Op1 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 1));
  VPValue *Op2 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 2));
  SmallVector<VPValue *, 4> Args;
  Args.push_back(Op1);
  Args.push_back(Op2);
  VPWidenGEPRecipe Recipe(GEP, make_range(Args.begin(), Args.end()));
  EXPECT_TRUE(isa<VPUser>(&Recipe));
  VPRecipeBase *BaseR = &Recipe;
  EXPECT_TRUE(isa<VPUser>(BaseR));
  EXPECT_EQ(&Recipe, BaseR);

  VPValue *VPV = &Recipe;
  EXPECT_TRUE(isa<VPRecipeBase>(VPV->getDefiningRecipe()));
  EXPECT_EQ(&Recipe, VPV->getDefiningRecipe());

  delete GEP;
}

TEST_F(VPRecipeTest, CastVPBlendRecipeToVPUser) {
  VPlan &Plan = getPlan();
  IntegerType *Int32 = IntegerType::get(C, 32);
  auto *Phi = PHINode::Create(Int32, 1);

  VPValue *I1 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 1));
  VPValue *I2 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 2));
  VPValue *M2 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 3));
  SmallVector<VPValue *, 4> Args;
  Args.push_back(I1);
  Args.push_back(I2);
  Args.push_back(M2);
  VPBlendRecipe Recipe(Phi, Args, {});
  EXPECT_TRUE(isa<VPUser>(&Recipe));
  VPRecipeBase *BaseR = &Recipe;
  EXPECT_TRUE(isa<VPUser>(BaseR));
  delete Phi;
}

TEST_F(VPRecipeTest, CastVPInterleaveRecipeToVPUser) {
  VPlan &Plan = getPlan();
  IntegerType *Int32 = IntegerType::get(C, 32);
  VPValue *Addr = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 1));
  VPValue *Mask = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 2));
  InterleaveGroup<Instruction> IG(4, false, Align(4));
  VPInterleaveRecipe Recipe(&IG, Addr, {}, Mask, false, {}, DebugLoc());
  EXPECT_TRUE(isa<VPUser>(&Recipe));
  VPRecipeBase *BaseR = &Recipe;
  EXPECT_TRUE(isa<VPUser>(BaseR));
  EXPECT_EQ(&Recipe, BaseR);
}

TEST_F(VPRecipeTest, CastVPReplicateRecipeToVPUser) {
  VPlan &Plan = getPlan();
  IntegerType *Int32 = IntegerType::get(C, 32);
  VPValue *Op1 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 1));
  VPValue *Op2 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 2));
  SmallVector<VPValue *, 4> Args;
  Args.push_back(Op1);
  Args.push_back(Op2);

  FunctionType *FTy = FunctionType::get(Int32, false);
  auto *Call = CallInst::Create(FTy, PoisonValue::get(FTy));
  VPReplicateRecipe Recipe(Call, make_range(Args.begin(), Args.end()), true);
  EXPECT_TRUE(isa<VPUser>(&Recipe));
  VPRecipeBase *BaseR = &Recipe;
  EXPECT_TRUE(isa<VPUser>(BaseR));
  delete Call;
}

TEST_F(VPRecipeTest, CastVPBranchOnMaskRecipeToVPUser) {
  VPlan &Plan = getPlan();
  IntegerType *Int32 = IntegerType::get(C, 32);
  VPValue *Mask = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 1));
  VPBranchOnMaskRecipe Recipe(Mask, {});
  EXPECT_TRUE(isa<VPUser>(&Recipe));
  VPRecipeBase *BaseR = &Recipe;
  EXPECT_TRUE(isa<VPUser>(BaseR));
  EXPECT_EQ(&Recipe, BaseR);
}

TEST_F(VPRecipeTest, CastVPWidenMemoryRecipeToVPUserAndVPDef) {
  VPlan &Plan = getPlan();
  IntegerType *Int32 = IntegerType::get(C, 32);
  PointerType *Int32Ptr = PointerType::get(C, 0);
  auto *Load =
      new LoadInst(Int32, PoisonValue::get(Int32Ptr), "", false, Align(1));
  VPValue *Addr = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 1));
  VPValue *Mask = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 2));
  VPWidenLoadRecipe Recipe(*Load, Addr, Mask, true, false, {}, {});
  EXPECT_TRUE(isa<VPUser>(&Recipe));
  VPRecipeBase *BaseR = &Recipe;
  EXPECT_TRUE(isa<VPUser>(BaseR));
  EXPECT_EQ(&Recipe, BaseR);

  VPValue *VPV = Recipe.getVPSingleValue();
  EXPECT_TRUE(isa<VPRecipeBase>(VPV->getDefiningRecipe()));
  EXPECT_EQ(&Recipe, VPV->getDefiningRecipe());

  delete Load;
}

TEST_F(VPRecipeTest, MayHaveSideEffectsAndMayReadWriteMemory) {
  IntegerType *Int1 = IntegerType::get(C, 1);
  IntegerType *Int32 = IntegerType::get(C, 32);
  PointerType *Int32Ptr = PointerType::get(C, 0);
  VPlan &Plan = getPlan();

  {
    auto *AI = BinaryOperator::CreateAdd(PoisonValue::get(Int32),
                                         PoisonValue::get(Int32));
    VPValue *Op1 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 1));
    VPValue *Op2 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 2));
    SmallVector<VPValue *, 2> Args;
    Args.push_back(Op1);
    Args.push_back(Op2);
    VPWidenRecipe Recipe(*AI, make_range(Args.begin(), Args.end()));
    EXPECT_FALSE(Recipe.mayHaveSideEffects());
    EXPECT_FALSE(Recipe.mayReadFromMemory());
    EXPECT_FALSE(Recipe.mayWriteToMemory());
    EXPECT_FALSE(Recipe.mayReadOrWriteMemory());
    delete AI;
  }

  {
    auto *SelectI =
        SelectInst::Create(PoisonValue::get(Int1), PoisonValue::get(Int32),
                           PoisonValue::get(Int32));
    VPValue *Op1 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 1));
    VPValue *Op2 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 2));
    VPValue *Op3 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 3));
    SmallVector<VPValue *, 4> Args;
    Args.push_back(Op1);
    Args.push_back(Op2);
    Args.push_back(Op3);
    VPWidenSelectRecipe Recipe(*SelectI, make_range(Args.begin(), Args.end()));
    EXPECT_FALSE(Recipe.mayHaveSideEffects());
    EXPECT_FALSE(Recipe.mayReadFromMemory());
    EXPECT_FALSE(Recipe.mayWriteToMemory());
    EXPECT_FALSE(Recipe.mayReadOrWriteMemory());
    delete SelectI;
  }

  {
    auto *GEP = GetElementPtrInst::Create(Int32, PoisonValue::get(Int32Ptr),
                                          PoisonValue::get(Int32));
    VPValue *Op1 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 1));
    VPValue *Op2 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 2));
    SmallVector<VPValue *, 4> Args;
    Args.push_back(Op1);
    Args.push_back(Op2);
    VPWidenGEPRecipe Recipe(GEP, make_range(Args.begin(), Args.end()));
    EXPECT_FALSE(Recipe.mayHaveSideEffects());
    EXPECT_FALSE(Recipe.mayReadFromMemory());
    EXPECT_FALSE(Recipe.mayWriteToMemory());
    EXPECT_FALSE(Recipe.mayReadOrWriteMemory());
    delete GEP;
  }

  {
    VPValue *Mask = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 1));

    VPBranchOnMaskRecipe Recipe(Mask, {});
    EXPECT_TRUE(Recipe.mayHaveSideEffects());
    EXPECT_FALSE(Recipe.mayReadFromMemory());
    EXPECT_FALSE(Recipe.mayWriteToMemory());
    EXPECT_FALSE(Recipe.mayReadOrWriteMemory());
  }

  {
    auto *Add = BinaryOperator::CreateAdd(PoisonValue::get(Int32),
                                          PoisonValue::get(Int32));
    VPValue *ChainOp = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 1));
    VPValue *VecOp = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 2));
    VPValue *CondOp = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 3));
    VPReductionRecipe Recipe(RecurKind::Add, FastMathFlags(), Add, ChainOp,
                             CondOp, VecOp, false);
    EXPECT_FALSE(Recipe.mayHaveSideEffects());
    EXPECT_FALSE(Recipe.mayReadFromMemory());
    EXPECT_FALSE(Recipe.mayWriteToMemory());
    EXPECT_FALSE(Recipe.mayReadOrWriteMemory());
    delete Add;
  }

  {
    auto *Add = BinaryOperator::CreateAdd(PoisonValue::get(Int32),
                                          PoisonValue::get(Int32));
    VPValue *ChainOp = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 1));
    VPValue *VecOp = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 2));
    VPValue *CondOp = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 3));
    VPReductionRecipe Recipe(RecurKind::Add, FastMathFlags(), Add, ChainOp,
                             CondOp, VecOp, false);
    VPValue *EVL = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 4));
    VPReductionEVLRecipe EVLRecipe(Recipe, *EVL, CondOp);
    EXPECT_FALSE(EVLRecipe.mayHaveSideEffects());
    EXPECT_FALSE(EVLRecipe.mayReadFromMemory());
    EXPECT_FALSE(EVLRecipe.mayWriteToMemory());
    EXPECT_FALSE(EVLRecipe.mayReadOrWriteMemory());
    delete Add;
  }

  {
    auto *Load =
        new LoadInst(Int32, PoisonValue::get(Int32Ptr), "", false, Align(1));
    VPValue *Mask = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 1));
    VPValue *Addr = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 2));
    VPWidenLoadRecipe Recipe(*Load, Addr, Mask, true, false, {}, {});
    EXPECT_FALSE(Recipe.mayHaveSideEffects());
    EXPECT_TRUE(Recipe.mayReadFromMemory());
    EXPECT_FALSE(Recipe.mayWriteToMemory());
    EXPECT_TRUE(Recipe.mayReadOrWriteMemory());
    delete Load;
  }

  {
    auto *Store = new StoreInst(PoisonValue::get(Int32),
                                PoisonValue::get(Int32Ptr), false, Align(1));
    VPValue *Mask = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 1));
    VPValue *Addr = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 2));
    VPValue *StoredV = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 3));
    VPWidenStoreRecipe Recipe(*Store, Addr, StoredV, Mask, false, false, {},
                              {});
    EXPECT_TRUE(Recipe.mayHaveSideEffects());
    EXPECT_FALSE(Recipe.mayReadFromMemory());
    EXPECT_TRUE(Recipe.mayWriteToMemory());
    EXPECT_TRUE(Recipe.mayReadOrWriteMemory());
    delete Store;
  }

  {
    FunctionType *FTy = FunctionType::get(Int32, false);
    Function *Fn = Function::Create(FTy, GlobalValue::ExternalLinkage, 0);
    auto *Call = CallInst::Create(FTy, Fn);
    VPValue *Op1 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 1));
    VPValue *Op2 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 2));
    VPValue *CalledFn = Plan.getOrAddLiveIn(Call->getCalledFunction());
    SmallVector<VPValue *, 3> Args;
    Args.push_back(Op1);
    Args.push_back(Op2);
    Args.push_back(CalledFn);
    VPWidenCallRecipe Recipe(Call, Fn, Args);
    EXPECT_TRUE(Recipe.mayHaveSideEffects());
    EXPECT_TRUE(Recipe.mayReadFromMemory());
    EXPECT_TRUE(Recipe.mayWriteToMemory());
    EXPECT_TRUE(Recipe.mayReadOrWriteMemory());
    delete Call;
    delete Fn;
  }

  {
    // Test for a call to a function without side-effects.
    Module M("", C);
    PointerType *PtrTy = PointerType::get(C, 0);
    Function *TheFn =
        Intrinsic::getOrInsertDeclaration(&M, Intrinsic::thread_pointer, PtrTy);

    auto *Call = CallInst::Create(TheFn->getFunctionType(), TheFn);
    VPValue *Op1 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 1));
    VPValue *Op2 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 2));
    VPValue *CalledFn = Plan.getOrAddLiveIn(Call->getCalledFunction());
    SmallVector<VPValue *, 3> Args;
    Args.push_back(Op1);
    Args.push_back(Op2);
    Args.push_back(CalledFn);
    VPWidenCallRecipe Recipe(Call, TheFn, Args);
    EXPECT_FALSE(Recipe.mayHaveSideEffects());
    EXPECT_FALSE(Recipe.mayReadFromMemory());
    EXPECT_FALSE(Recipe.mayWriteToMemory());
    EXPECT_FALSE(Recipe.mayReadOrWriteMemory());
    delete Call;
  }

  {
    VPValue *Op1 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 1));
    VPValue *Op2 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 2));
    InductionDescriptor IndDesc;
    VPScalarIVStepsRecipe Recipe(IndDesc, Op1, Op2, Op2);
    EXPECT_FALSE(Recipe.mayHaveSideEffects());
    EXPECT_FALSE(Recipe.mayReadFromMemory());
    EXPECT_FALSE(Recipe.mayWriteToMemory());
    EXPECT_FALSE(Recipe.mayReadOrWriteMemory());
  }

  {
    VPValue *Op1 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 1));
    VPValue *Op2 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 2));
    VPInstruction VPInst(Instruction::Add, {Op1, Op2});
    VPRecipeBase &Recipe = VPInst;
    EXPECT_FALSE(Recipe.mayHaveSideEffects());
    EXPECT_FALSE(Recipe.mayReadFromMemory());
    EXPECT_FALSE(Recipe.mayWriteToMemory());
    EXPECT_FALSE(Recipe.mayReadOrWriteMemory());
  }
  {
    VPValue *Op1 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 1));
    VPPredInstPHIRecipe Recipe(Op1, {});
    EXPECT_FALSE(Recipe.mayHaveSideEffects());
    EXPECT_FALSE(Recipe.mayReadFromMemory());
    EXPECT_FALSE(Recipe.mayWriteToMemory());
    EXPECT_FALSE(Recipe.mayReadOrWriteMemory());
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
TEST_F(VPRecipeTest, dumpRecipeInPlan) {
  VPlan &Plan = getPlan();
  VPBasicBlock *VPBB0 = Plan.getEntry();
  VPBasicBlock *VPBB1 = Plan.createVPBasicBlock("");
  VPBlockUtils::connectBlocks(VPBB1, Plan.getScalarHeader());
  VPBlockUtils::connectBlocks(VPBB0, VPBB1);

  IntegerType *Int32 = IntegerType::get(C, 32);
  auto *AI = BinaryOperator::CreateAdd(PoisonValue::get(Int32),
                                       PoisonValue::get(Int32));
  AI->setName("a");
  SmallVector<VPValue *, 2> Args;
  VPValue *ExtVPV1 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 1));
  VPValue *ExtVPV2 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 2));
  Args.push_back(ExtVPV1);
  Args.push_back(ExtVPV2);
  VPWidenRecipe *WidenR =
      new VPWidenRecipe(*AI, make_range(Args.begin(), Args.end()));
  VPBB1->appendRecipe(WidenR);

  {
    // Use EXPECT_EXIT to capture stderr and compare against expected output.
    //
    // Test VPValue::dump().
    VPValue *VPV = WidenR;
    EXPECT_EXIT(
        {
          VPV->dump();
          exit(0);
        },
        testing::ExitedWithCode(0), "WIDEN ir<%a> = add ir<1>, ir<2>");

    VPDef *Def = WidenR;
    EXPECT_EXIT(
        {
          Def->dump();
          exit(0);
        },
        testing::ExitedWithCode(0), "WIDEN ir<%a> = add ir<1>, ir<2>");

    EXPECT_EXIT(
        {
          WidenR->dump();
          exit(0);
        },
        testing::ExitedWithCode(0), "WIDEN ir<%a> = add ir<1>, ir<2>");

    // Test VPRecipeBase::dump().
    VPRecipeBase *R = WidenR;
    EXPECT_EXIT(
        {
          R->dump();
          exit(0);
        },
        testing::ExitedWithCode(0), "WIDEN ir<%a> = add ir<1>, ir<2>");

    // Test VPDef::dump().
    VPDef *D = WidenR;
    EXPECT_EXIT(
        {
          D->dump();
          exit(0);
        },
        testing::ExitedWithCode(0), "WIDEN ir<%a> = add ir<1>, ir<2>");
  }

  delete AI;
}

TEST_F(VPRecipeTest, dumpRecipeUnnamedVPValuesInPlan) {
  VPlan &Plan = getPlan();
  VPBasicBlock *VPBB0 = Plan.getEntry();
  VPBasicBlock *VPBB1 = Plan.createVPBasicBlock("");
  VPBlockUtils::connectBlocks(VPBB1, Plan.getScalarHeader());
  VPBlockUtils::connectBlocks(VPBB0, VPBB1);

  IntegerType *Int32 = IntegerType::get(C, 32);
  auto *AI = BinaryOperator::CreateAdd(PoisonValue::get(Int32),
                                       PoisonValue::get(Int32));
  AI->setName("a");
  SmallVector<VPValue *, 2> Args;
  VPValue *ExtVPV1 = Plan.getOrAddLiveIn(ConstantInt::get(Int32, 1));
  VPValue *ExtVPV2 = Plan.getOrAddLiveIn(AI);
  Args.push_back(ExtVPV1);
  Args.push_back(ExtVPV2);
  VPInstruction *I1 = new VPInstruction(Instruction::Add, {ExtVPV1, ExtVPV2});
  VPInstruction *I2 = new VPInstruction(Instruction::Mul, {I1, I1});
  VPBB1->appendRecipe(I1);
  VPBB1->appendRecipe(I2);

  // Check printing I1.
  {
    // Use EXPECT_EXIT to capture stderr and compare against expected output.
    //
    // Test VPValue::dump().
    VPValue *VPV = I1;
    EXPECT_EXIT(
        {
          VPV->dump();
          exit(0);
        },
        testing::ExitedWithCode(0), "EMIT vp<%1> = add ir<1>, ir<%a>");

    // Test VPRecipeBase::dump().
    VPRecipeBase *R = I1;
    EXPECT_EXIT(
        {
          R->dump();
          exit(0);
        },
        testing::ExitedWithCode(0), "EMIT vp<%1> = add ir<1>, ir<%a>");

    // Test VPDef::dump().
    VPDef *D = I1;
    EXPECT_EXIT(
        {
          D->dump();
          exit(0);
        },
        testing::ExitedWithCode(0), "EMIT vp<%1> = add ir<1>, ir<%a>");
  }
  // Check printing I2.
  {
    // Use EXPECT_EXIT to capture stderr and compare against expected output.
    //
    // Test VPValue::dump().
    VPValue *VPV = I2;
    EXPECT_EXIT(
        {
          VPV->dump();
          exit(0);
        },
        testing::ExitedWithCode(0), "EMIT vp<%2> = mul vp<%1>, vp<%1>");

    // Test VPRecipeBase::dump().
    VPRecipeBase *R = I2;
    EXPECT_EXIT(
        {
          R->dump();
          exit(0);
        },
        testing::ExitedWithCode(0), "EMIT vp<%2> = mul vp<%1>, vp<%1>");

    // Test VPDef::dump().
    VPDef *D = I2;
    EXPECT_EXIT(
        {
          D->dump();
          exit(0);
        },
        testing::ExitedWithCode(0), "EMIT vp<%2> = mul vp<%1>, vp<%1>");
  }
  delete AI;
}

TEST_F(VPRecipeTest, dumpRecipeUnnamedVPValuesNotInPlanOrBlock) {
  IntegerType *Int32 = IntegerType::get(C, 32);
  auto *AI = BinaryOperator::CreateAdd(PoisonValue::get(Int32),
                                       PoisonValue::get(Int32));
  AI->setName("a");
  VPValue *ExtVPV1 = getPlan().getOrAddLiveIn(ConstantInt::get(Int32, 1));
  VPValue *ExtVPV2 = getPlan().getOrAddLiveIn(AI);

  VPInstruction *I1 = new VPInstruction(Instruction::Add, {ExtVPV1, ExtVPV2});
  VPInstruction *I2 = new VPInstruction(Instruction::Mul, {I1, I1});

  // Check printing I1.
  {
    // Use EXPECT_EXIT to capture stderr and compare against expected output.
    //
    // Test VPValue::dump().
    VPValue *VPV = I1;
    EXPECT_EXIT(
        {
          VPV->dump();
          exit(0);
        },
        testing::ExitedWithCode(0), "EMIT <badref> = add ir<1>, ir<%a>");

    // Test VPRecipeBase::dump().
    VPRecipeBase *R = I1;
    EXPECT_EXIT(
        {
          R->dump();
          exit(0);
        },
        testing::ExitedWithCode(0), "EMIT <badref> = add ir<1>, ir<%a>");

    // Test VPDef::dump().
    VPDef *D = I1;
    EXPECT_EXIT(
        {
          D->dump();
          exit(0);
        },
        testing::ExitedWithCode(0), "EMIT <badref> = add ir<1>, ir<%a>");
  }
  // Check printing I2.
  {
    // Use EXPECT_EXIT to capture stderr and compare against expected output.
    //
    // Test VPValue::dump().
    VPValue *VPV = I2;
    EXPECT_EXIT(
        {
          VPV->dump();
          exit(0);
        },
        testing::ExitedWithCode(0), "EMIT <badref> = mul <badref>, <badref>");

    // Test VPRecipeBase::dump().
    VPRecipeBase *R = I2;
    EXPECT_EXIT(
        {
          R->dump();
          exit(0);
        },
        testing::ExitedWithCode(0), "EMIT <badref> = mul <badref>, <badref>");

    // Test VPDef::dump().
    VPDef *D = I2;
    EXPECT_EXIT(
        {
          D->dump();
          exit(0);
        },
        testing::ExitedWithCode(0), "EMIT <badref> = mul <badref>, <badref>");
  }

  delete I2;
  delete I1;
  delete AI;
}

#endif

TEST_F(VPRecipeTest, CastVPReductionRecipeToVPUser) {
  IntegerType *Int32 = IntegerType::get(C, 32);
  auto *Add = BinaryOperator::CreateAdd(PoisonValue::get(Int32),
                                        PoisonValue::get(Int32));
  VPValue *ChainOp = getPlan().getOrAddLiveIn(ConstantInt::get(Int32, 1));
  VPValue *VecOp = getPlan().getOrAddLiveIn(ConstantInt::get(Int32, 2));
  VPValue *CondOp = getPlan().getOrAddLiveIn(ConstantInt::get(Int32, 3));
  VPReductionRecipe Recipe(RecurKind::Add, FastMathFlags(), Add, ChainOp,
                           CondOp, VecOp, false);
  EXPECT_TRUE(isa<VPUser>(&Recipe));
  VPRecipeBase *BaseR = &Recipe;
  EXPECT_TRUE(isa<VPUser>(BaseR));
  delete Add;
}

TEST_F(VPRecipeTest, CastVPReductionEVLRecipeToVPUser) {
  IntegerType *Int32 = IntegerType::get(C, 32);
  auto *Add = BinaryOperator::CreateAdd(PoisonValue::get(Int32),
                                        PoisonValue::get(Int32));
  VPValue *ChainOp = getPlan().getOrAddLiveIn(ConstantInt::get(Int32, 1));
  VPValue *VecOp = getPlan().getOrAddLiveIn(ConstantInt::get(Int32, 2));
  VPValue *CondOp = getPlan().getOrAddLiveIn(ConstantInt::get(Int32, 3));
  VPReductionRecipe Recipe(RecurKind::Add, FastMathFlags(), Add, ChainOp,
                           CondOp, VecOp, false);
  VPValue *EVL = getPlan().getOrAddLiveIn(ConstantInt::get(Int32, 0));
  VPReductionEVLRecipe EVLRecipe(Recipe, *EVL, CondOp);
  EXPECT_TRUE(isa<VPUser>(&EVLRecipe));
  VPRecipeBase *BaseR = &EVLRecipe;
  EXPECT_TRUE(isa<VPUser>(BaseR));
  delete Add;
}
} // namespace

struct VPDoubleValueDef : public VPRecipeBase {
  VPDoubleValueDef(ArrayRef<VPValue *> Operands) : VPRecipeBase(99, Operands) {
    new VPValue(nullptr, this);
    new VPValue(nullptr, this);
  }

  VPRecipeBase *clone() override { return nullptr; }

  void execute(struct VPTransformState &State) override {}
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override {}
#endif
};

namespace {

TEST(VPDoubleValueDefTest, traverseUseLists) {
  // Check that the def-use chains of a multi-def can be traversed in both
  // directions.

  // Create a new VPDef which defines 2 values and has 2 operands.
  VPInstruction Op0(20, {});
  VPInstruction Op1(30, {});
  VPDoubleValueDef DoubleValueDef({&Op0, &Op1});

  // Create a new users of the defined values.
  VPInstruction I1(
      1, {DoubleValueDef.getVPValue(0), DoubleValueDef.getVPValue(1)});
  VPInstruction I2(2, {DoubleValueDef.getVPValue(0)});
  VPInstruction I3(3, {DoubleValueDef.getVPValue(1)});

  // Check operands of the VPDef (traversing upwards).
  SmallVector<VPValue *, 4> DoubleOperands(DoubleValueDef.op_begin(),
                                           DoubleValueDef.op_end());
  EXPECT_EQ(2u, DoubleOperands.size());
  EXPECT_EQ(&Op0, DoubleOperands[0]);
  EXPECT_EQ(&Op1, DoubleOperands[1]);

  // Check users of the defined values (traversing downwards).
  SmallVector<VPUser *, 4> DoubleValueDefV0Users(
      DoubleValueDef.getVPValue(0)->user_begin(),
      DoubleValueDef.getVPValue(0)->user_end());
  EXPECT_EQ(2u, DoubleValueDefV0Users.size());
  EXPECT_EQ(&I1, DoubleValueDefV0Users[0]);
  EXPECT_EQ(&I2, DoubleValueDefV0Users[1]);

  SmallVector<VPUser *, 4> DoubleValueDefV1Users(
      DoubleValueDef.getVPValue(1)->user_begin(),
      DoubleValueDef.getVPValue(1)->user_end());
  EXPECT_EQ(2u, DoubleValueDefV1Users.size());
  EXPECT_EQ(&I1, DoubleValueDefV1Users[0]);
  EXPECT_EQ(&I3, DoubleValueDefV1Users[1]);

  // Now check that we can get the right VPDef for each defined value.
  EXPECT_EQ(&DoubleValueDef, I1.getOperand(0)->getDefiningRecipe());
  EXPECT_EQ(&DoubleValueDef, I1.getOperand(1)->getDefiningRecipe());
  EXPECT_EQ(&DoubleValueDef, I2.getOperand(0)->getDefiningRecipe());
  EXPECT_EQ(&DoubleValueDef, I3.getOperand(0)->getDefiningRecipe());
}

TEST_F(VPRecipeTest, CastToVPSingleDefRecipe) {
  IntegerType *Int32 = IntegerType::get(C, 32);
  VPValue *Start = getPlan().getOrAddLiveIn(ConstantInt::get(Int32, 0));
  VPEVLBasedIVPHIRecipe R(Start, {});
  VPRecipeBase *B = &R;
  EXPECT_TRUE(isa<VPSingleDefRecipe>(B));
  // TODO: check other VPSingleDefRecipes.
}

} // namespace
} // namespace llvm

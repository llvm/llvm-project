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

TEST(VPInstructionTest, insertBefore) {
  VPInstruction *I1 = new VPInstruction(0, {});
  VPInstruction *I2 = new VPInstruction(1, {});
  VPInstruction *I3 = new VPInstruction(2, {});

  VPBasicBlock VPBB1;
  VPBB1.appendRecipe(I1);

  I2->insertBefore(I1);
  CHECK_ITERATOR(VPBB1, I2, I1);

  I3->insertBefore(I2);
  CHECK_ITERATOR(VPBB1, I3, I2, I1);
}

TEST(VPInstructionTest, eraseFromParent) {
  VPInstruction *I1 = new VPInstruction(0, {});
  VPInstruction *I2 = new VPInstruction(1, {});
  VPInstruction *I3 = new VPInstruction(2, {});

  VPBasicBlock VPBB1;
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

TEST(VPInstructionTest, moveAfter) {
  VPInstruction *I1 = new VPInstruction(0, {});
  VPInstruction *I2 = new VPInstruction(1, {});
  VPInstruction *I3 = new VPInstruction(2, {});

  VPBasicBlock VPBB1;
  VPBB1.appendRecipe(I1);
  VPBB1.appendRecipe(I2);
  VPBB1.appendRecipe(I3);

  I1->moveAfter(I2);

  CHECK_ITERATOR(VPBB1, I2, I1, I3);

  VPInstruction *I4 = new VPInstruction(4, {});
  VPInstruction *I5 = new VPInstruction(5, {});
  VPBasicBlock VPBB2;
  VPBB2.appendRecipe(I4);
  VPBB2.appendRecipe(I5);

  I3->moveAfter(I4);

  CHECK_ITERATOR(VPBB1, I2, I1);
  CHECK_ITERATOR(VPBB2, I4, I3, I5);
  EXPECT_EQ(I3->getParent(), I4->getParent());
}

TEST(VPInstructionTest, moveBefore) {
  VPInstruction *I1 = new VPInstruction(0, {});
  VPInstruction *I2 = new VPInstruction(1, {});
  VPInstruction *I3 = new VPInstruction(2, {});

  VPBasicBlock VPBB1;
  VPBB1.appendRecipe(I1);
  VPBB1.appendRecipe(I2);
  VPBB1.appendRecipe(I3);

  I1->moveBefore(VPBB1, I3->getIterator());

  CHECK_ITERATOR(VPBB1, I2, I1, I3);

  VPInstruction *I4 = new VPInstruction(4, {});
  VPInstruction *I5 = new VPInstruction(5, {});
  VPBasicBlock VPBB2;
  VPBB2.appendRecipe(I4);
  VPBB2.appendRecipe(I5);

  I3->moveBefore(VPBB2, I4->getIterator());

  CHECK_ITERATOR(VPBB1, I2, I1);
  CHECK_ITERATOR(VPBB2, I3, I4, I5);
  EXPECT_EQ(I3->getParent(), I4->getParent());

  VPBasicBlock VPBB3;

  I4->moveBefore(VPBB3, VPBB3.end());

  CHECK_ITERATOR(VPBB1, I2, I1);
  CHECK_ITERATOR(VPBB2, I3, I5);
  CHECK_ITERATOR(VPBB3, I4);
  EXPECT_EQ(&VPBB3, I4->getParent());
}

TEST(VPInstructionTest, setOperand) {
  VPValue *VPV1 = new VPValue();
  VPValue *VPV2 = new VPValue();
  VPInstruction *I1 = new VPInstruction(0, {VPV1, VPV2});
  EXPECT_EQ(1u, VPV1->getNumUsers());
  EXPECT_EQ(I1, *VPV1->user_begin());
  EXPECT_EQ(1u, VPV2->getNumUsers());
  EXPECT_EQ(I1, *VPV2->user_begin());

  // Replace operand 0 (VPV1) with VPV3.
  VPValue *VPV3 = new VPValue();
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
  VPValue *VPV4 = new VPValue();
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
  delete VPV1;
  delete VPV2;
  delete VPV3;
  delete VPV4;
}

TEST(VPInstructionTest, replaceAllUsesWith) {
  VPValue *VPV1 = new VPValue();
  VPValue *VPV2 = new VPValue();
  VPInstruction *I1 = new VPInstruction(0, {VPV1, VPV2});

  // Replace all uses of VPV1 with VPV3.
  VPValue *VPV3 = new VPValue();
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
  delete VPV1;
  delete VPV2;
  delete VPV3;
}

TEST(VPInstructionTest, releaseOperandsAtDeletion) {
  VPValue *VPV1 = new VPValue();
  VPValue *VPV2 = new VPValue();
  VPInstruction *I1 = new VPInstruction(0, {VPV1, VPV2});

  EXPECT_EQ(1u, VPV1->getNumUsers());
  EXPECT_EQ(I1, *VPV1->user_begin());
  EXPECT_EQ(1u, VPV2->getNumUsers());
  EXPECT_EQ(I1, *VPV2->user_begin());

  delete I1;

  EXPECT_EQ(0u, VPV1->getNumUsers());
  EXPECT_EQ(0u, VPV2->getNumUsers());

  delete VPV1;
  delete VPV2;
}
TEST(VPBasicBlockTest, getPlan) {
  {
    VPBasicBlock *VPPH = new VPBasicBlock("ph");
    VPBasicBlock *VPBB1 = new VPBasicBlock();
    VPBasicBlock *VPBB2 = new VPBasicBlock();
    VPBasicBlock *VPBB3 = new VPBasicBlock();
    VPBasicBlock *VPBB4 = new VPBasicBlock();

    //     VPBB1
    //     /   \
    // VPBB2  VPBB3
    //    \    /
    //    VPBB4
    VPBlockUtils::connectBlocks(VPBB1, VPBB2);
    VPBlockUtils::connectBlocks(VPBB1, VPBB3);
    VPBlockUtils::connectBlocks(VPBB2, VPBB4);
    VPBlockUtils::connectBlocks(VPBB3, VPBB4);

    auto TC = std::make_unique<VPValue>();
    VPlan Plan(VPPH, &*TC, VPBB1);

    EXPECT_EQ(&Plan, VPBB1->getPlan());
    EXPECT_EQ(&Plan, VPBB2->getPlan());
    EXPECT_EQ(&Plan, VPBB3->getPlan());
    EXPECT_EQ(&Plan, VPBB4->getPlan());
  }

  {
    VPBasicBlock *VPPH = new VPBasicBlock("ph");
    // VPBasicBlock is the entry into the VPlan, followed by a region.
    VPBasicBlock *R1BB1 = new VPBasicBlock();
    VPBasicBlock *R1BB2 = new VPBasicBlock();
    VPRegionBlock *R1 = new VPRegionBlock(R1BB1, R1BB2, "R1");
    VPBlockUtils::connectBlocks(R1BB1, R1BB2);

    VPBasicBlock *VPBB1 = new VPBasicBlock();
    VPBlockUtils::connectBlocks(VPBB1, R1);

    auto TC = std::make_unique<VPValue>();
    VPlan Plan(VPPH, &*TC, VPBB1);

    EXPECT_EQ(&Plan, VPBB1->getPlan());
    EXPECT_EQ(&Plan, R1->getPlan());
    EXPECT_EQ(&Plan, R1BB1->getPlan());
    EXPECT_EQ(&Plan, R1BB2->getPlan());
  }

  {
    VPBasicBlock *VPPH = new VPBasicBlock("ph");

    VPBasicBlock *R1BB1 = new VPBasicBlock();
    VPBasicBlock *R1BB2 = new VPBasicBlock();
    VPRegionBlock *R1 = new VPRegionBlock(R1BB1, R1BB2, "R1");
    VPBlockUtils::connectBlocks(R1BB1, R1BB2);

    VPBasicBlock *R2BB1 = new VPBasicBlock();
    VPBasicBlock *R2BB2 = new VPBasicBlock();
    VPRegionBlock *R2 = new VPRegionBlock(R2BB1, R2BB2, "R2");
    VPBlockUtils::connectBlocks(R2BB1, R2BB2);

    VPBasicBlock *VPBB1 = new VPBasicBlock();
    VPBlockUtils::connectBlocks(VPBB1, R1);
    VPBlockUtils::connectBlocks(VPBB1, R2);

    VPBasicBlock *VPBB2 = new VPBasicBlock();
    VPBlockUtils::connectBlocks(R1, VPBB2);
    VPBlockUtils::connectBlocks(R2, VPBB2);

    auto TC = std::make_unique<VPValue>();
    VPlan Plan(VPPH, &*TC, VPBB1);

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

TEST(VPBasicBlockTest, TraversingIteratorTest) {
  {
    // VPBasicBlocks only
    //     VPBB1
    //     /   \
    // VPBB2  VPBB3
    //    \    /
    //    VPBB4
    //
    VPBasicBlock *VPPH = new VPBasicBlock("ph");
    VPBasicBlock *VPBB1 = new VPBasicBlock();
    VPBasicBlock *VPBB2 = new VPBasicBlock();
    VPBasicBlock *VPBB3 = new VPBasicBlock();
    VPBasicBlock *VPBB4 = new VPBasicBlock();

    VPBlockUtils::connectBlocks(VPBB1, VPBB2);
    VPBlockUtils::connectBlocks(VPBB1, VPBB3);
    VPBlockUtils::connectBlocks(VPBB2, VPBB4);
    VPBlockUtils::connectBlocks(VPBB3, VPBB4);

    VPBlockDeepTraversalWrapper<const VPBlockBase *> Start(VPBB1);
    SmallVector<const VPBlockBase *> FromIterator(depth_first(Start));
    EXPECT_EQ(4u, FromIterator.size());
    EXPECT_EQ(VPBB1, FromIterator[0]);
    EXPECT_EQ(VPBB2, FromIterator[1]);

    // Use Plan to properly clean up created blocks.
    auto TC = std::make_unique<VPValue>();
    VPlan Plan(VPPH, &*TC, VPBB1);
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
    VPBasicBlock *VPPH = new VPBasicBlock("ph");
    VPBasicBlock *VPBB0 = new VPBasicBlock("VPBB0");
    VPBasicBlock *R1BB1 = new VPBasicBlock();
    VPBasicBlock *R1BB2 = new VPBasicBlock();
    VPBasicBlock *R1BB3 = new VPBasicBlock();
    VPBasicBlock *R1BB4 = new VPBasicBlock();
    VPRegionBlock *R1 = new VPRegionBlock(R1BB1, R1BB4, "R1");
    R1BB2->setParent(R1);
    R1BB3->setParent(R1);
    VPBlockUtils::connectBlocks(VPBB0, R1);
    VPBlockUtils::connectBlocks(R1BB1, R1BB2);
    VPBlockUtils::connectBlocks(R1BB1, R1BB3);
    VPBlockUtils::connectBlocks(R1BB2, R1BB4);
    VPBlockUtils::connectBlocks(R1BB3, R1BB4);
    // Cycle.
    VPBlockUtils::connectBlocks(R1BB3, R1BB3);

    VPBasicBlock *R2BB1 = new VPBasicBlock();
    VPBasicBlock *R2BB2 = new VPBasicBlock();
    VPRegionBlock *R2 = new VPRegionBlock(R2BB1, R2BB2, "R2");
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

    // Use Plan to properly clean up created blocks.
    auto TC = std::make_unique<VPValue>();
    VPlan Plan(VPPH, &*TC, VPBB0);
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
    VPBasicBlock *VPPH = new VPBasicBlock("ph");
    VPBasicBlock *R1BB1 = new VPBasicBlock("R1BB1");
    VPBasicBlock *R1BB2 = new VPBasicBlock("R1BB2");
    VPBasicBlock *R1BB3 = new VPBasicBlock("R1BB3");
    VPRegionBlock *R1 = new VPRegionBlock(R1BB1, R1BB3, "R1");

    VPBasicBlock *R2BB1 = new VPBasicBlock("R2BB1");
    VPBasicBlock *R2BB2 = new VPBasicBlock("R2BB2");
    VPBasicBlock *R2BB3 = new VPBasicBlock("R2BB3");
    VPRegionBlock *R2 = new VPRegionBlock(R2BB1, R2BB3, "R2");
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

    VPBasicBlock *VPBB1 = new VPBasicBlock("VPBB1");
    VPBlockUtils::connectBlocks(VPBB1, R1);
    VPBasicBlock *VPBB2 = new VPBasicBlock("VPBB2");
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

    // Use Plan to properly clean up created blocks.
    auto TC = std::make_unique<VPValue>();
    VPlan Plan(VPPH, &*TC, VPBB1);
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
    VPBasicBlock *VPPH = new VPBasicBlock("ph");
    VPBasicBlock *R2BB1 = new VPBasicBlock("R2BB1");
    VPBasicBlock *R2BB2 = new VPBasicBlock("R2BB2");
    VPRegionBlock *R2 = new VPRegionBlock(R2BB1, R2BB2, "R2");
    VPBlockUtils::connectBlocks(R2BB1, R2BB2);

    VPRegionBlock *R1 = new VPRegionBlock(R2, R2, "R1");
    R2->setParent(R1);

    VPBasicBlock *VPBB1 = new VPBasicBlock("VPBB1");
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

    // Use Plan to properly clean up created blocks.
    auto TC = std::make_unique<VPValue>();
    VPlan Plan(VPPH, &*TC, VPBB1);
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
    VPBasicBlock *VPPH = new VPBasicBlock("ph");
    VPBasicBlock *R3BB1 = new VPBasicBlock("R3BB1");
    VPRegionBlock *R3 = new VPRegionBlock(R3BB1, R3BB1, "R3");

    VPBasicBlock *R2BB1 = new VPBasicBlock("R2BB1");
    VPRegionBlock *R2 = new VPRegionBlock(R2BB1, R3, "R2");
    R3->setParent(R2);
    VPBlockUtils::connectBlocks(R2BB1, R3);

    VPRegionBlock *R1 = new VPRegionBlock(R2, R2, "R1");
    R2->setParent(R1);

    VPBasicBlock *VPBB1 = new VPBasicBlock("VPBB1");
    VPBasicBlock *VPBB2 = new VPBasicBlock("VPBB2");
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

    // Use Plan to properly clean up created blocks.
    auto TC = std::make_unique<VPValue>();
    VPlan Plan(VPPH, &*TC, VPBB1);
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
TEST(VPBasicBlockTest, print) {
  VPInstruction *TC = new VPInstruction(Instruction::Add, {});
  VPBasicBlock *VPBB0 = new VPBasicBlock("preheader");
  VPBB0->appendRecipe(TC);

  VPInstruction *I1 = new VPInstruction(Instruction::Add, {});
  VPInstruction *I2 = new VPInstruction(Instruction::Sub, {I1});
  VPInstruction *I3 = new VPInstruction(Instruction::Br, {I1, I2});

  VPBasicBlock *VPBB1 = new VPBasicBlock();
  VPBB1->appendRecipe(I1);
  VPBB1->appendRecipe(I2);
  VPBB1->appendRecipe(I3);
  VPBB1->setName("bb1");

  VPInstruction *I4 = new VPInstruction(Instruction::Mul, {I2, I1});
  VPInstruction *I5 = new VPInstruction(Instruction::Ret, {I4});
  VPBasicBlock *VPBB2 = new VPBasicBlock();
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
    OS.flush();
    EXPECT_EQ("EMIT br <badref>, <badref>", I3Dump);
  }

  VPlan Plan(VPBB0, TC, VPBB1);
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
    "  EMIT vp\<%1\> = add\l" +
    "No successors\l"
  ]
  N1 [label =
    "bb1:\l" +
    "  EMIT vp\<%2\> = add\l" +
    "  EMIT vp\<%3\> = sub vp\<%2\>\l" +
    "  EMIT br vp\<%2\>, vp\<%3\>\l" +
    "Successor(s): bb2\l"
  ]
  N1 -> N2 [ label=""]
  N2 [label =
    "bb2:\l" +
    "  EMIT vp\<%5\> = mul vp\<%3\>, vp\<%2\>\l" +
    "  EMIT ret vp\<%5\>\l" +
    "No successors\l"
  ]
}
)";
  EXPECT_EQ(ExpectedStr, FullDump);

  const char *ExpectedBlock1Str = R"(bb1:
  EMIT vp<%2> = add
  EMIT vp<%3> = sub vp<%2>
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
  EMIT ret vp<%5>
No successors
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
    OS.flush();
    EXPECT_EQ("EMIT br vp<%2>, vp<%3>", I3Dump);
  }

  {
    std::string I4Dump;
    raw_string_ostream OS(I4Dump);
    OS << *I4;
    OS.flush();
    EXPECT_EQ("EMIT vp<%5> = mul vp<%3>, vp<%2>", I4Dump);
  }
}

TEST(VPBasicBlockTest, printPlanWithVFsAndUFs) {

  VPInstruction *TC = new VPInstruction(Instruction::Sub, {});
  VPBasicBlock *VPBB0 = new VPBasicBlock("preheader");
  VPBB0->appendRecipe(TC);

  VPInstruction *I1 = new VPInstruction(Instruction::Add, {});
  VPBasicBlock *VPBB1 = new VPBasicBlock();
  VPBB1->appendRecipe(I1);
  VPBB1->setName("bb1");

  VPlan Plan(VPBB0, TC, VPBB1);
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
No successors

bb1:
  EMIT vp<%2> = add
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
No successors

bb1:
  EMIT vp<%2> = add
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
No successors

bb1:
  EMIT vp<%2> = add
No successors
}
)";
    EXPECT_EQ(ExpectedStr, FullDump);
  }
}
#endif

TEST(VPRecipeTest, CastVPInstructionToVPUser) {
  VPValue Op1;
  VPValue Op2;
  VPInstruction Recipe(Instruction::Add, {&Op1, &Op2});
  EXPECT_TRUE(isa<VPUser>(&Recipe));
  VPRecipeBase *BaseR = &Recipe;
  EXPECT_TRUE(isa<VPUser>(BaseR));
  EXPECT_EQ(&Recipe, BaseR);
}

TEST(VPRecipeTest, CastVPWidenRecipeToVPUser) {
  LLVMContext C;

  IntegerType *Int32 = IntegerType::get(C, 32);
  auto *AI =
      BinaryOperator::CreateAdd(UndefValue::get(Int32), UndefValue::get(Int32));
  VPValue Op1;
  VPValue Op2;
  SmallVector<VPValue *, 2> Args;
  Args.push_back(&Op1);
  Args.push_back(&Op1);
  VPWidenRecipe WidenR(*AI, make_range(Args.begin(), Args.end()));
  EXPECT_TRUE(isa<VPUser>(&WidenR));
  VPRecipeBase *WidenRBase = &WidenR;
  EXPECT_TRUE(isa<VPUser>(WidenRBase));
  EXPECT_EQ(&WidenR, WidenRBase);
  delete AI;
}

TEST(VPRecipeTest, CastVPWidenCallRecipeToVPUserAndVPDef) {
  LLVMContext C;

  IntegerType *Int32 = IntegerType::get(C, 32);
  FunctionType *FTy = FunctionType::get(Int32, false);
  Function *Fn = Function::Create(FTy, GlobalValue::ExternalLinkage, 0);
  auto *Call = CallInst::Create(FTy, Fn);
  VPValue Op1;
  VPValue Op2;
  VPValue CalledFn(Call->getCalledFunction());
  SmallVector<VPValue *, 2> Args;
  Args.push_back(&Op1);
  Args.push_back(&Op2);
  Args.push_back(&CalledFn);
  VPWidenCallRecipe Recipe(Call, make_range(Args.begin(), Args.end()), false);
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

TEST(VPRecipeTest, CastVPWidenSelectRecipeToVPUserAndVPDef) {
  LLVMContext C;

  IntegerType *Int1 = IntegerType::get(C, 1);
  IntegerType *Int32 = IntegerType::get(C, 32);
  auto *SelectI = SelectInst::Create(
      UndefValue::get(Int1), UndefValue::get(Int32), UndefValue::get(Int32));
  VPValue Op1;
  VPValue Op2;
  VPValue Op3;
  SmallVector<VPValue *, 4> Args;
  Args.push_back(&Op1);
  Args.push_back(&Op2);
  Args.push_back(&Op3);
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

TEST(VPRecipeTest, CastVPWidenGEPRecipeToVPUserAndVPDef) {
  LLVMContext C;

  IntegerType *Int32 = IntegerType::get(C, 32);
  PointerType *Int32Ptr = PointerType::get(Int32, 0);
  auto *GEP = GetElementPtrInst::Create(Int32, UndefValue::get(Int32Ptr),
                                        UndefValue::get(Int32));
  VPValue Op1;
  VPValue Op2;
  SmallVector<VPValue *, 4> Args;
  Args.push_back(&Op1);
  Args.push_back(&Op2);
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

TEST(VPRecipeTest, CastVPBlendRecipeToVPUser) {
  LLVMContext C;

  IntegerType *Int32 = IntegerType::get(C, 32);
  auto *Phi = PHINode::Create(Int32, 1);
  VPValue I1;
  VPValue I2;
  VPValue M2;
  SmallVector<VPValue *, 4> Args;
  Args.push_back(&I1);
  Args.push_back(&I2);
  Args.push_back(&M2);
  VPBlendRecipe Recipe(Phi, Args);
  EXPECT_TRUE(isa<VPUser>(&Recipe));
  VPRecipeBase *BaseR = &Recipe;
  EXPECT_TRUE(isa<VPUser>(BaseR));
  delete Phi;
}

TEST(VPRecipeTest, CastVPInterleaveRecipeToVPUser) {
  LLVMContext C;

  VPValue Addr;
  VPValue Mask;
  InterleaveGroup<Instruction> IG(4, false, Align(4));
  VPInterleaveRecipe Recipe(&IG, &Addr, {}, &Mask, false);
  EXPECT_TRUE(isa<VPUser>(&Recipe));
  VPRecipeBase *BaseR = &Recipe;
  EXPECT_TRUE(isa<VPUser>(BaseR));
  EXPECT_EQ(&Recipe, BaseR);
}

TEST(VPRecipeTest, CastVPReplicateRecipeToVPUser) {
  LLVMContext C;

  VPValue Op1;
  VPValue Op2;
  SmallVector<VPValue *, 4> Args;
  Args.push_back(&Op1);
  Args.push_back(&Op2);

  IntegerType *Int32 = IntegerType::get(C, 32);
  FunctionType *FTy = FunctionType::get(Int32, false);
  auto *Call = CallInst::Create(FTy, UndefValue::get(FTy));
  VPReplicateRecipe Recipe(Call, make_range(Args.begin(), Args.end()), true);
  EXPECT_TRUE(isa<VPUser>(&Recipe));
  VPRecipeBase *BaseR = &Recipe;
  EXPECT_TRUE(isa<VPUser>(BaseR));
  delete Call;
}

TEST(VPRecipeTest, CastVPBranchOnMaskRecipeToVPUser) {
  LLVMContext C;

  VPValue Mask;
  VPBranchOnMaskRecipe Recipe(&Mask);
  EXPECT_TRUE(isa<VPUser>(&Recipe));
  VPRecipeBase *BaseR = &Recipe;
  EXPECT_TRUE(isa<VPUser>(BaseR));
  EXPECT_EQ(&Recipe, BaseR);
}

TEST(VPRecipeTest, CastVPWidenMemoryRecipeToVPUserAndVPDef) {
  LLVMContext C;

  IntegerType *Int32 = IntegerType::get(C, 32);
  PointerType *Int32Ptr = PointerType::get(Int32, 0);
  auto *Load =
      new LoadInst(Int32, UndefValue::get(Int32Ptr), "", false, Align(1));
  VPValue Addr;
  VPValue Mask;
  VPWidenLoadRecipe Recipe(*Load, &Addr, &Mask, true, false, {});
  EXPECT_TRUE(isa<VPUser>(&Recipe));
  VPRecipeBase *BaseR = &Recipe;
  EXPECT_TRUE(isa<VPUser>(BaseR));
  EXPECT_EQ(&Recipe, BaseR);

  VPValue *VPV = Recipe.getVPSingleValue();
  EXPECT_TRUE(isa<VPRecipeBase>(VPV->getDefiningRecipe()));
  EXPECT_EQ(&Recipe, VPV->getDefiningRecipe());

  delete Load;
}

TEST(VPRecipeTest, MayHaveSideEffectsAndMayReadWriteMemory) {
  LLVMContext C;
  IntegerType *Int1 = IntegerType::get(C, 1);
  IntegerType *Int32 = IntegerType::get(C, 32);
  PointerType *Int32Ptr = PointerType::get(Int32, 0);

  {
    auto *AI = BinaryOperator::CreateAdd(UndefValue::get(Int32),
                                         UndefValue::get(Int32));
    VPValue Op1;
    VPValue Op2;
    SmallVector<VPValue *, 2> Args;
    Args.push_back(&Op1);
    Args.push_back(&Op1);
    VPWidenRecipe Recipe(*AI, make_range(Args.begin(), Args.end()));
    EXPECT_FALSE(Recipe.mayHaveSideEffects());
    EXPECT_FALSE(Recipe.mayReadFromMemory());
    EXPECT_FALSE(Recipe.mayWriteToMemory());
    EXPECT_FALSE(Recipe.mayReadOrWriteMemory());
    delete AI;
  }

  {
    auto *SelectI = SelectInst::Create(
        UndefValue::get(Int1), UndefValue::get(Int32), UndefValue::get(Int32));
    VPValue Op1;
    VPValue Op2;
    VPValue Op3;
    SmallVector<VPValue *, 4> Args;
    Args.push_back(&Op1);
    Args.push_back(&Op2);
    Args.push_back(&Op3);
    VPWidenSelectRecipe Recipe(*SelectI, make_range(Args.begin(), Args.end()));
    EXPECT_FALSE(Recipe.mayHaveSideEffects());
    EXPECT_FALSE(Recipe.mayReadFromMemory());
    EXPECT_FALSE(Recipe.mayWriteToMemory());
    EXPECT_FALSE(Recipe.mayReadOrWriteMemory());
    delete SelectI;
  }

  {
    auto *GEP = GetElementPtrInst::Create(Int32, UndefValue::get(Int32Ptr),
                                          UndefValue::get(Int32));
    VPValue Op1;
    VPValue Op2;
    SmallVector<VPValue *, 4> Args;
    Args.push_back(&Op1);
    Args.push_back(&Op2);
    VPWidenGEPRecipe Recipe(GEP, make_range(Args.begin(), Args.end()));
    EXPECT_FALSE(Recipe.mayHaveSideEffects());
    EXPECT_FALSE(Recipe.mayReadFromMemory());
    EXPECT_FALSE(Recipe.mayWriteToMemory());
    EXPECT_FALSE(Recipe.mayReadOrWriteMemory());
    delete GEP;
  }

  {
    VPValue Mask;
    VPBranchOnMaskRecipe Recipe(&Mask);
    EXPECT_TRUE(Recipe.mayHaveSideEffects());
    EXPECT_FALSE(Recipe.mayReadFromMemory());
    EXPECT_FALSE(Recipe.mayWriteToMemory());
    EXPECT_FALSE(Recipe.mayReadOrWriteMemory());
  }

  {
    VPValue ChainOp;
    VPValue VecOp;
    VPValue CondOp;
    VPReductionRecipe Recipe(RecurrenceDescriptor(), nullptr, &ChainOp, &CondOp,
                             &VecOp, false);
    EXPECT_FALSE(Recipe.mayHaveSideEffects());
    EXPECT_FALSE(Recipe.mayReadFromMemory());
    EXPECT_FALSE(Recipe.mayWriteToMemory());
    EXPECT_FALSE(Recipe.mayReadOrWriteMemory());
  }

  {
    auto *Load =
        new LoadInst(Int32, UndefValue::get(Int32Ptr), "", false, Align(1));
    VPValue Addr;
    VPValue Mask;
    VPWidenLoadRecipe Recipe(*Load, &Addr, &Mask, true, false, {});
    EXPECT_FALSE(Recipe.mayHaveSideEffects());
    EXPECT_TRUE(Recipe.mayReadFromMemory());
    EXPECT_FALSE(Recipe.mayWriteToMemory());
    EXPECT_TRUE(Recipe.mayReadOrWriteMemory());
    delete Load;
  }

  {
    auto *Store = new StoreInst(UndefValue::get(Int32),
                                UndefValue::get(Int32Ptr), false, Align(1));
    VPValue Addr;
    VPValue Mask;
    VPValue StoredV;
    VPWidenStoreRecipe Recipe(*Store, &Addr, &StoredV, &Mask, false, false, {});
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
    VPValue Op1;
    VPValue Op2;
    VPValue CalledFn(Call->getCalledFunction());
    SmallVector<VPValue *, 3> Args;
    Args.push_back(&Op1);
    Args.push_back(&Op2);
    Args.push_back(&CalledFn);
    VPWidenCallRecipe Recipe(Call, make_range(Args.begin(), Args.end()), false);
    EXPECT_TRUE(Recipe.mayHaveSideEffects());
    EXPECT_TRUE(Recipe.mayReadFromMemory());
    EXPECT_TRUE(Recipe.mayWriteToMemory());
    EXPECT_TRUE(Recipe.mayReadOrWriteMemory());
    delete Call;
    delete Fn;
  }

  {
    // Test for a call to a function without side-effects.
    LLVMContext C;
    Module M("", C);
    Function *TheFn = Intrinsic::getDeclaration(&M, Intrinsic::thread_pointer);

    auto *Call = CallInst::Create(TheFn->getFunctionType(), TheFn);
    VPValue Op1;
    VPValue Op2;
    VPValue CalledFn(TheFn);
    SmallVector<VPValue *, 3> Args;
    Args.push_back(&Op1);
    Args.push_back(&Op2);
    Args.push_back(&CalledFn);
    VPWidenCallRecipe Recipe(Call, make_range(Args.begin(), Args.end()), false);
    EXPECT_FALSE(Recipe.mayHaveSideEffects());
    EXPECT_FALSE(Recipe.mayReadFromMemory());
    EXPECT_FALSE(Recipe.mayWriteToMemory());
    EXPECT_FALSE(Recipe.mayReadOrWriteMemory());
    delete Call;
  }

  {
    VPValue Op1;
    VPValue Op2;
    InductionDescriptor IndDesc;
    VPScalarIVStepsRecipe Recipe(IndDesc, &Op1, &Op2);
    EXPECT_FALSE(Recipe.mayHaveSideEffects());
    EXPECT_FALSE(Recipe.mayReadFromMemory());
    EXPECT_FALSE(Recipe.mayWriteToMemory());
    EXPECT_FALSE(Recipe.mayReadOrWriteMemory());
  }

  // The initial implementation is conservative with respect to VPInstructions.
  {
    VPValue Op1;
    VPValue Op2;
    VPInstruction VPInst(Instruction::Add, {&Op1, &Op2});
    VPRecipeBase &Recipe = VPInst;
    EXPECT_TRUE(Recipe.mayHaveSideEffects());
    EXPECT_TRUE(Recipe.mayReadFromMemory());
    EXPECT_TRUE(Recipe.mayWriteToMemory());
    EXPECT_TRUE(Recipe.mayReadOrWriteMemory());
  }
  {
    VPValue Op1;
    VPPredInstPHIRecipe Recipe(&Op1);
    EXPECT_FALSE(Recipe.mayHaveSideEffects());
    EXPECT_FALSE(Recipe.mayReadFromMemory());
    EXPECT_FALSE(Recipe.mayWriteToMemory());
    EXPECT_FALSE(Recipe.mayReadOrWriteMemory());
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
TEST(VPRecipeTest, dumpRecipeInPlan) {
  VPBasicBlock *VPBB0 = new VPBasicBlock("preheader");
  VPBasicBlock *VPBB1 = new VPBasicBlock();
  VPlan Plan(VPBB0, VPBB1);

  LLVMContext C;

  IntegerType *Int32 = IntegerType::get(C, 32);
  auto *AI =
      BinaryOperator::CreateAdd(UndefValue::get(Int32), UndefValue::get(Int32));
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

TEST(VPRecipeTest, dumpRecipeUnnamedVPValuesInPlan) {
  VPBasicBlock *VPBB0 = new VPBasicBlock("preheader");
  VPBasicBlock *VPBB1 = new VPBasicBlock();
  VPlan Plan(VPBB0, VPBB1);

  LLVMContext C;

  IntegerType *Int32 = IntegerType::get(C, 32);
  auto *AI =
      BinaryOperator::CreateAdd(UndefValue::get(Int32), UndefValue::get(Int32));
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

TEST(VPRecipeTest, dumpRecipeUnnamedVPValuesNotInPlanOrBlock) {
  LLVMContext C;
  IntegerType *Int32 = IntegerType::get(C, 32);
  auto *AI =
      BinaryOperator::CreateAdd(UndefValue::get(Int32), UndefValue::get(Int32));
  AI->setName("a");
  VPValue *ExtVPV1 = new VPValue(ConstantInt::get(Int32, 1));
  VPValue *ExtVPV2 = new VPValue(AI);

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
  delete ExtVPV2;
  delete ExtVPV1;
  delete AI;
}

#endif

TEST(VPRecipeTest, CastVPReductionRecipeToVPUser) {
  LLVMContext C;

  VPValue ChainOp;
  VPValue VecOp;
  VPValue CondOp;
  VPReductionRecipe Recipe(RecurrenceDescriptor(), nullptr, &ChainOp, &CondOp,
                           &VecOp, false);
  EXPECT_TRUE(isa<VPUser>(&Recipe));
  VPRecipeBase *BaseR = &Recipe;
  EXPECT_TRUE(isa<VPUser>(BaseR));
}

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

TEST(VPRecipeTest, CastToVPSingleDefRecipe) {
  VPValue Start;
  VPEVLBasedIVPHIRecipe R(&Start, {});
  VPRecipeBase *B = &R;
  EXPECT_TRUE(isa<VPSingleDefRecipe>(B));
  // TODO: check other VPSingleDefRecipes.
}

} // namespace
} // namespace llvm

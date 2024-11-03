//===- llvm/unittests/Transforms/Vectorize/VPDomTreeTests.cpp - -----------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lib/Transforms/Vectorize/VPlan.h"
#include "../lib/Transforms/Vectorize/VPlanDominatorTree.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

TEST(VPDominatorTreeTest, DominanceNoRegionsTest) {
  //   R1 {
  //     VPBB1
  //     /   \
    // VPBB2  VPBB3
  //    \    /
  //    VPBB4
  //  }
  VPBasicBlock *VPBB1 = new VPBasicBlock("VPBB1");
  VPBasicBlock *VPBB2 = new VPBasicBlock("VPBB2");
  VPBasicBlock *VPBB3 = new VPBasicBlock("VPBB3");
  VPBasicBlock *VPBB4 = new VPBasicBlock("VPBB4");
  VPRegionBlock *R1 = new VPRegionBlock(VPBB1, VPBB4);
  VPBB2->setParent(R1);
  VPBB3->setParent(R1);

  VPBlockUtils::connectBlocks(VPBB1, VPBB2);
  VPBlockUtils::connectBlocks(VPBB1, VPBB3);
  VPBlockUtils::connectBlocks(VPBB2, VPBB4);
  VPBlockUtils::connectBlocks(VPBB3, VPBB4);

  VPlan Plan;
  Plan.setEntry(R1);
  VPDominatorTree VPDT;
  VPDT.recalculate(Plan);

  EXPECT_TRUE(VPDT.dominates(VPBB1, VPBB4));
  EXPECT_FALSE(VPDT.dominates(VPBB4, VPBB1));

  EXPECT_TRUE(VPDT.dominates(VPBB1, VPBB2));
  EXPECT_FALSE(VPDT.dominates(VPBB2, VPBB1));

  EXPECT_TRUE(VPDT.dominates(VPBB1, VPBB3));
  EXPECT_FALSE(VPDT.dominates(VPBB3, VPBB1));

  EXPECT_EQ(VPDT.findNearestCommonDominator(VPBB2, VPBB3), VPBB1);
  EXPECT_EQ(VPDT.findNearestCommonDominator(VPBB2, VPBB4), VPBB1);
  EXPECT_EQ(VPDT.findNearestCommonDominator(VPBB4, VPBB4), VPBB4);
}

static void
checkDomChildren(VPDominatorTree &VPDT, VPBlockBase *Src,
                 std::initializer_list<VPBlockBase *> ExpectedChildren) {
  SmallVector<VPDomTreeNode *> Children(VPDT.getNode(Src)->children());
  SmallVector<VPDomTreeNode *> ExpectedNodes;
  for (VPBlockBase *C : ExpectedChildren)
    ExpectedNodes.push_back(VPDT.getNode(C));

  EXPECT_EQ(Children, ExpectedNodes);
}

TEST(VPDominatorTreeTest, DominanceRegionsTest) {
  {
    // 2 consecutive regions.
    // R1 {
    //     \
    //     R1BB1     _
    //    /     \   / \
    //  R1BB2   R1BB3  |
    //    \      /  \_/
    //     R1BB4
    //  }
    //   |
    // R2 {
    //   \
    //    R2BB1
    //      |
    //    R2BB2
    // }
    //
    VPBasicBlock *R1BB1 = new VPBasicBlock();
    VPBasicBlock *R1BB2 = new VPBasicBlock();
    VPBasicBlock *R1BB3 = new VPBasicBlock();
    VPBasicBlock *R1BB4 = new VPBasicBlock();
    VPRegionBlock *R1 = new VPRegionBlock(R1BB1, R1BB4, "R1");
    R1BB2->setParent(R1);
    R1BB3->setParent(R1);
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

    VPlan Plan;
    Plan.setEntry(R1);
    VPDominatorTree VPDT;
    VPDT.recalculate(Plan);

    checkDomChildren(VPDT, R1, {R1BB1});
    checkDomChildren(VPDT, R1BB1, {R1BB2, R1BB4, R1BB3});
    checkDomChildren(VPDT, R1BB2, {});
    checkDomChildren(VPDT, R1BB3, {});
    checkDomChildren(VPDT, R1BB4, {R2});
    checkDomChildren(VPDT, R2, {R2BB1});
    checkDomChildren(VPDT, R2BB1, {R2BB2});

    EXPECT_TRUE(VPDT.dominates(R1, R2));
    EXPECT_FALSE(VPDT.dominates(R2, R1));

    EXPECT_TRUE(VPDT.dominates(R1BB1, R1BB4));
    EXPECT_FALSE(VPDT.dominates(R1BB4, R1BB1));

    EXPECT_TRUE(VPDT.dominates(R2BB1, R2BB2));
    EXPECT_FALSE(VPDT.dominates(R2BB2, R2BB1));

    EXPECT_TRUE(VPDT.dominates(R1BB1, R2BB1));
    EXPECT_FALSE(VPDT.dominates(R2BB1, R1BB1));

    EXPECT_TRUE(VPDT.dominates(R1BB4, R2BB1));
    EXPECT_FALSE(VPDT.dominates(R1BB3, R2BB1));

    EXPECT_TRUE(VPDT.dominates(R1, R2BB1));
    EXPECT_FALSE(VPDT.dominates(R2BB1, R1));
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
    //     R2BB2-/     |
    //        \        |
    //         R2BB3   |
    //   }            /
    //      \        /
    //        R1BB3
    //  }
    //   |
    //  VPBB2
    //
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

    VPlan Plan;
    Plan.setEntry(VPBB1);
    VPDominatorTree VPDT;
    VPDT.recalculate(Plan);

    checkDomChildren(VPDT, VPBB1, {R1});
    checkDomChildren(VPDT, R1, {R1BB1});
    checkDomChildren(VPDT, R1BB1, {R2, R1BB3, R1BB2});
    checkDomChildren(VPDT, R1BB2, {});
    checkDomChildren(VPDT, R2, {R2BB1});
    checkDomChildren(VPDT, R2BB1, {R2BB2});
    checkDomChildren(VPDT, R2BB2, {R2BB3});
    checkDomChildren(VPDT, R2BB3, {});
    checkDomChildren(VPDT, R1BB3, {VPBB2});
    checkDomChildren(VPDT, VPBB2, {});
  }
}

} // namespace
} // namespace llvm

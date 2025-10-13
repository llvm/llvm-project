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
#include "VPlanTestBase.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

using VPDominatorTreeTest = VPlanTestBase;

TEST_F(VPDominatorTreeTest, DominanceNoRegionsTest) {
  //   VPBB0
  //    |
  //   R1 {
  //     VPBB1
  //     /   \
    // VPBB2  VPBB3
  //    \    /
  //    VPBB4
  //  }
  VPlan &Plan = getPlan();
  VPBasicBlock *VPBB0 = Plan.getEntry();
  VPBasicBlock *VPBB1 = Plan.createVPBasicBlock("VPBB1");
  VPBasicBlock *VPBB2 = Plan.createVPBasicBlock("VPBB2");
  VPBasicBlock *VPBB3 = Plan.createVPBasicBlock("VPBB3");
  VPBasicBlock *VPBB4 = Plan.createVPBasicBlock("VPBB4");
  VPRegionBlock *R1 = Plan.createVPRegionBlock(VPBB1, VPBB4);
  VPBB2->setParent(R1);
  VPBB3->setParent(R1);

  VPBlockUtils::connectBlocks(VPBB0, R1);
  VPBlockUtils::connectBlocks(VPBB1, VPBB2);
  VPBlockUtils::connectBlocks(VPBB1, VPBB3);
  VPBlockUtils::connectBlocks(VPBB2, VPBB4);
  VPBlockUtils::connectBlocks(VPBB3, VPBB4);

  VPBlockUtils::connectBlocks(R1, Plan.getScalarHeader());

  VPDominatorTree VPDT(Plan);

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

TEST_F(VPDominatorTreeTest, DominanceRegionsTest) {
  {
    // 2 consecutive regions.
    // VPBB0
    //  |
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

    VPBlockUtils::connectBlocks(R2, Plan.getScalarHeader());
    VPDominatorTree VPDT(Plan);

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
    VPlan &Plan = getPlan();
    VPBasicBlock *R1BB1 = Plan.createVPBasicBlock("R1BB1");
    VPBasicBlock *R1BB2 = Plan.createVPBasicBlock("R1BB2");
    VPBasicBlock *R1BB3 = Plan.createVPBasicBlock("R1BB3");
    VPRegionBlock *R1 = Plan.createVPRegionBlock(R1BB1, R1BB3, "R1");

    VPBasicBlock *R2BB1 = Plan.createVPBasicBlock("R2BB1");
    VPBasicBlock *R2BB2 = Plan.createVPBasicBlock("R2BB2");
    VPBasicBlock *R2BB3 = Plan.createVPBasicBlock("R2BB#");
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

    VPBlockUtils::connectBlocks(VPBB2, Plan.getScalarHeader());
    VPDominatorTree VPDT(Plan);

    checkDomChildren(VPDT, VPBB1, {R1});
    checkDomChildren(VPDT, R1, {R1BB1});
    checkDomChildren(VPDT, R1BB1, {R2, R1BB3, R1BB2});
    checkDomChildren(VPDT, R1BB2, {});
    checkDomChildren(VPDT, R2, {R2BB1});
    checkDomChildren(VPDT, R2BB1, {R2BB2});
    checkDomChildren(VPDT, R2BB2, {R2BB3});
    checkDomChildren(VPDT, R2BB3, {});
    checkDomChildren(VPDT, R1BB3, {VPBB2});
    checkDomChildren(VPDT, VPBB2, {Plan.getScalarHeader()});
  }
}

} // namespace
} // namespace llvm

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
  VPDT.recalculate(*R1);

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
} // namespace
} // namespace llvm

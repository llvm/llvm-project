//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lib/Transforms/Vectorize/VPlan.h"

#include "VPlanTestBase.h"
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/DominanceFrontierImpl.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

using VPPostDomFrontierTest = VPlanTestBase;

TEST_F(VPPostDomFrontierTest, SingleExitTest) {
  //   VPBB0
  //  /    \
  // VBBB1 VBB2->VPBB3
  //   /  \ |     /
  // VPBB4 VPBB5 /
  //    \  /    /
  //    VPBB6  /
  //      |   /
  //     VPBB7
  VPlan &Plan = getPlan();
  VPBasicBlock *VPBB0 = Plan.getEntry();
  VPBasicBlock *VPBB1 = Plan.createVPBasicBlock("VPBB1");
  VPBasicBlock *VPBB2 = Plan.createVPBasicBlock("VPBB2");
  VPBasicBlock *VPBB3 = Plan.createVPBasicBlock("VPBB3");
  VPBasicBlock *VPBB4 = Plan.createVPBasicBlock("VPBB4");
  VPBasicBlock *VPBB5 = Plan.createVPBasicBlock("VPBB5");
  VPBasicBlock *VPBB6 = Plan.createVPBasicBlock("VPBB6");
  VPBasicBlock *VPBB7 = Plan.createVPBasicBlock("VPBB7");

  VPBlockUtils::connectBlocks(VPBB0, VPBB1);
  VPBlockUtils::connectBlocks(VPBB0, VPBB2);
  VPBlockUtils::connectBlocks(VPBB1, VPBB4);
  VPBlockUtils::connectBlocks(VPBB1, VPBB5);
  VPBlockUtils::connectBlocks(VPBB2, VPBB5);
  VPBlockUtils::connectBlocks(VPBB2, VPBB3);
  VPBlockUtils::connectBlocks(VPBB3, VPBB7);
  VPBlockUtils::connectBlocks(VPBB4, VPBB6);
  VPBlockUtils::connectBlocks(VPBB5, VPBB6);
  VPBlockUtils::connectBlocks(VPBB6, VPBB7);

  PostDomTreeBase<VPBlockBase> VPPDT;
  VPPDT.recalculate(Plan);
  DominanceFrontierBase<VPBlockBase, true> VPPDF;
  VPPDF.analyze(VPPDT);

  EXPECT_TRUE(VPPDF.find(VPBB0) != VPPDF.end());
  EXPECT_TRUE(VPPDF.find(VPBB1) != VPPDF.end());
  EXPECT_TRUE(VPPDF.find(VPBB2) != VPPDF.end());
  EXPECT_TRUE(VPPDF.find(VPBB3) != VPPDF.end());
  EXPECT_TRUE(VPPDF.find(VPBB4) != VPPDF.end());
  EXPECT_TRUE(VPPDF.find(VPBB5) != VPPDF.end());
  EXPECT_TRUE(VPPDF.find(VPBB6) != VPPDF.end());
  EXPECT_TRUE(VPPDF.find(VPBB7) != VPPDF.end());

  auto F0 = VPPDF.find(VPBB0)->second;
  auto F1 = VPPDF.find(VPBB1)->second;
  auto F2 = VPPDF.find(VPBB2)->second;
  auto F3 = VPPDF.find(VPBB3)->second;
  auto F4 = VPPDF.find(VPBB4)->second;
  auto F5 = VPPDF.find(VPBB5)->second;
  auto F6 = VPPDF.find(VPBB6)->second;
  auto F7 = VPPDF.find(VPBB7)->second;

  EXPECT_EQ(F0.size(), 0u);
  EXPECT_EQ(F1.size(), 1u);
  EXPECT_TRUE(is_contained(F1, VPBB0));
  EXPECT_EQ(F2.size(), 1u);
  EXPECT_TRUE(is_contained(F2, VPBB0));
  EXPECT_EQ(F3.size(), 1u);
  EXPECT_TRUE(is_contained(F3, VPBB2));
  EXPECT_EQ(F4.size(), 1u);
  EXPECT_TRUE(is_contained(F4, VPBB1));
  EXPECT_EQ(F5.size(), 2u);
  EXPECT_TRUE(is_contained(F5, VPBB1));
  EXPECT_TRUE(is_contained(F5, VPBB2));
  EXPECT_EQ(F6.size(), 2u);
  EXPECT_TRUE(is_contained(F6, VPBB0));
  EXPECT_TRUE(is_contained(F6, VPBB2));
  EXPECT_EQ(F7.size(), 0u);
}

} // namespace
} // namespace llvm

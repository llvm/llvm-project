//===- llvm/unittests/Transforms/Vectorize/VPlanTest.cpp - VPlan tests ----===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lib/Transforms/Vectorize/VPlan.h"
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

TEST(VPBasicBlockTest, getPlan) {
  {
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

    VPlan Plan;
    Plan.setEntry(VPBB1);

    EXPECT_EQ(&Plan, VPBB1->getPlan());
    EXPECT_EQ(&Plan, VPBB2->getPlan());
    EXPECT_EQ(&Plan, VPBB3->getPlan());
    EXPECT_EQ(&Plan, VPBB4->getPlan());
  }

  {
    // Region block is entry into VPlan.
    VPBasicBlock *R1BB1 = new VPBasicBlock();
    VPBasicBlock *R1BB2 = new VPBasicBlock();
    VPRegionBlock *R1 = new VPRegionBlock(R1BB1, R1BB2, "R1");
    VPBlockUtils::connectBlocks(R1BB1, R1BB2);

    VPlan Plan;
    Plan.setEntry(R1);
    EXPECT_EQ(&Plan, R1->getPlan());
    EXPECT_EQ(&Plan, R1BB1->getPlan());
    EXPECT_EQ(&Plan, R1BB2->getPlan());
  }

  {
    // VPBasicBlock is the entry into the VPlan, followed by a region.
    VPBasicBlock *R1BB1 = new VPBasicBlock();
    VPBasicBlock *R1BB2 = new VPBasicBlock();
    VPRegionBlock *R1 = new VPRegionBlock(R1BB1, R1BB2, "R1");
    VPBlockUtils::connectBlocks(R1BB1, R1BB2);

    VPBasicBlock *VPBB1 = new VPBasicBlock();
    VPBlockUtils::connectBlocks(VPBB1, R1);

    VPlan Plan;
    Plan.setEntry(VPBB1);
    EXPECT_EQ(&Plan, VPBB1->getPlan());
    EXPECT_EQ(&Plan, R1->getPlan());
    EXPECT_EQ(&Plan, R1BB1->getPlan());
    EXPECT_EQ(&Plan, R1BB2->getPlan());
  }

  {
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

    VPlan Plan;
    Plan.setEntry(VPBB1);
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

TEST(VPBasicBlockTest, print) {
  VPInstruction *I1 = new VPInstruction(10, {});
  VPInstruction *I2 = new VPInstruction(1, {I1});
  VPInstruction *I3 = new VPInstruction(2, {I1, I2});

  VPBasicBlock *VPBB1 = new VPBasicBlock();
  VPBB1->appendRecipe(I1);
  VPBB1->appendRecipe(I2);
  VPBB1->appendRecipe(I3);

  VPInstruction *I4 = new VPInstruction(4, {I3, I2});
  VPInstruction *I5 = new VPInstruction(5, {I1});
  VPBasicBlock *VPBB2 = new VPBasicBlock();
  VPBB2->appendRecipe(I4);
  VPBB2->appendRecipe(I5);

  VPBlockUtils::connectBlocks(VPBB1, VPBB2);

  // Check printing an instruction without associated VPlan.
  {
    std::string I3Dump;
    raw_string_ostream OS(I3Dump);
    I3->print(OS);
    OS.flush();
    EXPECT_EQ("<badref> = br <badref> <badref>", I3Dump);
  }

  VPlan Plan;
  Plan.setEntry(VPBB1);
  std::string FullDump;
  raw_string_ostream(FullDump) << Plan;

  EXPECT_EQ(R"(digraph VPlan {
graph [labelloc=t, fontsize=30; label="Vectorization Plan"]
node [shape=rect, fontname=Courier, fontsize=30]
edge [fontname=Courier, fontsize=30]
compound=true
  N0 [label =
    ":\n" +
      "EMIT %vp0 = catchswitch\l" +
      "EMIT %vp1 = ret %vp0\l" +
      "EMIT %vp2 = br %vp0 %vp1\l"
  ]
  N0 -> N1 [ label=""]
  N1 [label =
    ":\n" +
      "EMIT %vp3 = indirectbr %vp2 %vp1\l" +
      "EMIT %vp4 = invoke %vp0\l"
  ]
}
)",
            FullDump);

  {
    std::string I3Dump;
    raw_string_ostream OS(I3Dump);
    I3->print(OS);
    OS.flush();
    EXPECT_EQ("%vp2 = br %vp0 %vp1", I3Dump);
  }

  {
    std::string I2Dump;
    raw_string_ostream OS(I2Dump);
    OS << *I2;
    OS.flush();
    EXPECT_EQ("%vp1 = ret %vp0", I2Dump);
  }
}

} // namespace
} // namespace llvm

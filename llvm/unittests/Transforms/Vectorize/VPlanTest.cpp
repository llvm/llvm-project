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

} // namespace
} // namespace llvm

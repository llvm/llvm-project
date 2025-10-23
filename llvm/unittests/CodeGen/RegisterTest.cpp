//===- RegisterTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Register.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
TEST(RegisterTest, Idx2StackSlot) {
  EXPECT_EQ(Register::index2StackSlot(0), Register::StackSlotZero);
  EXPECT_EQ(Register::index2StackSlot(1), Register::StackSlotZero | 1);
  EXPECT_EQ(Register::index2StackSlot(-1),
            Register::StackSlotZero | Register::StackSlotMask);
  // check that we do not crash on the highest possible value of frame index.
  EXPECT_NO_FATAL_FAILURE(Register::index2StackSlot((1 << 29) - 1));
  // check that we do not crash on the lowest possible value of frame index.
  EXPECT_NO_FATAL_FAILURE(Register::index2StackSlot(-(1 << 29)));
}

TEST(RegisterTest, StackSlotIndex) {
  int MaxPowOf2 = 1 << (Register::MaxFrameIndexBitwidth - 1);
  std::vector<int> FIs = {0, 1 - 1, MaxPowOf2 - 1, -MaxPowOf2};

  for (int FI : FIs) {
    Register Reg = Register::index2StackSlot(FI);
    EXPECT_EQ(Reg.stackSlotIndex(), FI);
  }
}
} // end namespace

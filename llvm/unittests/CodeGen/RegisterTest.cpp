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
  ASSERT_EQ(Register::index2StackSlot(0), Register::StackSlotZero);
  ASSERT_EQ(Register::index2StackSlot(-1),
            Register::StackSlotZero | Register::StackSlotMask);
  ASSERT_EQ(Register::index2StackSlot(Register::StackSlotMask),
            Register::StackSlotZero | Register::StackSlotMask);
  ASSERT_EQ(Register::index2StackSlot(1), Register::StackSlotZero | 1);
}

TEST(RegisterTest, StackSlotIndex) {
  Register Reg;
  std::vector<int64_t> FIs = {0, 1 - 1, (1 << 29) - 1, -(1 << 29)};

  for (int64_t FI : FIs) {
    Reg = Register::index2StackSlot(FI);
    ASSERT_EQ(Reg.stackSlotIndex(), FI);
  }
}
} // end namespace

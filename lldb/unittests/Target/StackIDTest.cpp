//===-- StackIDTest.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/StackID.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(StackIDTest, TailCallDepthOrdersFramesSharingACFA) {
  StackID unwound(0x1000, 0x7f00, nullptr, nullptr);
  StackID tail_caller = unwound;
  tail_caller.SetTailCallDepth(1);
  StackID outer_tail_caller = unwound;
  outer_tail_caller.SetTailCallDepth(2);
  StackID caller(0x1000, 0x7f10, nullptr, nullptr);

  EXPECT_NE(unwound, tail_caller);
  EXPECT_NE(tail_caller, outer_tail_caller);
  EXPECT_TRUE(unwound.IsYoungerThan(tail_caller));
  EXPECT_FALSE(tail_caller.IsYoungerThan(unwound));
  EXPECT_TRUE(tail_caller.IsYoungerThan(outer_tail_caller));
  EXPECT_FALSE(outer_tail_caller.IsYoungerThan(tail_caller));
  EXPECT_TRUE(outer_tail_caller.IsYoungerThan(caller));
  EXPECT_FALSE(caller.IsYoungerThan(outer_tail_caller));
}

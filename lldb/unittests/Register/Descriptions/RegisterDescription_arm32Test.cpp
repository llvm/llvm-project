//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Descriptions/RegisterDescription_arm32.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(RegisterDescription_arm32Test, OptionalTLSPreservesRegisterIdentity) {
  auto base = RegisterDescription_arm32::GetShared();
  EXPECT_EQ(base->Find("tpidruro"), nullptr);
  EXPECT_EQ(base->Find("exception"), nullptr);
  EXPECT_EQ(base->Find("bvr0"), nullptr);
  auto extended = RegisterDescription_arm32::GetShared(true);
  ASSERT_NE(extended->Find("tpidruro"), nullptr);
  EXPECT_EQ(base->Find("pc")->id, extended->Find("pc")->id);
}

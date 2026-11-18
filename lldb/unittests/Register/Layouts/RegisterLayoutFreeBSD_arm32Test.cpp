//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Layouts/FreeBSD/RegisterLayoutFreeBSD_arm32.h"
#include "lldb/Utility/ArchSpec.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(RegisterLayoutFreeBSD_arm32Test, UsesFreeBSDSaveBankOffsets) {
  ArchSpec arch("armv7-unknown-freebsd");
  auto saved = RegisterLayoutFreeBSD_arm32(arch);
  EXPECT_EQ(saved.GetBankSize(0), 68u);
  auto *fpscr = saved.GetDescription()->Find("fpscr");
  ASSERT_NE(fpscr, nullptr);
  EXPECT_EQ(saved.Find(fpscr->id)->pieces[0].byte_offset, 256u);
}

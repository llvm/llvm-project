//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Layouts/FreeBSD/RegisterLayoutFreeBSD_x86_64.h"
#include "lldb/Register/Arch/lldb-x86-register-enums.h"
#include "lldb/Register/RegisterLayout.h"
#include "gtest/gtest.h"

namespace reg_ids = lldb_private::registers::x86::id;
using namespace lldb_private;

TEST(RegisterInfoX86LayoutTest, Compat32LayoutOwnsNamesFromTemporaryProvider) {
  auto provider =
      CreateFreeBSDX86_64RegisterLayout(ArchSpec("i386-unknown-freebsd"));
  EXPECT_EQ(provider.GetBankSize(0), 176u);
  EXPECT_EQ(provider.GetDescription()->GetRegisters()[reg_ids::eax].name,
            "eax");
  EXPECT_EQ(provider.GetDescription()->GetRegisters()[reg_ids::eax].byte_size,
            4u);
  EXPECT_EQ(provider.Find(provider.GetDescription()->Find("eax")->id)
                ->pieces[0]
                .byte_offset,
            14 * 8u);
  EXPECT_EQ(provider.GetDescription()->Find("eax")->byte_size, 4u);
}

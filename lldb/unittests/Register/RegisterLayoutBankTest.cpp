//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/RegisterLayout.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace llvm;

TEST(RegisterLayoutBankTest, SelectsBanksForLogicalValuesAndWiderWrites) {
  RegisterDescriptor parent;
  parent.id = 90;
  parent.name = "parent";
  parent.byte_size = 8;
  auto alias = parent;
  alias.id = 7;
  alias.name = "alias";
  alias.byte_size = 4;
  alias.value_registers = {90};
  alias.zero_extend_parent = 90;
  auto constant = parent;
  constant.id = 200;
  constant.name = "zero";
  constant.constant_value.resize(8, 0);
  auto description = RegisterDescription::Create({alias, constant, parent});
  ASSERT_THAT_EXPECTED(description, Succeeded());
  RegisterLayout layout(lldb::eByteOrderLittle);
  ASSERT_THAT_ERROR(layout.AddBank(3, 8), Succeeded());
  ASSERT_THAT_ERROR(layout.AddBank(5, 8), Succeeded());
  RegisterLocation whole;
  whole.kind = RegisterLocation::Stored;
  whole.pieces = {{3, 0, 4, 0}, {5, 0, 4, 4}};
  ASSERT_THAT_ERROR(layout.Add(90, whole), Succeeded());
  RegisterLocation part;
  part.kind = RegisterLocation::Stored;
  part.pieces = {{3, 0, 4, 0}};
  ASSERT_THAT_ERROR(layout.Add(7, part), Succeeded());
  layout = llvm::cantFail(RegisterLayout::Create(
      std::make_shared<RegisterDescription>(*description), std::move(layout)));

  auto read = layout.GetRequiredBanks(7);
  auto write = layout.GetRequiredBanks(7, true);
  auto zero = layout.GetRequiredBanks(200);
  ASSERT_THAT_EXPECTED(read, Succeeded());
  ASSERT_THAT_EXPECTED(write, Succeeded());
  ASSERT_THAT_EXPECTED(zero, Succeeded());
  EXPECT_EQ(*read, (std::vector<RegisterBankID>{3}));
  EXPECT_EQ(*write, (std::vector<RegisterBankID>{3, 5}));
  EXPECT_TRUE(zero->empty());
  EXPECT_THAT_EXPECTED(layout.GetRequiredBanks(200, true), Failed());
  EXPECT_THAT_EXPECTED(layout.GetRequiredBanks(99), Failed());
}

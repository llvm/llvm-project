//===-- ABITest.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ABI.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(MCBasedABI, MapRegisterName) {
  auto map = [](std::string name) {
    MCBasedABI::MapRegisterName(name, "foo", "bar");
    return name;
  };
  EXPECT_EQ("bar", map("foo"));
  EXPECT_EQ("bar0", map("foo0"));
  EXPECT_EQ("bar47", map("foo47"));
  EXPECT_EQ("foo47x", map("foo47x"));
  EXPECT_EQ("fooo47", map("fooo47"));
  EXPECT_EQ("bar47", map("bar47"));
}


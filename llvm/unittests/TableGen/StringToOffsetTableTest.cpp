//===- StringToOffsetTableTest.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/TableGen/StringToOffsetTable.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Main.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(StringToOffsetTableTest, EscapesCharactersInLongTables) {
  StringToOffsetTable Table;
  Table.GetOrAddStringOffset(std::string(64 * 1024, '\''));

  bool Previous = EmitLongStrLiterals;
  EmitLongStrLiterals = false;
  std::string Output;
  raw_string_ostream OS(Output);
  Table.EmitStringTableStorageDef(OS, "Strings");
  EmitLongStrLiterals = Previous;

  EXPECT_NE(Output.find("'\\x27'"), std::string::npos);
  EXPECT_EQ(Output.find("'''"), std::string::npos);
}

} // namespace

//===-- SBLineEntryTest.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===/

#include "gtest/gtest.h"

#include "lldb/API/LLDB.h"

TEST(SBLineEntryTest, SetLineAndColumn) {
  constexpr uint32_t expected_line_no = 40;
  constexpr uint32_t expected_column_no = 20;

  lldb::SBLineEntry line_entry{};
  line_entry.SetLine(expected_line_no);
  line_entry.SetColumn(expected_column_no);

  const uint32_t line_no = line_entry.GetLine();
  const uint32_t column_no = line_entry.GetColumn();

  EXPECT_EQ(line_no, line_no);
  EXPECT_EQ(column_no, expected_column_no);
}

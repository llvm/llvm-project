//===-- SBLineSpecTest.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===/

#include "gtest/gtest.h"

#include "lldb/API/LLDB.h"
#include "lldb/lldb-defines.h"

TEST(SBLineSpecTest, Constructors) {
  lldb::SBLineSpec spec;
  EXPECT_FALSE(spec.IsValid());
  EXPECT_EQ(spec.GetLine(), LLDB_INVALID_LINE_NUMBER);
  EXPECT_EQ(spec.GetColumn(), LLDB_INVALID_COLUMN_NUMBER);
  EXPECT_TRUE(spec.GetCheckInlines());

  constexpr uint32_t expected_line = 42;
  constexpr uint32_t expected_column = 7;
  lldb::SBFileSpec filespec("/some/random/path", /*resolve=*/false);

  lldb::SBLineSpec all_args(filespec, expected_line, expected_column);
  EXPECT_EQ(all_args.GetFileSpec(), filespec);
  EXPECT_EQ(all_args.GetLine(), expected_line);
  EXPECT_EQ(all_args.GetColumn(), expected_column);
  EXPECT_TRUE(all_args.GetCheckInlines());
  EXPECT_TRUE(all_args.IsValid());

  lldb::SBLineSpec no_column(filespec, expected_line);
  EXPECT_EQ(no_column.GetLine(), expected_line);
  EXPECT_EQ(no_column.GetColumn(), LLDB_INVALID_COLUMN_NUMBER);
  EXPECT_TRUE(no_column.IsValid());

  lldb::SBLineSpec only_file(filespec);
  EXPECT_EQ(only_file.GetLine(), LLDB_INVALID_LINE_NUMBER);
  EXPECT_EQ(only_file.GetColumn(), LLDB_INVALID_COLUMN_NUMBER);
  EXPECT_FALSE(only_file.IsValid());
}

TEST(SBLineSpecTest, Methods) {
  lldb::SBLineSpec spec;
  lldb::SBFileSpec filespec("/tmp/foo.cpp", /*resolve=*/false);
  spec.SetFileSpec(filespec);
  spec.SetLine(10);
  spec.SetColumn(20);
  spec.SetCheckInlines(false);

  EXPECT_EQ(spec.GetFileSpec(), filespec);
  EXPECT_EQ(spec.GetLine(), 10U);
  EXPECT_EQ(spec.GetColumn(), 20U);
  EXPECT_FALSE(spec.GetCheckInlines());
  EXPECT_TRUE(spec.IsValid());
}

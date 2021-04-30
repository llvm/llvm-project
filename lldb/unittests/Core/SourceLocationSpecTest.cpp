//===-- SourceLocationSpecTest.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Core/SourceLocationSpec.h"
#include "lldb/Utility/LLDBAssert.h"

#include "llvm/Testing/Support/Error.h"

using namespace lldb_private;

TEST(SourceLocationSpecTest, OperatorBool) {
  SourceLocationSpec invalid(FileSpec(), 0);
  EXPECT_FALSE(invalid);

  SourceLocationSpec invalid_filespec(FileSpec(), 4);
  EXPECT_FALSE(invalid_filespec);

  SourceLocationSpec invalid_line(FileSpec("/foo/bar"), 0);
  EXPECT_FALSE(invalid_line);

  SourceLocationSpec valid_fs_line_no_column(FileSpec("/foo/bar"), 4);
  EXPECT_TRUE(valid_fs_line_no_column);

  SourceLocationSpec invalid_fs_column(FileSpec(), 4, 0);
  EXPECT_FALSE(invalid_fs_column);

  SourceLocationSpec invalid_line_column(FileSpec("/foo/bar"), 0, 19);
  EXPECT_FALSE(invalid_line_column);

  SourceLocationSpec valid_fs_line_zero_column(FileSpec("/foo/bar"), 4, 0);
  EXPECT_TRUE(valid_fs_line_zero_column);

  SourceLocationSpec valid_fs_line_column(FileSpec("/foo/bar"), 4, 19);
  EXPECT_TRUE(valid_fs_line_column);
}

TEST(SourceLocationSpecTest, FileLineColumnComponents) {
  FileSpec fs("/foo/bar", FileSpec::Style::posix);
  const uint32_t line = 19;
  const uint16_t column = 4;

  SourceLocationSpec without_column(fs, line, LLDB_INVALID_COLUMN_NUMBER, false,
                                    true);
  EXPECT_TRUE(without_column);
  EXPECT_EQ(fs, without_column.GetFileSpec());
  EXPECT_EQ(line, without_column.GetLine().getValueOr(0));
  EXPECT_EQ(llvm::None, without_column.GetColumn());
  EXPECT_FALSE(without_column.GetCheckInlines());
  EXPECT_TRUE(without_column.GetExactMatch());
  EXPECT_STREQ("check inlines = false, exact match = true, decl = /foo/bar:19",
               without_column.GetString().c_str());

  SourceLocationSpec with_column(fs, line, column, true, false);
  EXPECT_TRUE(with_column);
  EXPECT_EQ(column, *with_column.GetColumn());
  EXPECT_TRUE(with_column.GetCheckInlines());
  EXPECT_FALSE(with_column.GetExactMatch());
  EXPECT_STREQ(
      "check inlines = true, exact match = false, decl = /foo/bar:19:4",
      with_column.GetString().c_str());
}

static SourceLocationSpec Create(bool check_inlines, bool exact_match,
                                 FileSpec fs, uint32_t line,
                                 uint16_t column = LLDB_INVALID_COLUMN_NUMBER) {
  return SourceLocationSpec(fs, line, column, check_inlines, exact_match);
}

TEST(SourceLocationSpecTest, Equal) {
  auto Equal = [](SourceLocationSpec lhs, SourceLocationSpec rhs, bool full) {
    return SourceLocationSpec::Equal(lhs, rhs, full);
  };

  const FileSpec fs("/foo/bar", FileSpec::Style::posix);
  const FileSpec other_fs("/foo/baz", FileSpec::Style::posix);

  // mutating FileSpec + const Inlined, ExactMatch, Line
  EXPECT_TRUE(
      Equal(Create(false, false, fs, 4), Create(false, false, fs, 4), true));
  EXPECT_TRUE(
      Equal(Create(true, true, fs, 4), Create(true, true, fs, 4), false));
  EXPECT_FALSE(Equal(Create(false, false, fs, 4),
                     Create(false, false, other_fs, 4), true));
  EXPECT_FALSE(
      Equal(Create(true, true, fs, 4), Create(true, true, other_fs, 4), false));

  // Mutating FileSpec + const Inlined, ExactMatch, Line, Column
  EXPECT_TRUE(Equal(Create(false, false, fs, 4, 19),
                    Create(false, false, fs, 4, 19), true));
  EXPECT_TRUE(Equal(Create(true, true, fs, 4, 19),
                    Create(true, true, fs, 4, 19), false));
  EXPECT_FALSE(Equal(Create(false, false, fs, 4, 19),
                     Create(false, false, other_fs, 4, 19), true));
  EXPECT_FALSE(Equal(Create(true, true, fs, 4, 19),
                     Create(true, true, other_fs, 4, 19), false));

  // Asymetric match
  EXPECT_FALSE(
      Equal(Create(true, true, fs, 4), Create(true, true, fs, 4, 19), true));
  EXPECT_TRUE(Equal(Create(false, false, fs, 4),
                    Create(false, false, fs, 4, 19), false));

  // Mutating Inlined, ExactMatch
  EXPECT_FALSE(
      Equal(Create(true, false, fs, 4), Create(false, true, fs, 4), true));
  EXPECT_TRUE(
      Equal(Create(false, true, fs, 4), Create(true, false, fs, 4), false));

  // Mutating Column
  EXPECT_FALSE(Equal(Create(true, true, fs, 4, 96),
                     Create(true, true, fs, 4, 19), true));
  EXPECT_TRUE(Equal(Create(false, false, fs, 4, 96),
                    Create(false, false, fs, 4, 19), false));
}

TEST(SourceLocationSpecTest, Compare) {
  auto Cmp = [](SourceLocationSpec a, SourceLocationSpec b) {
    return SourceLocationSpec::Compare(a, b);
  };

  FileSpec fs("/foo/bar", FileSpec::Style::posix);
  FileSpec other_fs("/foo/baz", FileSpec::Style::posix);

  // Asymetric comparaison
  EXPECT_EQ(-1, Cmp(Create(true, true, fs, 4), Create(true, true, fs, 4, 19)));
  EXPECT_EQ(-1,
            Cmp(Create(false, false, fs, 4), Create(false, false, fs, 4, 19)));
  EXPECT_EQ(1, Cmp(Create(true, true, fs, 4, 19), Create(true, true, fs, 4)));

  // Mutating FS, const Line
  EXPECT_EQ(
      -1, Cmp(Create(false, false, fs, 4), Create(false, false, other_fs, 4)));
  EXPECT_EQ(-1,
            Cmp(Create(true, true, fs, 4), Create(true, true, other_fs, 4)));
  EXPECT_EQ(1,
            Cmp(Create(false, true, other_fs, 4), Create(false, true, fs, 4)));
  EXPECT_EQ(1,
            Cmp(Create(true, false, other_fs, 4), Create(true, false, fs, 4)));

  // Const FS, mutating Line
  EXPECT_EQ(-1, Cmp(Create(false, false, fs, 1), Create(false, false, fs, 4)));
  EXPECT_EQ(-1, Cmp(Create(true, true, fs, 1), Create(true, true, fs, 4)));
  EXPECT_EQ(0, Cmp(Create(false, true, fs, 4), Create(false, true, fs, 4)));
  EXPECT_EQ(0, Cmp(Create(true, false, fs, 4), Create(true, false, fs, 4)));
  EXPECT_EQ(1, Cmp(Create(false, false, fs, 4), Create(false, false, fs, 1)));
  EXPECT_EQ(1, Cmp(Create(true, true, fs, 4), Create(true, true, fs, 1)));

  // Const FS, mutating Line, const Column
  EXPECT_EQ(-1,
            Cmp(Create(false, true, fs, 1), Create(false, true, fs, 4, 19)));
  EXPECT_EQ(-1, Cmp(Create(true, true, fs, 1), Create(true, true, fs, 4, 19)));
  EXPECT_EQ(1, Cmp(Create(true, false, fs, 4, 19), Create(true, false, fs, 1)));
  EXPECT_EQ(1, Cmp(Create(true, false, fs, 4, 19), Create(true, false, fs, 1)));

  // Mutating FS, const Line, const Column
  EXPECT_EQ(-1, Cmp(Create(false, false, fs, 4, 19),
                    Create(false, false, other_fs, 4, 19)));
  EXPECT_EQ(-1, Cmp(Create(true, true, fs, 4, 19),
                    Create(true, true, other_fs, 4, 19)));
  EXPECT_EQ(
      0, Cmp(Create(false, false, fs, 4, 19), Create(false, false, fs, 4, 19)));
  EXPECT_EQ(0,
            Cmp(Create(true, true, fs, 4, 19), Create(true, true, fs, 4, 19)));
  EXPECT_EQ(1, Cmp(Create(false, true, other_fs, 4, 19),
                   Create(false, true, fs, 4, 19)));
  EXPECT_EQ(1, Cmp(Create(true, false, other_fs, 4, 19),
                   Create(true, false, fs, 4, 19)));

  // Const FS, const Line, mutating Column
  EXPECT_EQ(-1, Cmp(Create(false, false, fs, 4, 19),
                    Create(false, false, fs, 4, 96)));
  EXPECT_EQ(1,
            Cmp(Create(true, true, fs, 4, 96), Create(true, true, fs, 4, 19)));
  EXPECT_EQ(
      1, Cmp(Create(false, true, fs, 4, 96), Create(false, true, fs, 4, 19)));
}

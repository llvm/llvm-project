//===-- SupportFileTest.cpp
//--------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Utility/Checksum.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/SupportFile.h"

using namespace lldb_private;

static llvm::MD5::MD5Result hash1 = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};

static llvm::MD5::MD5Result hash2 = {
    {8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7}};

TEST(SupportFileTest, TestDefaultConstructor) {
  SupportFile support_file;

  EXPECT_EQ(support_file.GetSpecOnly(), FileSpec());
  EXPECT_EQ(support_file.GetChecksum(), Checksum());
}

TEST(SupportFileTest, TestConstructor) {
  FileSpec file_spec("/foo/bar");
  Checksum checksum(hash1);
  SupportFile support_file(file_spec, checksum);

  EXPECT_EQ(support_file.GetSpecOnly(), file_spec);
  EXPECT_EQ(support_file.GetChecksum(), checksum);
}

TEST(SupportFileTest, TestEqual) {
  auto EQ = [&](const SupportFile &LHS, const SupportFile &RHS,
                SupportFile::SupportFileEquality equality) -> bool {
    EXPECT_EQ(LHS.Equal(RHS, equality), RHS.Equal(LHS, equality));
    return LHS.Equal(RHS, equality);
  };

  FileSpec foo_bar("/foo/bar");
  Checksum checksum_foo_bar(hash1);

  FileSpec bar_baz("/bar/baz");
  Checksum checksum_bar_baz(hash2);

  // The canonical support file we're comparing against.
  SupportFile support_file(foo_bar, checksum_foo_bar);

  // Support file A is identical.
  SupportFile support_file_a(foo_bar, checksum_foo_bar);
  EXPECT_TRUE(EQ(support_file, support_file_a, SupportFile::eEqualFileSpec));
  EXPECT_TRUE(EQ(support_file, support_file_a, SupportFile::eEqualChecksum));
  EXPECT_TRUE(
      EQ(support_file, support_file_a, SupportFile::eEqualFileSpecAndChecksum));
  EXPECT_TRUE(EQ(support_file, support_file_a,
                 SupportFile::eEqualFileSpecAndChecksumIfSet));

  // Support file C is has the same path but no checksum.
  SupportFile support_file_b(foo_bar);
  EXPECT_TRUE(EQ(support_file, support_file_b, SupportFile::eEqualFileSpec));
  EXPECT_FALSE(EQ(support_file, support_file_b, SupportFile::eEqualChecksum));
  EXPECT_FALSE(
      EQ(support_file, support_file_b, SupportFile::eEqualFileSpecAndChecksum));
  EXPECT_TRUE(EQ(support_file, support_file_b,
                 SupportFile::eEqualFileSpecAndChecksumIfSet));

  // Support file D has a different path and checksum.
  SupportFile support_file_c(bar_baz, checksum_bar_baz);
  EXPECT_FALSE(EQ(support_file, support_file_c, SupportFile::eEqualFileSpec));
  EXPECT_FALSE(EQ(support_file, support_file_c,
                  SupportFile::eEqualFileSpecAndChecksumIfSet));
  EXPECT_FALSE(EQ(support_file, support_file_c, SupportFile::eEqualChecksum));
  EXPECT_FALSE(
      EQ(support_file, support_file_c, SupportFile::eEqualFileSpecAndChecksum));

  // Support file E has a different path but the same checksum.
  SupportFile support_file_d(bar_baz, checksum_foo_bar);
  EXPECT_FALSE(EQ(support_file, support_file_d, SupportFile::eEqualFileSpec));
  EXPECT_FALSE(EQ(support_file, support_file_d,
                  SupportFile::eEqualFileSpecAndChecksumIfSet));
  EXPECT_TRUE(EQ(support_file, support_file_d, SupportFile::eEqualChecksum));
  EXPECT_FALSE(
      EQ(support_file, support_file_d, SupportFile::eEqualFileSpecAndChecksum));
}

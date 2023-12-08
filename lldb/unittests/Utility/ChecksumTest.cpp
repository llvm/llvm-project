//===-- ChecksumTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Utility/Checksum.h"

using namespace lldb_private;

static llvm::MD5::MD5Result hash1 = {0, 1, 2,  3,  4,  5,  6,  7,
                                     8, 9, 10, 11, 12, 13, 14, 15};

static llvm::MD5::MD5Result hash2 = {0, 1, 2,  3,  4,  5,  6,  7,
                                     8, 9, 10, 11, 12, 13, 14, 15};

static llvm::MD5::MD5Result hash3 = {8, 9, 10, 11, 12, 13, 14, 15,
                                     0, 1, 2,  3,  4,  5,  6,  7};

TEST(ChecksumTest, TestConstructor) {
  Checksum checksum1;
  EXPECT_FALSE(static_cast<bool>(checksum1));
  EXPECT_EQ(checksum1, Checksum());

  Checksum checksum2 = Checksum(hash1);
  EXPECT_EQ(checksum2, Checksum(hash1));

  Checksum checksum3(checksum2);
  EXPECT_EQ(checksum3, Checksum(hash1));
}

TEST(ChecksumTest, TestCopyConstructor) {
  Checksum checksum1;
  EXPECT_FALSE(static_cast<bool>(checksum1));
  EXPECT_EQ(checksum1, Checksum());

  Checksum checksum2 = checksum1;
  EXPECT_EQ(checksum2, checksum1);

  Checksum checksum3(checksum1);
  EXPECT_EQ(checksum3, checksum1);
}

TEST(ChecksumTest, TestMD5) {
  Checksum checksum1(hash1);
  EXPECT_TRUE(static_cast<bool>(checksum1));

  // Make sure two checksums with the same underlying hashes are the same.
  EXPECT_EQ(Checksum(hash1), Checksum(hash2));

  // Make sure two checksums with different underlying hashes are different.
  EXPECT_NE(Checksum(hash1), Checksum(hash3));
}

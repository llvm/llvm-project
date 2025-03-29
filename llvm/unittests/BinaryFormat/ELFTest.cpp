//===- ELFTest.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/ELF.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::ELF;

namespace {
TEST(ELFTest, OSABI) {
  EXPECT_EQ(ELFOSABI_GNU, convertNameToOSABI("gnu"));
  EXPECT_EQ(ELFOSABI_FREEBSD, convertNameToOSABI("freebsd"));
  EXPECT_EQ(ELFOSABI_STANDALONE, convertNameToOSABI("standalone"));
  EXPECT_EQ(ELFOSABI_NONE, convertNameToOSABI("none"));
  // Test unrecognized strings.
  EXPECT_EQ(ELFOSABI_NONE, convertNameToOSABI(""));
  EXPECT_EQ(ELFOSABI_NONE, convertNameToOSABI("linux"));

  EXPECT_EQ("gnu", convertOSABIToName(ELFOSABI_GNU));
  EXPECT_EQ("freebsd", convertOSABIToName(ELFOSABI_FREEBSD));
  EXPECT_EQ("standalone", convertOSABIToName(ELFOSABI_STANDALONE));
  EXPECT_EQ("none", convertOSABIToName(ELFOSABI_NONE));
  // Test unrecognized values.
  EXPECT_EQ("none", convertOSABIToName(0xfe));
}
} // namespace

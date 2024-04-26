//===- ELFTest.cpp ----------------------------------------------*- C++ -*-===//
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
TEST(ELFTest, OSAbi) {
  EXPECT_EQ(ELFOSABI_GNU, convertOSToOSAbi("gnu"));
  EXPECT_EQ(ELFOSABI_FREEBSD, convertOSToOSAbi("freebsd"));
  EXPECT_EQ(ELFOSABI_STANDALONE, convertOSToOSAbi("standalone"));
  EXPECT_EQ(ELFOSABI_NONE, convertOSToOSAbi("none"));
  // Test unrecognized strings.
  EXPECT_EQ(ELFOSABI_NONE, convertOSToOSAbi(""));
  EXPECT_EQ(ELFOSABI_NONE, convertOSToOSAbi("linux"));

  EXPECT_EQ("gnu", convertOSAbiToOS(ELFOSABI_GNU));
  EXPECT_EQ("freebsd", convertOSAbiToOS(ELFOSABI_FREEBSD));
  EXPECT_EQ("standalone", convertOSAbiToOS(ELFOSABI_STANDALONE));
  EXPECT_EQ("none", convertOSAbiToOS(ELFOSABI_NONE));
  // Test unrecognized values.
  EXPECT_EQ("none", convertOSAbiToOS(0xfe));
}
} // namespace

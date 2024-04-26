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
  EXPECT_EQ(ELFOSABI_GNU, convertNameToOSAbi("gnu"));
  EXPECT_EQ(ELFOSABI_FREEBSD, convertNameToOSAbi("freebsd"));
  EXPECT_EQ(ELFOSABI_STANDALONE, convertNameToOSAbi("standalone"));
  EXPECT_EQ(ELFOSABI_NONE, convertNameToOSAbi("none"));
  // Test unrecognized strings.
  EXPECT_EQ(ELFOSABI_NONE, convertNameToOSAbi(""));
  EXPECT_EQ(ELFOSABI_NONE, convertNameToOSAbi("linux"));

  EXPECT_EQ("gnu", convertOSAbiToName(ELFOSABI_GNU));
  EXPECT_EQ("freebsd", convertOSAbiToName(ELFOSABI_FREEBSD));
  EXPECT_EQ("standalone", convertOSAbiToName(ELFOSABI_STANDALONE));
  EXPECT_EQ("none", convertOSAbiToName(ELFOSABI_NONE));
  // Test unrecognized values.
  EXPECT_EQ("none", convertOSAbiToName(0xfe));
}
} // namespace

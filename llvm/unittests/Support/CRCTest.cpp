//===- llvm/unittest/Support/CRCTest.cpp - CRC tests ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements unit tests for CRC calculation functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CRC.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(CRCTest, CRC32) {
  EXPECT_EQ(0x414FA339U,
            llvm::crc32(
                0, StringRef("The quick brown fox jumps over the lazy dog")));
  // CRC-32/ISO-HDLC test vector
  // http://reveng.sourceforge.net/crc-catalogue/17plus.htm#crc.cat.crc-32c
  EXPECT_EQ(0xCBF43926U, llvm::crc32(0, StringRef("123456789")));
}

} // end anonymous namespace

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
#include "llvm/ADT/StringExtras.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(CRCTest, CRC32) {
  EXPECT_EQ(0x414FA339U, llvm::crc32(arrayRefFromStringRef(
                             "The quick brown fox jumps over the lazy dog")));

  // CRC-32/ISO-HDLC test vector
  // http://reveng.sourceforge.net/crc-catalogue/17plus.htm#crc.cat.crc-32c
  EXPECT_EQ(0xCBF43926U, llvm::crc32(arrayRefFromStringRef("123456789")));

  // Check the CRC-32 of each byte value, exercising all of CRCTable.
  for (int i = 0; i < 256; i++) {
    // Compute CRCTable[i] using Hacker's Delight (2nd ed.) Figure 14-7.
    uint32_t crc = i;
    for (int j = 7; j >= 0; j--) {
      uint32_t mask = -(crc & 1);
      crc = (crc >> 1) ^ (0xEDB88320 & mask);
    }

    // CRCTable[i] is the CRC-32 of i without the initial and final bit flips.
    uint8_t byte = i;
    EXPECT_EQ(crc, ~llvm::crc32(0xFFFFFFFFU, byte));
  }
}

} // end anonymous namespace

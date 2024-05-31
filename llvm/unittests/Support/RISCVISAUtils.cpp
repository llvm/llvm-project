//===- LLvm/unittest/Support/RISCVISAUtils.cpp - RISCVISAUtils tests ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/RISCVISAUtils.h"
#include "gtest/gtest.h"

using namespace llvm::RISCVISAUtils;

TEST(ZicfilpFuncSigHash, testCommonCase) {
  // A common representitive case is the signature of the main function
  EXPECT_EQ(zicfilpFuncSigHash("FiiPPcE"), 853561U);
}

// The lowest 20 bits of a MD5 hash should be discarded if they're all zeros
TEST(ZicfilpFuncSigHash, testDiscardAllZeroLabels) {
  // as_number(md5('20412333')) = 0x7a13472ff22eb53e31f6a76027000000
  EXPECT_EQ(zicfilpFuncSigHash("20412333"), 0x60270U);
}

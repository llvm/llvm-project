//===---- RISCVTargetParserTest.cpp - RISCVTargetParser unit tests --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/TargetParser/RISCVTargetParser.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
TEST(RISCVVType, CheckSameRatioLMUL) {
  // Smaller LMUL.
  EXPECT_EQ(RISCVII::LMUL_1,
            RISCVVType::getSameRatioLMUL(16, RISCVII::LMUL_2, 8));
  EXPECT_EQ(RISCVII::LMUL_F2,
            RISCVVType::getSameRatioLMUL(16, RISCVII::LMUL_1, 8));
  // Smaller fractional LMUL.
  EXPECT_EQ(RISCVII::LMUL_F8,
            RISCVVType::getSameRatioLMUL(16, RISCVII::LMUL_F4, 8));
  // Bigger LMUL.
  EXPECT_EQ(RISCVII::LMUL_2,
            RISCVVType::getSameRatioLMUL(8, RISCVII::LMUL_1, 16));
  EXPECT_EQ(RISCVII::LMUL_1,
            RISCVVType::getSameRatioLMUL(8, RISCVII::LMUL_F2, 16));
  // Bigger fractional LMUL.
  EXPECT_EQ(RISCVII::LMUL_F2,
            RISCVVType::getSameRatioLMUL(8, RISCVII::LMUL_F4, 16));
}
} // namespace

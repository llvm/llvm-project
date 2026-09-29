//===- TestStackOffset.cpp - StackOffset unit tests------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/TypeSize.h"
#include "AArch64InstrInfo.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(StackOffset, decomposeStackOffsetForFrameOffsets) {
  StackOffset A = StackOffset::getFixed(8);
  StackOffset B = StackOffset::getFixed(4);
  StackOffset C = StackOffset::getScalable(16);

  // If all offsets can be materialized with only ADDVL,
  // make sure PLSized is 0.
  int64_t ByteSized, VLSized, PLSized;
  AArch64InstrInfo::decomposeStackOffsetForFrameOffsets(A + B + C, ByteSized, PLSized,
                                            VLSized);
  EXPECT_EQ(12, ByteSized);
  EXPECT_EQ(1, VLSized);
  EXPECT_EQ(0, PLSized);

  // If we need an ADDPL to materialize the offset, and the number of scalable
  // bytes fits the ADDPL immediate, fold the scalable bytes to fit in PLSized.
  StackOffset D = StackOffset::getScalable(2);
  AArch64InstrInfo::decomposeStackOffsetForFrameOffsets(C + D, ByteSized, PLSized, VLSized);
  EXPECT_EQ(0, ByteSized);
  EXPECT_EQ(0, VLSized);
  EXPECT_EQ(9, PLSized);

  StackOffset E = StackOffset::getScalable(64);
  StackOffset F = StackOffset::getScalable(2);
  AArch64InstrInfo::decomposeStackOffsetForFrameOffsets(E + F, ByteSized, PLSized, VLSized);
  EXPECT_EQ(0, ByteSized);
  EXPECT_EQ(0, VLSized);
  EXPECT_EQ(33, PLSized);

  // If the offset requires an ADDPL instruction to materialize, and would
  // require more than two instructions, decompose it into both
  // ADDVL (n x 16 bytes) and ADDPL (n x 2 bytes) instructions.
  StackOffset G = StackOffset::getScalable(128);
  StackOffset H = StackOffset::getScalable(2);
  AArch64InstrInfo::decomposeStackOffsetForFrameOffsets(G + H, ByteSized, PLSized, VLSized);
  EXPECT_EQ(0, ByteSized);
  EXPECT_EQ(8, VLSized);
  EXPECT_EQ(1, PLSized);
}

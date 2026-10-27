//===- KnownFPClassTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GISelMITest.h"
#include "llvm/CodeGen/GlobalISel/GISelValueTracking.h"
#include "gtest/gtest.h"

// This test exercises the signaling-NaN query that the
// `print<gisel-value-tracking-fpclass>` pass cannot represent.

TEST_F(AArch64GISelMITest, TestFPClassFPowPosNeverSNaN) {
  StringRef MIRString = R"(
    %ptr:_(p0) = G_IMPLICIT_DEF
    %val:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %exp:_(s32) = G_LOAD %ptr(p0) :: (load (s32))
    %fabs:_(s32) = G_FABS %val
    %fpow:_(s32) = G_FPOW %fabs, %exp
    %copy_fpow:_(s32) = COPY %fpow
)";

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  Register CopyReg = Copies[Copies.size() - 1];
  MachineInstr *FinalCopy = MRI->getVRegDef(CopyReg);
  Register SrcReg = FinalCopy->getOperand(1).getReg();

  GISelValueTracking Info(*MF);
  EXPECT_TRUE(Info.isKnownNeverNaN(SrcReg, true));
}

//===- MachineRegisterInfoTest.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "GISelMITest.h"

TEST_F(AMDGPUGISelMITest, ConstrainRegAttrsPreservesSpecificLLT) {
  setUp();
  if (!TM)
    GTEST_SKIP();

  const LLT S64 = LLT::scalar(64);
  const LLT I64 = LLT::integer(64);
  Register SpecificReg = MRI->createGenericVirtualRegister(I64);
  Register AnyReg = MRI->createGenericVirtualRegister(S64);

  EXPECT_TRUE(MRI->constrainRegAttrs(SpecificReg, AnyReg));
  EXPECT_TRUE(MRI->getType(SpecificReg).isInteger());

  EXPECT_TRUE(MRI->constrainRegAttrs(AnyReg, SpecificReg));
  EXPECT_TRUE(MRI->getType(AnyReg).isInteger());
}

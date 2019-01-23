//===- MachineIRBuilderTest.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GISelMITest.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"

TEST_F(GISelMITest, TestBuildConstantFConstant) {
  if (!TM)
    return;

  MachineIRBuilder B(*MF);
  B.setInsertPt(*EntryMBB, EntryMBB->begin());

  B.buildConstant(LLT::scalar(32), 42);
  B.buildFConstant(LLT::scalar(32), 1.0);

  B.buildConstant(LLT::vector(2, 32), 99);
  B.buildFConstant(LLT::vector(2, 32), 2.0);

  auto CheckStr = R"(
  CHECK: [[CONST0:%[0-9]+]]:_(s32) = G_CONSTANT i32 42
  CHECK: [[FCONST0:%[0-9]+]]:_(s32) = G_FCONSTANT float 1.000000e+00
  CHECK: [[CONST1:%[0-9]+]]:_(s32) = G_CONSTANT i32 99
  CHECK: [[VEC0:%[0-9]+]]:_(<2 x s32>) = G_BUILD_VECTOR [[CONST1]]:_(s32), [[CONST1]]:_(s32)
  CHECK: [[FCONST1:%[0-9]+]]:_(s32) = G_FCONSTANT double 2.000000e+00
  CHECK: [[VEC1:%[0-9]+]]:_(<2 x s32>) = G_BUILD_VECTOR [[FCONST1]]:_(s32), [[FCONST1]]:_(s32)

  )";

  ASSERT_TRUE(CheckMachineFunction(*MF, CheckStr));
}

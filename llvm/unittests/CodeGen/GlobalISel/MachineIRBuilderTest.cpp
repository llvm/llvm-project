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

  B.buildConstant(LLT::scalar(32), 42);
  B.buildFConstant(LLT::scalar(32), 1.0);

  B.buildConstant(LLT::vector(2, 32), 99);
  B.buildFConstant(LLT::vector(2, 32), 2.0);

  auto CheckStr = R"(
  CHECK: [[CONST0:%[0-9]+]]:_(s32) = G_CONSTANT i32 42
  CHECK: [[FCONST0:%[0-9]+]]:_(s32) = G_FCONSTANT float 1.000000e+00
  CHECK: [[CONST1:%[0-9]+]]:_(s32) = G_CONSTANT i32 99
  CHECK: [[VEC0:%[0-9]+]]:_(<2 x s32>) = G_BUILD_VECTOR [[CONST1]]:_(s32), [[CONST1]]:_(s32)
  CHECK: [[FCONST1:%[0-9]+]]:_(s32) = G_FCONSTANT float 2.000000e+00
  CHECK: [[VEC1:%[0-9]+]]:_(<2 x s32>) = G_BUILD_VECTOR [[FCONST1]]:_(s32), [[FCONST1]]:_(s32)
  )";

  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}


#ifdef GTEST_HAS_DEATH_TEST
#ifndef NDEBUG

TEST_F(GISelMITest, TestBuildConstantFConstantDeath) {
  if (!TM)
    return;

  LLVMContext &Ctx = MF->getFunction().getContext();
  APInt APV32(32, 12345);

  // Test APInt version breaks
  EXPECT_DEATH(B.buildConstant(LLT::scalar(16), APV32),
               "creating constant with the wrong size");
  EXPECT_DEATH(B.buildConstant(LLT::vector(2, 16), APV32),
               "creating constant with the wrong size");

  // Test ConstantInt version breaks
  ConstantInt *CI = ConstantInt::get(Ctx, APV32);
  EXPECT_DEATH(B.buildConstant(LLT::scalar(16), *CI),
               "creating constant with the wrong size");
  EXPECT_DEATH(B.buildConstant(LLT::vector(2, 16), *CI),
               "creating constant with the wrong size");

  APFloat DoubleVal(APFloat::IEEEdouble());
  ConstantFP *CF = ConstantFP::get(Ctx, DoubleVal);
  EXPECT_DEATH(B.buildFConstant(LLT::scalar(16), *CF),
               "creating fconstant with the wrong size");
  EXPECT_DEATH(B.buildFConstant(LLT::vector(2, 16), *CF),
               "creating fconstant with the wrong size");
}

#endif
#endif

TEST_F(GISelMITest, DstOpSrcOp) {
  if (!TM)
    return;

  SmallVector<unsigned, 4> Copies;
  collectCopies(Copies, MF);

  LLT s64 = LLT::scalar(64);
  auto MIBAdd = B.buildAdd(s64, Copies[0], Copies[1]);

  // Test SrcOp and DstOp can be constructed directly from MachineOperand by
  // copying the instruction
  B.buildAdd(MIBAdd->getOperand(0), MIBAdd->getOperand(1), MIBAdd->getOperand(2));


  auto CheckStr = R"(
  ; CHECK: [[COPY0:%[0-9]+]]:_(s64) = COPY $x0
  ; CHECK: [[COPY1:%[0-9]+]]:_(s64) = COPY $x1
  ; CHECK: [[ADD:%[0-9]+]]:_(s64) = G_ADD [[COPY0]]:_, [[COPY1]]:_
  ; CHECK: [[ADD]]:_(s64) = G_ADD [[COPY0]]:_, [[COPY1]]:_
  )";

  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, BuildUnmerge) {
  if (!TM)
    return;

  SmallVector<unsigned, 4> Copies;
  collectCopies(Copies, MF);
  B.buildUnmerge(LLT::scalar(32), Copies[0]);
  B.buildUnmerge(LLT::scalar(16), Copies[1]);

  auto CheckStr = R"(
  ; CHECK: [[COPY0:%[0-9]+]]:_(s64) = COPY $x0
  ; CHECK: [[COPY1:%[0-9]+]]:_(s64) = COPY $x1
  ; CHECK: [[UNMERGE32_0:%[0-9]+]]:_(s32), [[UNMERGE32_1:%[0-9]+]]:_(s32) = G_UNMERGE_VALUES [[COPY0]]
  ; CHECK: [[UNMERGE16_0:%[0-9]+]]:_(s16), [[UNMERGE16_1:%[0-9]+]]:_(s16), [[UNMERGE16_2:%[0-9]+]]:_(s16), [[UNMERGE16_3:%[0-9]+]]:_(s16) = G_UNMERGE_VALUES [[COPY1]]

  )";

  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

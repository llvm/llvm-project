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

  // Test APFloat overload.
  APFloat KVal(APFloat::IEEEdouble(), "4.0");
  B.buildFConstant(LLT::scalar(64), KVal);

  auto CheckStr = R"(
  CHECK: [[CONST0:%[0-9]+]]:_(s32) = G_CONSTANT i32 42
  CHECK: [[FCONST0:%[0-9]+]]:_(s32) = G_FCONSTANT float 1.000000e+00
  CHECK: [[CONST1:%[0-9]+]]:_(s32) = G_CONSTANT i32 99
  CHECK: [[VEC0:%[0-9]+]]:_(<2 x s32>) = G_BUILD_VECTOR [[CONST1]]:_(s32), [[CONST1]]:_(s32)
  CHECK: [[FCONST1:%[0-9]+]]:_(s32) = G_FCONSTANT float 2.000000e+00
  CHECK: [[VEC1:%[0-9]+]]:_(<2 x s32>) = G_BUILD_VECTOR [[FCONST1]]:_(s32), [[FCONST1]]:_(s32)
  CHECK: [[FCONST2:%[0-9]+]]:_(s64) = G_FCONSTANT double 4.000000e+00
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

TEST_F(GISelMITest, TestBuildFPInsts) {
  if (!TM)
    return;

  SmallVector<unsigned, 4> Copies;
  collectCopies(Copies, MF);

  LLT S64 = LLT::scalar(64);

  B.buildFAdd(S64, Copies[0], Copies[1]);
  B.buildFSub(S64, Copies[0], Copies[1]);
  B.buildFNeg(S64, Copies[0]);
  B.buildFAbs(S64, Copies[0]);
  B.buildFCopysign(S64, Copies[0], Copies[1]);

  auto CheckStr = R"(
  ; CHECK: [[COPY0:%[0-9]+]]:_(s64) = COPY $x0
  ; CHECK: [[COPY1:%[0-9]+]]:_(s64) = COPY $x1
  ; CHECK: [[FADD:%[0-9]+]]:_(s64) = G_FADD [[COPY0]]:_, [[COPY1]]:_
  ; CHECK: [[FSUB:%[0-9]+]]:_(s64) = G_FSUB [[COPY0]]:_, [[COPY1]]:_
  ; CHECK: [[FNEG:%[0-9]+]]:_(s64) = G_FNEG [[COPY0]]:_
  ; CHECK: [[FABS:%[0-9]+]]:_(s64) = G_FABS [[COPY0]]:_
  ; CHECK: [[FCOPYSIGN:%[0-9]+]]:_(s64) = G_FCOPYSIGN [[COPY0]]:_, [[COPY1]]:_
  )";

  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, BuildIntrinsic) {
  if (!TM)
    return;

  LLT S64 = LLT::scalar(64);
  SmallVector<unsigned, 4> Copies;
  collectCopies(Copies, MF);

  // Make sure DstOp version works. sqrt is just a placeholder intrinsic.
  B.buildIntrinsic(Intrinsic::sqrt, {S64}, false)
    .addUse(Copies[0]);

  // Make sure register version works
  SmallVector<unsigned, 1> Results;
  Results.push_back(MRI->createGenericVirtualRegister(S64));
  B.buildIntrinsic(Intrinsic::sqrt, Results, false)
    .addUse(Copies[1]);

  auto CheckStr = R"(
  ; CHECK: [[COPY0:%[0-9]+]]:_(s64) = COPY $x0
  ; CHECK: [[COPY1:%[0-9]+]]:_(s64) = COPY $x1
  ; CHECK: [[SQRT0:%[0-9]+]]:_(s64) = G_INTRINSIC intrinsic(@llvm.sqrt), [[COPY0]]:_
  ; CHECK: [[SQRT1:%[0-9]+]]:_(s64) = G_INTRINSIC intrinsic(@llvm.sqrt), [[COPY1]]:_
  )";

  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, BuildXor) {
  if (!TM)
    return;

  LLT S64 = LLT::scalar(64);
  LLT S128 = LLT::scalar(128);
  SmallVector<unsigned, 4> Copies;
  collectCopies(Copies, MF);
  B.buildXor(S64, Copies[0], Copies[1]);
  B.buildNot(S64, Copies[0]);

  // Make sure this works with > 64-bit types
  auto Merge = B.buildMerge(S128, {Copies[0], Copies[1]});
  B.buildNot(S128, Merge);
  auto CheckStr = R"(
  ; CHECK: [[COPY0:%[0-9]+]]:_(s64) = COPY $x0
  ; CHECK: [[COPY1:%[0-9]+]]:_(s64) = COPY $x1
  ; CHECK: [[XOR0:%[0-9]+]]:_(s64) = G_XOR [[COPY0]]:_, [[COPY1]]:_
  ; CHECK: [[NEGONE64:%[0-9]+]]:_(s64) = G_CONSTANT i64 -1
  ; CHECK: [[XOR1:%[0-9]+]]:_(s64) = G_XOR [[COPY0]]:_, [[NEGONE64]]:_
  ; CHECK: [[MERGE:%[0-9]+]]:_(s128) = G_MERGE_VALUES [[COPY0]]:_(s64), [[COPY1]]:_(s64)
  ; CHECK: [[NEGONE128:%[0-9]+]]:_(s128) = G_CONSTANT i128 -1
  ; CHECK: [[XOR2:%[0-9]+]]:_(s128) = G_XOR [[MERGE]]:_, [[NEGONE128]]:_
  )";

  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

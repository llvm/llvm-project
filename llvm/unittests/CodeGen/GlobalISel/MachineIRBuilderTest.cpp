//===- MachineIRBuilderTest.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GISelMITest.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"

TEST_F(AArch64GISelMITest, TestBuildConstantFConstant) {
  setUp();
  if (!TM)
    return;

  B.buildConstant(LLT::scalar(32), 42);
  B.buildFConstant(LLT::scalar(32), 1.0);

  B.buildConstant(LLT::fixed_vector(2, 32), 99);
  B.buildFConstant(LLT::fixed_vector(2, 32), 2.0);

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

TEST_F(AArch64GISelMITest, TestBuildConstantFConstantDeath) {
  setUp();
  if (!TM)
    return;

  LLVMContext &Ctx = MF->getFunction().getContext();
  APInt APV32(32, 12345);

  // Test APInt version breaks
  EXPECT_DEATH(B.buildConstant(LLT::scalar(16), APV32),
               "creating constant with the wrong size");
  EXPECT_DEATH(B.buildConstant(LLT::fixed_vector(2, 16), APV32),
               "creating constant with the wrong size");

  // Test ConstantInt version breaks
  ConstantInt *CI = ConstantInt::get(Ctx, APV32);
  EXPECT_DEATH(B.buildConstant(LLT::scalar(16), *CI),
               "creating constant with the wrong size");
  EXPECT_DEATH(B.buildConstant(LLT::fixed_vector(2, 16), *CI),
               "creating constant with the wrong size");

  APFloat DoubleVal(APFloat::IEEEdouble());
  ConstantFP *CF = ConstantFP::get(Ctx, DoubleVal);
  EXPECT_DEATH(B.buildFConstant(LLT::scalar(16), *CF),
               "creating fconstant with the wrong size");
  EXPECT_DEATH(B.buildFConstant(LLT::fixed_vector(2, 16), *CF),
               "creating fconstant with the wrong size");
}

#endif
#endif

TEST_F(AArch64GISelMITest, DstOpSrcOp) {
  setUp();
  if (!TM)
    return;

  SmallVector<Register, 4> Copies;
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

TEST_F(AArch64GISelMITest, BuildUnmerge) {
  setUp();
  if (!TM)
    return;

  SmallVector<Register, 4> Copies;
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

TEST_F(AArch64GISelMITest, TestBuildFPInsts) {
  setUp();
  if (!TM)
    return;

  SmallVector<Register, 4> Copies;
  collectCopies(Copies, MF);

  LLT S64 = LLT::scalar(64);

  B.buildFAdd(S64, Copies[0], Copies[1]);
  B.buildFSub(S64, Copies[0], Copies[1]);
  B.buildFMA(S64, Copies[0], Copies[1], Copies[2]);
  B.buildFMAD(S64, Copies[0], Copies[1], Copies[2]);
  B.buildFMAD(S64, Copies[0], Copies[1], Copies[2], MachineInstr::FmNoNans);
  B.buildFNeg(S64, Copies[0]);
  B.buildFAbs(S64, Copies[0]);
  B.buildFCopysign(S64, Copies[0], Copies[1]);

  auto CheckStr = R"(
  ; CHECK: [[COPY0:%[0-9]+]]:_(s64) = COPY $x0
  ; CHECK: [[COPY1:%[0-9]+]]:_(s64) = COPY $x1
  ; CHECK: [[COPY2:%[0-9]+]]:_(s64) = COPY $x2
  ; CHECK: [[FADD:%[0-9]+]]:_(s64) = G_FADD [[COPY0]]:_, [[COPY1]]:_
  ; CHECK: [[FSUB:%[0-9]+]]:_(s64) = G_FSUB [[COPY0]]:_, [[COPY1]]:_
  ; CHECK: [[FMA:%[0-9]+]]:_(s64) = G_FMA [[COPY0]]:_, [[COPY1]]:_, [[COPY2]]:_
  ; CHECK: [[FMAD0:%[0-9]+]]:_(s64) = G_FMAD [[COPY0]]:_, [[COPY1]]:_, [[COPY2]]:_
  ; CHECK: [[FMAD1:%[0-9]+]]:_(s64) = nnan G_FMAD [[COPY0]]:_, [[COPY1]]:_, [[COPY2]]:_
  ; CHECK: [[FNEG:%[0-9]+]]:_(s64) = G_FNEG [[COPY0]]:_
  ; CHECK: [[FABS:%[0-9]+]]:_(s64) = G_FABS [[COPY0]]:_
  ; CHECK: [[FCOPYSIGN:%[0-9]+]]:_(s64) = G_FCOPYSIGN [[COPY0]]:_, [[COPY1]]:_
  )";

  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(AArch64GISelMITest, BuildIntrinsic) {
  setUp();
  if (!TM)
    return;

  LLT S64 = LLT::scalar(64);
  SmallVector<Register, 4> Copies;
  collectCopies(Copies, MF);

  // Make sure DstOp version works. sqrt is just a placeholder intrinsic.
  B.buildIntrinsic(Intrinsic::sqrt, {S64}, false)
    .addUse(Copies[0]);

  // Make sure register version works
  SmallVector<Register, 1> Results;
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

TEST_F(AArch64GISelMITest, BuildXor) {
  setUp();
  if (!TM)
    return;

  LLT S64 = LLT::scalar(64);
  LLT S128 = LLT::scalar(128);
  SmallVector<Register, 4> Copies;
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

TEST_F(AArch64GISelMITest, BuildBitCounts) {
  setUp();
  if (!TM)
    return;

  LLT S32 = LLT::scalar(32);
  SmallVector<Register, 4> Copies;
  collectCopies(Copies, MF);

  B.buildCTPOP(S32, Copies[0]);
  B.buildCTLZ(S32, Copies[0]);
  B.buildCTLZ_ZERO_UNDEF(S32, Copies[1]);
  B.buildCTTZ(S32, Copies[0]);
  B.buildCTTZ_ZERO_UNDEF(S32, Copies[1]);

  auto CheckStr = R"(
  ; CHECK: [[COPY0:%[0-9]+]]:_(s64) = COPY $x0
  ; CHECK: [[COPY1:%[0-9]+]]:_(s64) = COPY $x1
  ; CHECK: [[CTPOP:%[0-9]+]]:_(s32) = G_CTPOP [[COPY0]]:_
  ; CHECK: [[CTLZ0:%[0-9]+]]:_(s32) = G_CTLZ [[COPY0]]:_
  ; CHECK: [[CTLZ_UNDEF0:%[0-9]+]]:_(s32) = G_CTLZ_ZERO_UNDEF [[COPY1]]:_
  ; CHECK: [[CTTZ:%[0-9]+]]:_(s32) = G_CTTZ [[COPY0]]:_
  ; CHECK: [[CTTZ_UNDEF0:%[0-9]+]]:_(s32) = G_CTTZ_ZERO_UNDEF [[COPY1]]:_
  )";

  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(AArch64GISelMITest, BuildCasts) {
  setUp();
  if (!TM)
    return;

  LLT S32 = LLT::scalar(32);
  SmallVector<Register, 4> Copies;
  collectCopies(Copies, MF);

  B.buildUITOFP(S32, Copies[0]);
  B.buildSITOFP(S32, Copies[0]);
  B.buildFPTOUI(S32, Copies[0]);
  B.buildFPTOSI(S32, Copies[0]);

  auto CheckStr = R"(
  ; CHECK: [[COPY0:%[0-9]+]]:_(s64) = COPY $x0
  ; CHECK: [[UITOFP:%[0-9]+]]:_(s32) = G_UITOFP [[COPY0]]:_
  ; CHECK: [[SITOFP:%[0-9]+]]:_(s32) = G_SITOFP [[COPY0]]:_
  ; CHECK: [[FPTOUI:%[0-9]+]]:_(s32) = G_FPTOUI [[COPY0]]:_
  ; CHECK: [[FPTOSI:%[0-9]+]]:_(s32) = G_FPTOSI [[COPY0]]:_
  )";

  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(AArch64GISelMITest, BuildMinMaxAbs) {
  setUp();
  if (!TM)
    return;

  LLT S64 = LLT::scalar(64);
  SmallVector<Register, 4> Copies;
  collectCopies(Copies, MF);

  B.buildSMin(S64, Copies[0], Copies[1]);
  B.buildSMax(S64, Copies[0], Copies[1]);
  B.buildUMin(S64, Copies[0], Copies[1]);
  B.buildUMax(S64, Copies[0], Copies[1]);
  B.buildAbs(S64, Copies[0]);

  auto CheckStr = R"(
  ; CHECK: [[COPY0:%[0-9]+]]:_(s64) = COPY $x0
  ; CHECK: [[COPY1:%[0-9]+]]:_(s64) = COPY $x1
  ; CHECK: [[SMIN0:%[0-9]+]]:_(s64) = G_SMIN [[COPY0]]:_, [[COPY1]]:_
  ; CHECK: [[SMAX0:%[0-9]+]]:_(s64) = G_SMAX [[COPY0]]:_, [[COPY1]]:_
  ; CHECK: [[UMIN0:%[0-9]+]]:_(s64) = G_UMIN [[COPY0]]:_, [[COPY1]]:_
  ; CHECK: [[UMAX0:%[0-9]+]]:_(s64) = G_UMAX [[COPY0]]:_, [[COPY1]]:_
  ; CHECK: [[UABS0:%[0-9]+]]:_(s64) = G_ABS [[COPY0]]:_
  )";

  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(AArch64GISelMITest, BuildAtomicRMW) {
  setUp();
  if (!TM)
    return;

  LLT S64 = LLT::scalar(64);
  LLT P0 = LLT::pointer(0, 64);
  SmallVector<Register, 4> Copies;
  collectCopies(Copies, MF);

  MachineMemOperand *MMO = MF->getMachineMemOperand(
      MachinePointerInfo(),
      MachineMemOperand::MOLoad | MachineMemOperand::MOStore, 8, Align(8),
      AAMDNodes(), nullptr, SyncScope::System, AtomicOrdering::Unordered);

  auto Ptr = B.buildUndef(P0);
  B.buildAtomicRMWFAdd(S64, Ptr, Copies[0], *MMO);
  B.buildAtomicRMWFSub(S64, Ptr, Copies[0], *MMO);

  auto CheckStr = R"(
  ; CHECK: [[COPY0:%[0-9]+]]:_(s64) = COPY $x0
  ; CHECK: [[COPY1:%[0-9]+]]:_(s64) = COPY $x1
  ; CHECK: [[PTR:%[0-9]+]]:_(p0) = G_IMPLICIT_DEF
  ; CHECK: [[FADD:%[0-9]+]]:_(s64) = G_ATOMICRMW_FADD [[PTR]]:_(p0), [[COPY0]]:_ :: (load store unordered 8)
  ; CHECK: [[FSUB:%[0-9]+]]:_(s64) = G_ATOMICRMW_FSUB [[PTR]]:_(p0), [[COPY0]]:_ :: (load store unordered 8)
  )";

  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(AArch64GISelMITest, BuildMerge) {
  setUp();
  if (!TM)
    return;

  LLT S32 = LLT::scalar(32);
  Register RegC0 = B.buildConstant(S32, 0).getReg(0);
  Register RegC1 = B.buildConstant(S32, 1).getReg(0);
  Register RegC2 = B.buildConstant(S32, 2).getReg(0);
  Register RegC3 = B.buildConstant(S32, 3).getReg(0);

  // Merging plain constants as one big blob of bit should produce a
  // G_MERGE_VALUES.
  B.buildMerge(LLT::scalar(128), {RegC0, RegC1, RegC2, RegC3});
  // Merging plain constants to a vector should produce a G_BUILD_VECTOR.
  LLT V2x32 = LLT::fixed_vector(2, 32);
  Register RegC0C1 =
      B.buildMerge(V2x32, {RegC0, RegC1}).getReg(0);
  Register RegC2C3 =
      B.buildMerge(V2x32, {RegC2, RegC3}).getReg(0);
  // Merging vector constants to a vector should produce a G_CONCAT_VECTORS.
  B.buildMerge(LLT::fixed_vector(4, 32), {RegC0C1, RegC2C3});
  // Merging vector constants to a plain type is not allowed.
  // Nothing else to test.

  auto CheckStr = R"(
  ; CHECK: [[C0:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
  ; CHECK: [[C1:%[0-9]+]]:_(s32) = G_CONSTANT i32 1
  ; CHECK: [[C2:%[0-9]+]]:_(s32) = G_CONSTANT i32 2
  ; CHECK: [[C3:%[0-9]+]]:_(s32) = G_CONSTANT i32 3
  ; CHECK: {{%[0-9]+}}:_(s128) = G_MERGE_VALUES [[C0]]:_(s32), [[C1]]:_(s32), [[C2]]:_(s32), [[C3]]:_(s32)
  ; CHECK: [[LOW2x32:%[0-9]+]]:_(<2 x s32>) = G_BUILD_VECTOR [[C0]]:_(s32), [[C1]]:_(s32)
  ; CHECK: [[HIGH2x32:%[0-9]+]]:_(<2 x s32>) = G_BUILD_VECTOR [[C2]]:_(s32), [[C3]]:_(s32)
  ; CHECK: {{%[0-9]+}}:_(<4 x s32>) = G_CONCAT_VECTORS [[LOW2x32]]:_(<2 x s32>), [[HIGH2x32]]:_(<2 x s32>)
  )";

  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(AArch64GISelMITest, BuildAddoSubo) {
  setUp();
  if (!TM)
    return;

  LLT S1 = LLT::scalar(1);
  LLT S64 = LLT::scalar(64);
  SmallVector<Register, 4> Copies;
  collectCopies(Copies, MF);

  auto UAddo = B.buildUAddo(S64, S1, Copies[0], Copies[1]);
  auto USubo = B.buildUSubo(S64, S1, Copies[0], Copies[1]);
  auto SAddo = B.buildSAddo(S64, S1, Copies[0], Copies[1]);
  auto SSubo = B.buildSSubo(S64, S1, Copies[0], Copies[1]);

  B.buildUAdde(S64, S1, Copies[0], Copies[1], UAddo.getReg(1));
  B.buildUSube(S64, S1, Copies[0], Copies[1], USubo.getReg(1));
  B.buildSAdde(S64, S1, Copies[0], Copies[1], SAddo.getReg(1));
  B.buildSSube(S64, S1, Copies[0], Copies[1], SSubo.getReg(1));

  auto CheckStr = R"(
  ; CHECK: [[COPY0:%[0-9]+]]:_(s64) = COPY $x0
  ; CHECK: [[COPY1:%[0-9]+]]:_(s64) = COPY $x1
  ; CHECK: [[UADDO:%[0-9]+]]:_(s64), [[UADDO_FLAG:%[0-9]+]]:_(s1) = G_UADDO [[COPY0]]:_, [[COPY1]]:_
  ; CHECK: [[USUBO:%[0-9]+]]:_(s64), [[USUBO_FLAG:%[0-9]+]]:_(s1) = G_USUBO [[COPY0]]:_, [[COPY1]]:_
  ; CHECK: [[SADDO:%[0-9]+]]:_(s64), [[SADDO_FLAG:%[0-9]+]]:_(s1) = G_SADDO [[COPY0]]:_, [[COPY1]]:_
  ; CHECK: [[SSUBO:%[0-9]+]]:_(s64), [[SSUBO_FLAG:%[0-9]+]]:_(s1) = G_SSUBO [[COPY0]]:_, [[COPY1]]:_
  ; CHECK: [[UADDE:%[0-9]+]]:_(s64), [[UADDE_FLAG:%[0-9]+]]:_(s1) = G_UADDE [[COPY0]]:_, [[COPY1]]:_, [[UADDO_FLAG]]
  ; CHECK: [[USUBE:%[0-9]+]]:_(s64), [[USUBE_FLAG:%[0-9]+]]:_(s1) = G_USUBE [[COPY0]]:_, [[COPY1]]:_, [[USUBO_FLAG]]
  ; CHECK: [[SADDE:%[0-9]+]]:_(s64), [[SADDE_FLAG:%[0-9]+]]:_(s1) = G_SADDE [[COPY0]]:_, [[COPY1]]:_, [[SADDO_FLAG]]
  ; CHECK: [[SSUBE:%[0-9]+]]:_(s64), [[SSUBE_FLAG:%[0-9]+]]:_(s1) = G_SSUBE [[COPY0]]:_, [[COPY1]]:_, [[SSUBO_FLAG]]
  )";

  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(AArch64GISelMITest, BuildBitfieldExtract) {
  setUp();
  if (!TM)
    return;
  LLT S64 = LLT::scalar(64);
  SmallVector<Register, 4> Copies;
  collectCopies(Copies, MF);

  auto Ubfx = B.buildUbfx(S64, Copies[0], Copies[1], Copies[2]);
  B.buildSbfx(S64, Ubfx, Copies[0], Copies[2]);

  const auto *CheckStr = R"(
  ; CHECK: [[COPY0:%[0-9]+]]:_(s64) = COPY $x0
  ; CHECK: [[COPY1:%[0-9]+]]:_(s64) = COPY $x1
  ; CHECK: [[COPY2:%[0-9]+]]:_(s64) = COPY $x2
  ; CHECK: [[UBFX:%[0-9]+]]:_(s64) = G_UBFX [[COPY0]]:_, [[COPY1]]:_(s64), [[COPY2]]:_
  ; CHECK: [[SBFX:%[0-9]+]]:_(s64) = G_SBFX [[UBFX]]:_, [[COPY0]]:_(s64), [[COPY2]]:_
  )";

  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

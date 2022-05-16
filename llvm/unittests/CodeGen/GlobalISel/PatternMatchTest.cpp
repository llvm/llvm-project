//===- PatternMatchTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GISelMITest.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace MIPatternMatch;

namespace {

TEST_F(AArch64GISelMITest, MatchIntConstant) {
  setUp();
  if (!TM)
    return;
  auto MIBCst = B.buildConstant(LLT::scalar(64), 42);
  int64_t Cst;
  bool match = mi_match(MIBCst.getReg(0), *MRI, m_ICst(Cst));
  EXPECT_TRUE(match);
  EXPECT_EQ(Cst, 42);
}

TEST_F(AArch64GISelMITest, MatchIntConstantRegister) {
  setUp();
  if (!TM)
    return;
  auto MIBCst = B.buildConstant(LLT::scalar(64), 42);
  Optional<ValueAndVReg> Src0;
  bool match = mi_match(MIBCst.getReg(0), *MRI, m_GCst(Src0));
  EXPECT_TRUE(match);
  EXPECT_EQ(Src0->VReg, MIBCst.getReg(0));
}

TEST_F(AArch64GISelMITest, MatchIntConstantSplat) {
  setUp();
  if (!TM)
    return;

  LLT s64 = LLT::scalar(64);
  LLT v4s64 = LLT::fixed_vector(4, s64);

  MachineInstrBuilder FortyTwoSplat =
      B.buildSplatVector(v4s64, B.buildConstant(s64, 42));
  int64_t Cst;
  EXPECT_TRUE(mi_match(FortyTwoSplat.getReg(0), *MRI, m_ICstOrSplat(Cst)));
  EXPECT_EQ(Cst, 42);

  MachineInstrBuilder NonConstantSplat =
      B.buildBuildVector(v4s64, {Copies[0], Copies[0], Copies[0], Copies[0]});
  EXPECT_FALSE(mi_match(NonConstantSplat.getReg(0), *MRI, m_ICstOrSplat(Cst)));
}

TEST_F(AArch64GISelMITest, MachineInstrPtrBind) {
  setUp();
  if (!TM)
    return;
  auto MIBAdd = B.buildAdd(LLT::scalar(64), Copies[0], Copies[1]);
  // Test 'MachineInstr *' bind.
  // Default mi_match.
  MachineInstr *MIPtr = MIBAdd.getInstr();
  bool match = mi_match(MIPtr, *MRI, m_GAdd(m_Reg(), m_Reg()));
  EXPECT_TRUE(match);
  // Specialized mi_match for MachineInstr &.
  MachineInstr &MI = *MIBAdd.getInstr();
  match = mi_match(MI, *MRI, m_GAdd(m_Reg(), m_Reg()));
  EXPECT_TRUE(match);
  // MachineInstrBuilder has automatic conversion to MachineInstr *.
  match = mi_match(MIBAdd, *MRI, m_GAdd(m_Reg(), m_Reg()));
  EXPECT_TRUE(match);
  // Match instruction without def.
  auto MIBBrcond = B.buildBrCond(Copies[0], B.getMBB());
  MachineInstr *MatchedMI;
  match = mi_match(MIBBrcond, *MRI, m_MInstr(MatchedMI));
  EXPECT_TRUE(match);
  EXPECT_TRUE(MIBBrcond.getInstr() == MatchedMI);
  // Match instruction with two defs.
  auto MIBUAddO =
      B.buildUAddo(LLT::scalar(64), LLT::scalar(1), Copies[0], Copies[1]);
  match = mi_match(MIBUAddO, *MRI, m_MInstr(MatchedMI));
  EXPECT_TRUE(match);
  EXPECT_TRUE(MIBUAddO.getInstr() == MatchedMI);
}

TEST_F(AArch64GISelMITest, MatchBinaryOp) {
  setUp();
  if (!TM)
    return;
  LLT s32 = LLT::scalar(32);
  LLT s64 = LLT::scalar(64);
  LLT p0 = LLT::pointer(0, 64);
  auto MIBAdd = B.buildAdd(s64, Copies[0], Copies[1]);
  // Test case for no bind.
  bool match =
      mi_match(MIBAdd.getReg(0), *MRI, m_GAdd(m_Reg(), m_Reg()));
  EXPECT_TRUE(match);
  Register Src0, Src1, Src2;
  match = mi_match(MIBAdd.getReg(0), *MRI,
                   m_GAdd(m_Reg(Src0), m_Reg(Src1)));
  EXPECT_TRUE(match);
  EXPECT_EQ(Src0, Copies[0]);
  EXPECT_EQ(Src1, Copies[1]);

  // Build MUL(ADD %0, %1), %2
  auto MIBMul = B.buildMul(s64, MIBAdd, Copies[2]);

  // Try to match MUL.
  match = mi_match(MIBMul.getReg(0), *MRI,
                   m_GMul(m_Reg(Src0), m_Reg(Src1)));
  EXPECT_TRUE(match);
  EXPECT_EQ(Src0, MIBAdd.getReg(0));
  EXPECT_EQ(Src1, Copies[2]);

  // Try to match MUL(ADD)
  match = mi_match(MIBMul.getReg(0), *MRI,
                   m_GMul(m_GAdd(m_Reg(Src0), m_Reg(Src1)), m_Reg(Src2)));
  EXPECT_TRUE(match);
  EXPECT_EQ(Src0, Copies[0]);
  EXPECT_EQ(Src1, Copies[1]);
  EXPECT_EQ(Src2, Copies[2]);

  // Test Commutativity.
  auto MIBMul2 = B.buildMul(s64, Copies[0], B.buildConstant(s64, 42));
  // Try to match MUL(Cst, Reg) on src of MUL(Reg, Cst) to validate
  // commutativity.
  int64_t Cst;
  match = mi_match(MIBMul2.getReg(0), *MRI,
                   m_GMul(m_ICst(Cst), m_Reg(Src0)));
  EXPECT_TRUE(match);
  EXPECT_EQ(Cst, 42);
  EXPECT_EQ(Src0, Copies[0]);

  // Make sure commutative doesn't work with something like SUB.
  auto MIBSub = B.buildSub(s64, Copies[0], B.buildConstant(s64, 42));
  match = mi_match(MIBSub.getReg(0), *MRI,
                   m_GSub(m_ICst(Cst), m_Reg(Src0)));
  EXPECT_FALSE(match);

  auto MIBFMul = B.buildInstr(TargetOpcode::G_FMUL, {s64},
                              {Copies[0], B.buildConstant(s64, 42)});
  // Match and test commutativity for FMUL.
  match = mi_match(MIBFMul.getReg(0), *MRI,
                   m_GFMul(m_ICst(Cst), m_Reg(Src0)));
  EXPECT_TRUE(match);
  EXPECT_EQ(Cst, 42);
  EXPECT_EQ(Src0, Copies[0]);

  // FSUB
  auto MIBFSub = B.buildInstr(TargetOpcode::G_FSUB, {s64},
                              {Copies[0], B.buildConstant(s64, 42)});
  match = mi_match(MIBFSub.getReg(0), *MRI,
                   m_GFSub(m_Reg(Src0), m_Reg()));
  EXPECT_TRUE(match);
  EXPECT_EQ(Src0, Copies[0]);

  // Build AND %0, %1
  auto MIBAnd = B.buildAnd(s64, Copies[0], Copies[1]);
  // Try to match AND.
  match = mi_match(MIBAnd.getReg(0), *MRI,
                   m_GAnd(m_Reg(Src0), m_Reg(Src1)));
  EXPECT_TRUE(match);
  EXPECT_EQ(Src0, Copies[0]);
  EXPECT_EQ(Src1, Copies[1]);

  // Build OR %0, %1
  auto MIBOr = B.buildOr(s64, Copies[0], Copies[1]);
  // Try to match OR.
  match = mi_match(MIBOr.getReg(0), *MRI,
                   m_GOr(m_Reg(Src0), m_Reg(Src1)));
  EXPECT_TRUE(match);
  EXPECT_EQ(Src0, Copies[0]);
  EXPECT_EQ(Src1, Copies[1]);

  // Match lshr, and make sure a different shift amount type works.
  auto TruncCopy1 = B.buildTrunc(s32, Copies[1]);
  auto LShr = B.buildLShr(s64, Copies[0], TruncCopy1);
  match = mi_match(LShr.getReg(0), *MRI,
                   m_GLShr(m_Reg(Src0), m_Reg(Src1)));
  EXPECT_TRUE(match);
  EXPECT_EQ(Src0, Copies[0]);
  EXPECT_EQ(Src1, TruncCopy1.getReg(0));

  // Match shl, and make sure a different shift amount type works.
  auto Shl = B.buildShl(s64, Copies[0], TruncCopy1);
  match = mi_match(Shl.getReg(0), *MRI,
                   m_GShl(m_Reg(Src0), m_Reg(Src1)));
  EXPECT_TRUE(match);
  EXPECT_EQ(Src0, Copies[0]);
  EXPECT_EQ(Src1, TruncCopy1.getReg(0));

  // Build a G_PTR_ADD and check that we can match it.
  auto PtrAdd = B.buildPtrAdd(p0, {B.buildUndef(p0)}, Copies[0]);
  match = mi_match(PtrAdd.getReg(0), *MRI, m_GPtrAdd(m_Reg(Src0), m_Reg(Src1)));
  EXPECT_TRUE(match);
  EXPECT_EQ(Src0, PtrAdd->getOperand(1).getReg());
  EXPECT_EQ(Src1, Copies[0]);

  auto MIBCst = B.buildConstant(s64, 42);
  auto MIBAddCst = B.buildAdd(s64, MIBCst, Copies[0]);
  auto MIBUnmerge = B.buildUnmerge({s32, s32}, B.buildConstant(s64, 42));

  // m_BinOp with opcode.
  // Match binary instruction, opcode and its non-commutative operands.
  match = mi_match(MIBAddCst, *MRI,
                   m_BinOp(TargetOpcode::G_ADD, m_ICst(Cst), m_Reg(Src0)));
  EXPECT_TRUE(match);
  EXPECT_EQ(Src0, Copies[0]);
  EXPECT_EQ(Cst, 42);

  // Opcode doesn't match.
  match = mi_match(MIBAddCst, *MRI,
                   m_BinOp(TargetOpcode::G_MUL, m_ICst(Cst), m_Reg(Src0)));
  EXPECT_FALSE(match);

  match = mi_match(MIBAddCst, *MRI,
                   m_BinOp(TargetOpcode::G_ADD, m_Reg(Src0), m_ICst(Cst)));
  EXPECT_FALSE(match);

  // Instruction is not binary.
  match = mi_match(MIBCst, *MRI,
                   m_BinOp(TargetOpcode::G_MUL, m_Reg(Src0), m_Reg(Src1)));
  EXPECT_FALSE(match);
  match = mi_match(MIBUnmerge, *MRI,
                   m_BinOp(TargetOpcode::G_MUL, m_Reg(Src0), m_Reg(Src1)));
  EXPECT_FALSE(match);

  // m_CommutativeBinOp with opcode.
  match = mi_match(
      MIBAddCst, *MRI,
      m_CommutativeBinOp(TargetOpcode::G_ADD, m_ICst(Cst), m_Reg(Src0)));
  EXPECT_TRUE(match);
  EXPECT_EQ(Src0, Copies[0]);
  EXPECT_EQ(Cst, 42);

  match = mi_match(
      MIBAddCst, *MRI,
      m_CommutativeBinOp(TargetOpcode::G_MUL, m_ICst(Cst), m_Reg(Src0)));
  EXPECT_FALSE(match);

  match = mi_match(
      MIBAddCst, *MRI,
      m_CommutativeBinOp(TargetOpcode::G_ADD, m_Reg(Src0), m_ICst(Cst)));
  EXPECT_TRUE(match);
  EXPECT_EQ(Src0, Copies[0]);
  EXPECT_EQ(Cst, 42);

  match = mi_match(
      MIBCst, *MRI,
      m_CommutativeBinOp(TargetOpcode::G_MUL, m_Reg(Src0), m_Reg(Src1)));
  EXPECT_FALSE(match);
  match = mi_match(
      MIBUnmerge, *MRI,
      m_CommutativeBinOp(TargetOpcode::G_MUL, m_Reg(Src0), m_Reg(Src1)));
  EXPECT_FALSE(match);
}

TEST_F(AArch64GISelMITest, MatchICmp) {
  setUp();
  if (!TM)
    return;

  const LLT s1 = LLT::scalar(1);
  auto CmpEq = B.buildICmp(CmpInst::ICMP_EQ, s1, Copies[0], Copies[1]);

  // Check match any predicate.
  bool match =
      mi_match(CmpEq.getReg(0), *MRI, m_GICmp(m_Pred(), m_Reg(), m_Reg()));
  EXPECT_TRUE(match);

  // Check we get the predicate and registers.
  CmpInst::Predicate Pred;
  Register Reg0;
  Register Reg1;
  match = mi_match(CmpEq.getReg(0), *MRI,
                   m_GICmp(m_Pred(Pred), m_Reg(Reg0), m_Reg(Reg1)));
  EXPECT_TRUE(match);
  EXPECT_EQ(CmpInst::ICMP_EQ, Pred);
  EXPECT_EQ(Copies[0], Reg0);
  EXPECT_EQ(Copies[1], Reg1);
}

TEST_F(AArch64GISelMITest, MatchFCmp) {
  setUp();
  if (!TM)
    return;

  const LLT s1 = LLT::scalar(1);
  auto CmpEq = B.buildFCmp(CmpInst::FCMP_OEQ, s1, Copies[0], Copies[1]);

  // Check match any predicate.
  bool match =
      mi_match(CmpEq.getReg(0), *MRI, m_GFCmp(m_Pred(), m_Reg(), m_Reg()));
  EXPECT_TRUE(match);

  // Check we get the predicate and registers.
  CmpInst::Predicate Pred;
  Register Reg0;
  Register Reg1;
  match = mi_match(CmpEq.getReg(0), *MRI,
                   m_GFCmp(m_Pred(Pred), m_Reg(Reg0), m_Reg(Reg1)));
  EXPECT_TRUE(match);
  EXPECT_EQ(CmpInst::FCMP_OEQ, Pred);
  EXPECT_EQ(Copies[0], Reg0);
  EXPECT_EQ(Copies[1], Reg1);
}

TEST_F(AArch64GISelMITest, MatchFPUnaryOp) {
  setUp();
  if (!TM)
    return;

  // Truncate s64 to s32.
  LLT s32 = LLT::scalar(32);
  auto Copy0s32 = B.buildFPTrunc(s32, Copies[0]);

  // Match G_FABS.
  auto MIBFabs = B.buildInstr(TargetOpcode::G_FABS, {s32}, {Copy0s32});
  bool match =
      mi_match(MIBFabs.getReg(0), *MRI, m_GFabs(m_Reg()));
  EXPECT_TRUE(match);

  Register Src;
  auto MIBFNeg = B.buildInstr(TargetOpcode::G_FNEG, {s32}, {Copy0s32});
  match = mi_match(MIBFNeg.getReg(0), *MRI, m_GFNeg(m_Reg(Src)));
  EXPECT_TRUE(match);
  EXPECT_EQ(Src, Copy0s32.getReg(0));

  match = mi_match(MIBFabs.getReg(0), *MRI, m_GFabs(m_Reg(Src)));
  EXPECT_TRUE(match);
  EXPECT_EQ(Src, Copy0s32.getReg(0));

  // Build and match FConstant.
  auto MIBFCst = B.buildFConstant(s32, .5);
  const ConstantFP *TmpFP{};
  match = mi_match(MIBFCst.getReg(0), *MRI, m_GFCst(TmpFP));
  EXPECT_TRUE(match);
  EXPECT_TRUE(TmpFP);
  APFloat APF((float).5);
  auto *CFP = ConstantFP::get(Context, APF);
  EXPECT_EQ(CFP, TmpFP);

  // Build double float.
  LLT s64 = LLT::scalar(64);
  auto MIBFCst64 = B.buildFConstant(s64, .5);
  const ConstantFP *TmpFP64{};
  match = mi_match(MIBFCst64.getReg(0), *MRI, m_GFCst(TmpFP64));
  EXPECT_TRUE(match);
  EXPECT_TRUE(TmpFP64);
  APFloat APF64(.5);
  auto CFP64 = ConstantFP::get(Context, APF64);
  EXPECT_EQ(CFP64, TmpFP64);
  EXPECT_NE(TmpFP64, TmpFP);

  // Build half float.
  LLT s16 = LLT::scalar(16);
  auto MIBFCst16 = B.buildFConstant(s16, .5);
  const ConstantFP *TmpFP16{};
  match = mi_match(MIBFCst16.getReg(0), *MRI, m_GFCst(TmpFP16));
  EXPECT_TRUE(match);
  EXPECT_TRUE(TmpFP16);
  bool Ignored;
  APFloat APF16(.5);
  APF16.convert(APFloat::IEEEhalf(), APFloat::rmNearestTiesToEven, &Ignored);
  auto CFP16 = ConstantFP::get(Context, APF16);
  EXPECT_EQ(TmpFP16, CFP16);
  EXPECT_NE(TmpFP16, TmpFP);
}

TEST_F(AArch64GISelMITest, MatchExtendsTrunc) {
  setUp();
  if (!TM)
    return;

  LLT s64 = LLT::scalar(64);
  LLT s32 = LLT::scalar(32);

  auto MIBTrunc = B.buildTrunc(s32, Copies[0]);
  auto MIBAExt = B.buildAnyExt(s64, MIBTrunc);
  auto MIBZExt = B.buildZExt(s64, MIBTrunc);
  auto MIBSExt = B.buildSExt(s64, MIBTrunc);
  Register Src0;
  bool match =
      mi_match(MIBTrunc.getReg(0), *MRI, m_GTrunc(m_Reg(Src0)));
  EXPECT_TRUE(match);
  EXPECT_EQ(Src0, Copies[0]);
  match =
      mi_match(MIBAExt.getReg(0), *MRI, m_GAnyExt(m_Reg(Src0)));
  EXPECT_TRUE(match);
  EXPECT_EQ(Src0, MIBTrunc.getReg(0));

  match = mi_match(MIBSExt.getReg(0), *MRI, m_GSExt(m_Reg(Src0)));
  EXPECT_TRUE(match);
  EXPECT_EQ(Src0, MIBTrunc.getReg(0));

  match = mi_match(MIBZExt.getReg(0), *MRI, m_GZExt(m_Reg(Src0)));
  EXPECT_TRUE(match);
  EXPECT_EQ(Src0, MIBTrunc.getReg(0));

  // Match ext(trunc src)
  match = mi_match(MIBAExt.getReg(0), *MRI,
                   m_GAnyExt(m_GTrunc(m_Reg(Src0))));
  EXPECT_TRUE(match);
  EXPECT_EQ(Src0, Copies[0]);

  match = mi_match(MIBSExt.getReg(0), *MRI,
                   m_GSExt(m_GTrunc(m_Reg(Src0))));
  EXPECT_TRUE(match);
  EXPECT_EQ(Src0, Copies[0]);

  match = mi_match(MIBZExt.getReg(0), *MRI,
                   m_GZExt(m_GTrunc(m_Reg(Src0))));
  EXPECT_TRUE(match);
  EXPECT_EQ(Src0, Copies[0]);
}

TEST_F(AArch64GISelMITest, MatchSpecificType) {
  setUp();
  if (!TM)
    return;

  // Try to match a 64bit add.
  LLT s64 = LLT::scalar(64);
  LLT s32 = LLT::scalar(32);
  auto MIBAdd = B.buildAdd(s64, Copies[0], Copies[1]);
  EXPECT_FALSE(mi_match(MIBAdd.getReg(0), *MRI,
                        m_GAdd(m_SpecificType(s32), m_Reg())));
  EXPECT_TRUE(mi_match(MIBAdd.getReg(0), *MRI,
                       m_GAdd(m_SpecificType(s64), m_Reg())));

  // Try to match the destination type of a bitcast.
  LLT v2s32 = LLT::fixed_vector(2, 32);
  auto MIBCast = B.buildCast(v2s32, Copies[0]);
  EXPECT_TRUE(
      mi_match(MIBCast.getReg(0), *MRI, m_GBitcast(m_Reg())));
  EXPECT_TRUE(
      mi_match(MIBCast.getReg(0), *MRI, m_SpecificType(v2s32)));
  EXPECT_TRUE(
      mi_match(MIBCast.getReg(1), *MRI, m_SpecificType(s64)));

  // Build a PTRToInt and INTTOPTR and match and test them.
  LLT PtrTy = LLT::pointer(0, 64);
  auto MIBIntToPtr = B.buildCast(PtrTy, Copies[0]);
  auto MIBPtrToInt = B.buildCast(s64, MIBIntToPtr);
  Register Src0;

  // match the ptrtoint(inttoptr reg)
  bool match = mi_match(MIBPtrToInt.getReg(0), *MRI,
                        m_GPtrToInt(m_GIntToPtr(m_Reg(Src0))));
  EXPECT_TRUE(match);
  EXPECT_EQ(Src0, Copies[0]);
}

TEST_F(AArch64GISelMITest, MatchCombinators) {
  setUp();
  if (!TM)
    return;

  LLT s64 = LLT::scalar(64);
  LLT s32 = LLT::scalar(32);
  auto MIBAdd = B.buildAdd(s64, Copies[0], Copies[1]);
  Register Src0, Src1;
  bool match =
      mi_match(MIBAdd.getReg(0), *MRI,
               m_all_of(m_SpecificType(s64), m_GAdd(m_Reg(Src0), m_Reg(Src1))));
  EXPECT_TRUE(match);
  EXPECT_EQ(Src0, Copies[0]);
  EXPECT_EQ(Src1, Copies[1]);
  // Check for s32 (which should fail).
  match =
      mi_match(MIBAdd.getReg(0), *MRI,
               m_all_of(m_SpecificType(s32), m_GAdd(m_Reg(Src0), m_Reg(Src1))));
  EXPECT_FALSE(match);
  match =
      mi_match(MIBAdd.getReg(0), *MRI,
               m_any_of(m_SpecificType(s32), m_GAdd(m_Reg(Src0), m_Reg(Src1))));
  EXPECT_TRUE(match);
  EXPECT_EQ(Src0, Copies[0]);
  EXPECT_EQ(Src1, Copies[1]);

  // Match a case where none of the predicates hold true.
  match = mi_match(
      MIBAdd.getReg(0), *MRI,
      m_any_of(m_SpecificType(LLT::scalar(16)), m_GSub(m_Reg(), m_Reg())));
  EXPECT_FALSE(match);
}

TEST_F(AArch64GISelMITest, MatchMiscellaneous) {
  setUp();
  if (!TM)
    return;

  LLT s64 = LLT::scalar(64);
  auto MIBAdd = B.buildAdd(s64, Copies[0], Copies[1]);
  Register Reg = MIBAdd.getReg(0);

  // Only one use of Reg.
  B.buildCast(LLT::pointer(0, 32), MIBAdd);
  EXPECT_TRUE(mi_match(Reg, *MRI, m_OneUse(m_GAdd(m_Reg(), m_Reg()))));
  EXPECT_TRUE(mi_match(Reg, *MRI, m_OneNonDBGUse(m_GAdd(m_Reg(), m_Reg()))));

  // Add multiple debug uses of Reg.
  B.buildInstr(TargetOpcode::DBG_VALUE, {}, {Reg});
  B.buildInstr(TargetOpcode::DBG_VALUE, {}, {Reg});

  EXPECT_FALSE(mi_match(Reg, *MRI, m_OneUse(m_GAdd(m_Reg(), m_Reg()))));
  EXPECT_TRUE(mi_match(Reg, *MRI, m_OneNonDBGUse(m_GAdd(m_Reg(), m_Reg()))));

  // Multiple non-debug uses of Reg.
  B.buildCast(LLT::pointer(1, 32), MIBAdd);
  EXPECT_FALSE(mi_match(Reg, *MRI, m_OneUse(m_GAdd(m_Reg(), m_Reg()))));
  EXPECT_FALSE(mi_match(Reg, *MRI, m_OneNonDBGUse(m_GAdd(m_Reg(), m_Reg()))));
}

TEST_F(AArch64GISelMITest, MatchSpecificConstant) {
  setUp();
  if (!TM)
    return;

  // Basic case: Can we match a G_CONSTANT with a specific value?
  auto FortyTwo = B.buildConstant(LLT::scalar(64), 42);
  EXPECT_TRUE(mi_match(FortyTwo.getReg(0), *MRI, m_SpecificICst(42)));
  EXPECT_FALSE(mi_match(FortyTwo.getReg(0), *MRI, m_SpecificICst(123)));

  // Test that this works inside of a more complex pattern.
  LLT s64 = LLT::scalar(64);
  auto MIBAdd = B.buildAdd(s64, Copies[0], FortyTwo);
  EXPECT_TRUE(mi_match(MIBAdd.getReg(2), *MRI, m_SpecificICst(42)));

  // Wrong constant.
  EXPECT_FALSE(mi_match(MIBAdd.getReg(2), *MRI, m_SpecificICst(123)));

  // No constant on the LHS.
  EXPECT_FALSE(mi_match(MIBAdd.getReg(1), *MRI, m_SpecificICst(42)));
}

TEST_F(AArch64GISelMITest, MatchSpecificConstantSplat) {
  setUp();
  if (!TM)
    return;

  LLT s64 = LLT::scalar(64);
  LLT v4s64 = LLT::fixed_vector(4, s64);

  MachineInstrBuilder FortyTwoSplat =
      B.buildSplatVector(v4s64, B.buildConstant(s64, 42));
  MachineInstrBuilder FortyTwo = B.buildConstant(s64, 42);

  EXPECT_TRUE(mi_match(FortyTwoSplat.getReg(0), *MRI, m_SpecificICstSplat(42)));
  EXPECT_FALSE(
      mi_match(FortyTwoSplat.getReg(0), *MRI, m_SpecificICstSplat(43)));
  EXPECT_FALSE(mi_match(FortyTwo.getReg(0), *MRI, m_SpecificICstSplat(42)));

  MachineInstrBuilder NonConstantSplat =
      B.buildBuildVector(v4s64, {Copies[0], Copies[0], Copies[0], Copies[0]});

  MachineInstrBuilder AddSplat =
      B.buildAdd(v4s64, NonConstantSplat, FortyTwoSplat);
  EXPECT_TRUE(mi_match(AddSplat.getReg(2), *MRI, m_SpecificICstSplat(42)));
  EXPECT_FALSE(mi_match(AddSplat.getReg(2), *MRI, m_SpecificICstSplat(43)));
  EXPECT_FALSE(mi_match(AddSplat.getReg(1), *MRI, m_SpecificICstSplat(42)));

  MachineInstrBuilder Add = B.buildAdd(s64, Copies[0], FortyTwo);
  EXPECT_FALSE(mi_match(Add.getReg(2), *MRI, m_SpecificICstSplat(42)));
}

TEST_F(AArch64GISelMITest, MatchSpecificConstantOrSplat) {
  setUp();
  if (!TM)
    return;

  LLT s64 = LLT::scalar(64);
  LLT v4s64 = LLT::fixed_vector(4, s64);

  MachineInstrBuilder FortyTwoSplat =
      B.buildSplatVector(v4s64, B.buildConstant(s64, 42));
  MachineInstrBuilder FortyTwo = B.buildConstant(s64, 42);

  EXPECT_TRUE(
      mi_match(FortyTwoSplat.getReg(0), *MRI, m_SpecificICstOrSplat(42)));
  EXPECT_FALSE(
      mi_match(FortyTwoSplat.getReg(0), *MRI, m_SpecificICstOrSplat(43)));
  EXPECT_TRUE(mi_match(FortyTwo.getReg(0), *MRI, m_SpecificICstOrSplat(42)));

  MachineInstrBuilder NonConstantSplat =
      B.buildBuildVector(v4s64, {Copies[0], Copies[0], Copies[0], Copies[0]});

  MachineInstrBuilder AddSplat =
      B.buildAdd(v4s64, NonConstantSplat, FortyTwoSplat);
  EXPECT_TRUE(mi_match(AddSplat.getReg(2), *MRI, m_SpecificICstOrSplat(42)));
  EXPECT_FALSE(mi_match(AddSplat.getReg(2), *MRI, m_SpecificICstOrSplat(43)));
  EXPECT_FALSE(mi_match(AddSplat.getReg(1), *MRI, m_SpecificICstOrSplat(42)));

  MachineInstrBuilder Add = B.buildAdd(s64, Copies[0], FortyTwo);
  EXPECT_TRUE(mi_match(Add.getReg(2), *MRI, m_SpecificICstOrSplat(42)));
}

TEST_F(AArch64GISelMITest, MatchZeroInt) {
  setUp();
  if (!TM)
    return;
  auto Zero = B.buildConstant(LLT::scalar(64), 0);
  EXPECT_TRUE(mi_match(Zero.getReg(0), *MRI, m_ZeroInt()));

  auto FortyTwo = B.buildConstant(LLT::scalar(64), 42);
  EXPECT_FALSE(mi_match(FortyTwo.getReg(0), *MRI, m_ZeroInt()));
}

TEST_F(AArch64GISelMITest, MatchAllOnesInt) {
  setUp();
  if (!TM)
    return;
  auto AllOnes = B.buildConstant(LLT::scalar(64), -1);
  EXPECT_TRUE(mi_match(AllOnes.getReg(0), *MRI, m_AllOnesInt()));

  auto FortyTwo = B.buildConstant(LLT::scalar(64), 42);
  EXPECT_FALSE(mi_match(FortyTwo.getReg(0), *MRI, m_AllOnesInt()));
}

TEST_F(AArch64GISelMITest, MatchFPOrIntConst) {
  setUp();
  if (!TM)
    return;

  Register IntOne = B.buildConstant(LLT::scalar(64), 1).getReg(0);
  Register FPOne = B.buildFConstant(LLT::scalar(64), 1.0).getReg(0);
  Optional<ValueAndVReg> ValReg;
  Optional<FPValueAndVReg> FValReg;

  EXPECT_TRUE(mi_match(IntOne, *MRI, m_GCst(ValReg)));
  EXPECT_EQ(IntOne, ValReg->VReg);
  EXPECT_FALSE(mi_match(IntOne, *MRI, m_GFCst(FValReg)));

  EXPECT_FALSE(mi_match(FPOne, *MRI, m_GCst(ValReg)));
  EXPECT_TRUE(mi_match(FPOne, *MRI, m_GFCst(FValReg)));
  EXPECT_EQ(FPOne, FValReg->VReg);
}

TEST_F(AArch64GISelMITest, MatchConstantSplat) {
  setUp();
  if (!TM)
    return;

  LLT s64 = LLT::scalar(64);
  LLT v4s64 = LLT::fixed_vector(4, 64);

  Register FPOne = B.buildFConstant(s64, 1.0).getReg(0);
  Register FPZero = B.buildFConstant(s64, 0.0).getReg(0);
  Register Undef = B.buildUndef(s64).getReg(0);
  Optional<FPValueAndVReg> FValReg;

  // GFCstOrSplatGFCstMatch allows undef as part of splat. Undef often comes
  // from padding to legalize into available operation and then ignore added
  // elements e.g. v3s64 to v4s64.

  EXPECT_TRUE(mi_match(FPZero, *MRI, GFCstOrSplatGFCstMatch(FValReg)));
  EXPECT_EQ(FPZero, FValReg->VReg);

  EXPECT_FALSE(mi_match(Undef, *MRI, GFCstOrSplatGFCstMatch(FValReg)));

  auto ZeroSplat = B.buildBuildVector(v4s64, {FPZero, FPZero, FPZero, FPZero});
  EXPECT_TRUE(
      mi_match(ZeroSplat.getReg(0), *MRI, GFCstOrSplatGFCstMatch(FValReg)));
  EXPECT_EQ(FPZero, FValReg->VReg);

  auto ZeroUndef = B.buildBuildVector(v4s64, {FPZero, FPZero, FPZero, Undef});
  EXPECT_TRUE(
      mi_match(ZeroUndef.getReg(0), *MRI, GFCstOrSplatGFCstMatch(FValReg)));
  EXPECT_EQ(FPZero, FValReg->VReg);

  // All undefs are not constant splat.
  auto UndefSplat = B.buildBuildVector(v4s64, {Undef, Undef, Undef, Undef});
  EXPECT_FALSE(
      mi_match(UndefSplat.getReg(0), *MRI, GFCstOrSplatGFCstMatch(FValReg)));

  auto ZeroOne = B.buildBuildVector(v4s64, {FPZero, FPZero, FPZero, FPOne});
  EXPECT_FALSE(
      mi_match(ZeroOne.getReg(0), *MRI, GFCstOrSplatGFCstMatch(FValReg)));

  auto NonConstantSplat =
      B.buildBuildVector(v4s64, {Copies[0], Copies[0], Copies[0], Copies[0]});
  EXPECT_FALSE(mi_match(NonConstantSplat.getReg(0), *MRI,
                        GFCstOrSplatGFCstMatch(FValReg)));

  auto Mixed = B.buildBuildVector(v4s64, {FPZero, FPZero, FPZero, Copies[0]});
  EXPECT_FALSE(
      mi_match(Mixed.getReg(0), *MRI, GFCstOrSplatGFCstMatch(FValReg)));
}

TEST_F(AArch64GISelMITest, MatchNeg) {
  setUp();
  if (!TM)
    return;

  LLT s64 = LLT::scalar(64);
  auto Zero = B.buildConstant(LLT::scalar(64), 0);
  auto NegInst = B.buildSub(s64, Zero, Copies[0]);
  Register NegatedReg;

  // Match: G_SUB = 0, %Reg
  EXPECT_TRUE(mi_match(NegInst.getReg(0), *MRI, m_Neg(m_Reg(NegatedReg))));
  EXPECT_EQ(NegatedReg, Copies[0]);

  // Don't match: G_SUB = %Reg, 0
  auto NotNegInst1 = B.buildSub(s64, Copies[0], Zero);
  EXPECT_FALSE(mi_match(NotNegInst1.getReg(0), *MRI, m_Neg(m_Reg(NegatedReg))));

  // Don't match: G_SUB = 42, %Reg
  auto FortyTwo = B.buildConstant(LLT::scalar(64), 42);
  auto NotNegInst2 = B.buildSub(s64, FortyTwo, Copies[0]);
  EXPECT_FALSE(mi_match(NotNegInst2.getReg(0), *MRI, m_Neg(m_Reg(NegatedReg))));

  // Complex testcase.
  // %sub = G_SUB = 0, %negated_reg
  // %add = G_ADD = %x, %sub
  auto AddInst = B.buildAdd(s64, Copies[1], NegInst);
  NegatedReg = Register();
  EXPECT_TRUE(mi_match(AddInst.getReg(2), *MRI, m_Neg(m_Reg(NegatedReg))));
  EXPECT_EQ(NegatedReg, Copies[0]);
}

TEST_F(AArch64GISelMITest, MatchNot) {
  setUp();
  if (!TM)
    return;

  LLT s64 = LLT::scalar(64);
  auto AllOnes = B.buildConstant(LLT::scalar(64), -1);
  auto NotInst1 = B.buildXor(s64, Copies[0], AllOnes);
  Register NotReg;

  // Match: G_XOR %NotReg, -1
  EXPECT_TRUE(mi_match(NotInst1.getReg(0), *MRI, m_Not(m_Reg(NotReg))));
  EXPECT_EQ(NotReg, Copies[0]);

  // Match: G_XOR -1, %NotReg
  auto NotInst2 = B.buildXor(s64, AllOnes, Copies[1]);
  EXPECT_TRUE(mi_match(NotInst2.getReg(0), *MRI, m_Not(m_Reg(NotReg))));
  EXPECT_EQ(NotReg, Copies[1]);

  // Don't match: G_XOR %NotReg, 42
  auto FortyTwo = B.buildConstant(LLT::scalar(64), 42);
  auto WrongCst = B.buildXor(s64, Copies[0], FortyTwo);
  EXPECT_FALSE(mi_match(WrongCst.getReg(0), *MRI, m_Not(m_Reg(NotReg))));

  // Complex testcase.
  // %xor = G_XOR %NotReg, -1
  // %add = G_ADD %x, %xor
  auto AddInst = B.buildAdd(s64, Copies[1], NotInst1);
  NotReg = Register();
  EXPECT_TRUE(mi_match(AddInst.getReg(2), *MRI, m_Not(m_Reg(NotReg))));
  EXPECT_EQ(NotReg, Copies[0]);
}
} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  initLLVM();
  return RUN_ALL_TESTS();
}

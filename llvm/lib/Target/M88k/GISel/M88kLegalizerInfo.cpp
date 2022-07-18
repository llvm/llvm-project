//===-- M88kLegalizerInfo.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the targeting of the Machinelegalizer class for M88k.
//===----------------------------------------------------------------------===//

#include "M88kLegalizerInfo.h"
#include "M88kSubtarget.h"
#include "llvm/CodeGen/GlobalISel/LegalizerHelper.h"
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"

using namespace llvm;
using namespace LegalityPredicates;

M88kLegalizerInfo::M88kLegalizerInfo(const M88kSubtarget &ST) {
  using namespace TargetOpcode;
  const LLT S1 = LLT::scalar(1);
  const LLT S8 = LLT::scalar(8);
  const LLT S16 = LLT::scalar(16);
  const LLT S32 = LLT::scalar(32);
  const LLT S64 = LLT::scalar(64);
  const LLT S80 = LLT::scalar(80);
  const LLT P0 = LLT::pointer(0, 32);
  getActionDefinitionsBuilder(G_PHI).legalFor({S32, P0});
  getActionDefinitionsBuilder(G_SELECT)
      .customForCartesianProduct({S32, S64, P0}, {S1});
  getActionDefinitionsBuilder({G_IMPLICIT_DEF, G_FREEZE}).legalFor({S32});
  getActionDefinitionsBuilder(G_MERGE_VALUES).legalFor({{S64, S32}});
  getActionDefinitionsBuilder(G_UNMERGE_VALUES).legalFor({{S32, S64}});
  getActionDefinitionsBuilder(G_CONSTANT)
      .legalFor({S32, P0})
      .clampScalar(0, S32, S32);
  getActionDefinitionsBuilder(G_INTTOPTR)
      .legalFor({{P0, S32}})
      .minScalar(1, S32);
  getActionDefinitionsBuilder(G_PTRTOINT)
      .legalFor({{S32, P0}})
      .minScalar(0, S32);

  getActionDefinitionsBuilder({G_SEXT, G_ZEXT, G_ANYEXT})
      .legalFor({{S32, S16}, {S32, S8}, {S32, S1}})
      .clampScalar(0, S32, S32)
      .widenScalarToNextPow2(1, 32);
  getActionDefinitionsBuilder(G_SEXT_INREG).legalForTypeWithAnyImm({S32});
  getActionDefinitionsBuilder(G_TRUNC).alwaysLegal();
  getActionDefinitionsBuilder({G_SEXTLOAD, G_ZEXTLOAD})
      .legalForTypesWithMemDesc({{S32, P0, S8, 8}, {S32, P0, S16, 16}});
  getActionDefinitionsBuilder({G_LOAD, G_STORE})
      .legalForTypesWithMemDesc({{S32, P0, S8, 8},
                                 {S32, P0, S16, 16},
                                 {S32, P0, S32, 32},
                                 {P0, P0, P0, 32},
                                 {S64, P0, S64, 64}})
      .unsupportedIfMemSizeNotPow2()
      .minScalar(0, S32);
  getActionDefinitionsBuilder(G_PTR_ADD)
      .legalFor({{P0, S32}})
      .clampScalar(1, S32, S32);
  getActionDefinitionsBuilder(G_ADD).legalFor({S32});
  getActionDefinitionsBuilder(G_SUB).legalFor({S32});
  getActionDefinitionsBuilder(G_MUL).legalFor({S32});
  getActionDefinitionsBuilder(G_UDIV).legalFor({S32}).libcallFor({S64});
  getActionDefinitionsBuilder(G_SDIV)
#if 0
      // The only problem with these 2 rules is that the custom code requires
      // inserting new basic blocks. This is currently not supported by the
      // Legalizer.
      .legalIf(all(typeInSet(0, {S32}),
                   LegalityPredicate(([=, &ST](const LegalityQuery &Query) {
                     return ST.useDivInstr() || ST.isMC88110();
                   }))))
      .customIf(all(typeInSet(0, {S32}),
                    LegalityPredicate(([=, &ST](const LegalityQuery &Query) {
                      return !ST.useDivInstr();
                    }))))
#else
      .legalFor({S32})
#endif
      .libcallFor({S64});
  getActionDefinitionsBuilder({G_AND, G_OR, G_XOR})
      .legalFor({S32})
      .clampScalar(0, S32, S32);
  getActionDefinitionsBuilder({G_SBFX, G_UBFX})
      .legalFor({{S32, S32}})
      .clampScalar(0, S32, S32);
  getActionDefinitionsBuilder({G_SHL, G_LSHR, G_ASHR})
      .legalFor({{S32, S32}})
      .clampScalar(0, S32, S32)
      .clampScalar(1, S32, S32);
  getActionDefinitionsBuilder(G_ROTR).legalFor({{S32}, {S32}});
  getActionDefinitionsBuilder({G_ROTL, G_FSHL, G_FSHR}).lower();

  getActionDefinitionsBuilder(G_ICMP)
      .legalForCartesianProduct({S1}, {S32, P0})
      .clampScalar(1, S32, S32);
  getActionDefinitionsBuilder(G_BRCOND).legalFor({S1});
  getActionDefinitionsBuilder(G_BRJT).legalFor({{P0, S32}});
  getActionDefinitionsBuilder(G_BRINDIRECT).legalFor({P0});
  getActionDefinitionsBuilder(G_JUMP_TABLE).legalFor({P0});

  getActionDefinitionsBuilder(G_FRAME_INDEX).legalFor({P0});
  getActionDefinitionsBuilder(G_GLOBAL_VALUE).legalFor({P0});

  getActionDefinitionsBuilder(G_FCONSTANT).customFor({S32, S64});
  getActionDefinitionsBuilder({G_FADD, G_FSUB, G_FMUL, G_FDIV})
      .legalFor({S32, S64});
  getActionDefinitionsBuilder({G_FNEG, G_FABS}).lower();
  getActionDefinitionsBuilder(G_FPEXT).legalFor({{S64, S32}});
  getActionDefinitionsBuilder(G_FPTRUNC).legalFor({{S32, S64}});

  // FP to int conversion instructions
  getActionDefinitionsBuilder(G_FPTOSI)
      .legalForCartesianProduct({S32}, {S64, S32})
      .libcallForCartesianProduct({S64}, {S64, S32})
      .minScalar(0, S32);

  getActionDefinitionsBuilder(G_FPTOUI)
      .libcallForCartesianProduct({S64}, {S64, S32})
      .lowerForCartesianProduct({S32}, {S64, S32})
      .minScalar(0, S32);

  // Int to FP conversion instructions
  getActionDefinitionsBuilder(G_SITOFP)
      .legalForCartesianProduct({S64, S32}, {S32})
      .libcallForCartesianProduct({S64, S32}, {S64})
      .minScalar(1, S32);
/*
  getActionDefinitionsBuilder(G_UITOFP)
      .libcallForCartesianProduct({S64, S32}, {S64})
      .customForCartesianProduct({S64, S32}, {S32})
      .minScalar(1, S32);
*/
  getLegacyLegalizerInfo().computeTables();
}

bool M88kLegalizerInfo::legalizeCustom(LegalizerHelper &Helper,
                                       MachineInstr &MI) const {
  using namespace TargetOpcode;

  MachineIRBuilder &MIRBuilder = Helper.MIRBuilder;
  MachineRegisterInfo &MRI = *MIRBuilder.getMRI();

  const LLT S1 = LLT::scalar(1);
  const LLT S32 = LLT::scalar(32);
  const LLT S64 = LLT::scalar(64);

  switch (MI.getOpcode()) {
  case G_FCONSTANT: {
    LLVMContext &Ctx = MIRBuilder.getMF().getFunction().getContext();
    // Convert to integer constants, while preserving the binary representation.
    auto AsInteger =
        MI.getOperand(1).getFPImm()->getValueAPF().bitcastToAPInt();
    MIRBuilder.buildConstant(MI.getOperand(0),
                             *ConstantInt::get(Ctx, AsInteger));
    MI.eraseFromParent();
    break;
  }
  case G_SELECT: {
    using namespace MIPatternMatch;
    // The instruction
    // %4:_(s32) = G_SELECT %1:_(s32), %2:_(s32), %3:_(s32)
    // is lowered to:
    // %5:_(s32) = G_SEXT_INREG %1:_(s32), 0
    // %6:_(s32) = G_AND %5:_(s32), %2:_(s32)
    // %7:_(s32) = G_XOR %5:_(s32), -1
    // %8:_(s32) = G_AND %5:_(s32), %3:_(s32)
    // %4:_(s32) = G_OR %6:_(s32), %8:_(s32)
    //
    // If one of the values to select is zero, then the G_AND belonging to that
    // value is not generated.
    Register Dst = MI.getOperand(0).getReg();
    Register Tst = MI.getOperand(1).getReg();
    Register TVal = MI.getOperand(2).getReg();
    Register FVal = MI.getOperand(3).getReg();
    LLT DstTy = MRI.getType(Dst);
    LLT TstTy = MRI.getType(Tst);
    //LLT TValTy = MRI.getType(TVal);
    //LLT FValy = MRI.getType(FVal);
    if (TstTy != S1)
      return false;
    if (DstTy != S32 && DstTy != S64)
      return false;
    int64_t Cst;
    bool MissT = mi_match(TVal, MRI, m_ICst(Cst)) && Cst == 0;
    bool MissF = mi_match(FVal, MRI, m_ICst(Cst)) && Cst == 0;
    if (MissT && MissF) {
      MIRBuilder.buildConstant(Dst, 0);
    } else {
      auto Mask = MIRBuilder.buildSExtInReg(S32, Tst, 0);
      if (MissF) {
        MIRBuilder.buildAnd(Dst, TVal, Mask);
      } else if (MissT) {
        auto NegMask = MIRBuilder.buildNot(S32, Mask);
        MIRBuilder.buildAnd(Dst, FVal, NegMask);
      } else {
        auto MaskT = MIRBuilder.buildAnd(S32, TVal, Mask);
        auto NegMask = MIRBuilder.buildNot(S32, Mask);
        auto MaskF = MIRBuilder.buildAnd(S32, FVal, NegMask);
        MIRBuilder.buildOr(Dst, MaskT, MaskF);
      }
    }
    MI.eraseFromParent();
    break;
  }
#if 0
  // This is a first attempt to lower G_SDIV to G_UDIV.
  // It does not work because the Legalizer currently does not support inserting
  // new basic blocks.
  case G_SDIV: {
    // The instruction
    //   %3:_ = G_SDIV %1:_, %2:_
    // is lowered to G_UDIV. This is done by making both operands non-negative,
    // and correcting the sign of the result. All 4 cases need to be
    // considered. The code is a bit lengthy, but not complicated.
    Register Dst = MI.getOperand(0).getReg();
    Register Dividend = MI.getOperand(1).getReg();
    Register Divisor = MI.getOperand(2).getReg();

    if (MRI.getType(Dst) != S32 || MRI.getType(Dividend) != S32 ||
        MRI.getType(Divisor) != S32)
      return false;

    MachineBasicBlock *CurBB = MI.getParent();
    MachineFunction *MF = CurBB->getParent();

    // First split the current basic block. Move all instructions after MI into
    // the new block.
    MachineFunction::iterator BBI(CurBB);
    MachineBasicBlock *TailBB =
        MF->CreateMachineBasicBlock(CurBB->getBasicBlock());
    MF->insert(++BBI, TailBB);
    MachineBasicBlock::iterator MII(&MI);
    TailBB->splice(TailBB->begin(), CurBB, ++MII, CurBB->end());

    // Create basic blocks.
    MachineBasicBlock *DivisorNotPosBB =
        MF->CreateMachineBasicBlock(CurBB->getBasicBlock());
    MachineBasicBlock *DividendNegDivisorPosBB =
        MF->CreateMachineBasicBlock(CurBB->getBasicBlock());
    MachineBasicBlock *DividendNegDivisorNotPosBB =
        MF->CreateMachineBasicBlock(CurBB->getBasicBlock());

    MF->insert(--BBI, DivisorNotPosBB);
    MF->insert(BBI, DividendNegDivisorPosBB);
    MF->insert(BBI, DividendNegDivisorNotPosBB);

    // Compare dividend and divisor against zero, and branch accordingly.
    // Handle case 1: dividend zero or positiv, divisor positiv.
    auto Zero = MIRBuilder.buildConstant(S32, 0);
    auto Cmp1 = MIRBuilder.buildICmp(CmpInst::ICMP_SLE, S1, Divisor, Zero);
    MIRBuilder.buildBrCond(Cmp1, *DivisorNotPosBB);
    auto Cmp2 = MIRBuilder.buildICmp(CmpInst::ICMP_SLT, S1, Dividend, Zero);
    MIRBuilder.buildBrCond(Cmp2, *DividendNegDivisorPosBB);
    auto Div1 =
        MIRBuilder.buildInstr(TargetOpcode::G_UDIV, {S32}, {Dividend, Divisor});
    MIRBuilder.buildBr(*TailBB);

    // Handle case 2: dividend zero or positiv, divisor negativ.
    // Result is negativ.
    MIRBuilder.setMBB(*DivisorNotPosBB);
    auto AbsDivisor = MIRBuilder.buildNeg(S32, Divisor);
    auto Cmp3 = MIRBuilder.buildICmp(CmpInst::ICMP_SLT, S1, Dividend, Zero);
    MIRBuilder.buildBrCond(Cmp2, *DividendNegDivisorNotPosBB);
    auto Div2 = MIRBuilder.buildNeg(
        S32, MIRBuilder.buildInstr(TargetOpcode::G_UDIV, {S32},
                                   {Dividend, AbsDivisor}));
    MIRBuilder.buildBr(*TailBB);

    // Handle case 3: dividend negativ, divisor negativ.
    // Result is positiv.
    MIRBuilder.setMBB(*DividendNegDivisorNotPosBB);
    auto AbsDividend = MIRBuilder.buildNeg(S32, Dividend);
    auto Cmp4 = MIRBuilder.buildICmp(CmpInst::ICMP_SLT, S1, Dividend, Zero);
    MIRBuilder.buildBrCond(Cmp4, *DividendNegDivisorNotPosBB);
    auto Div3 = MIRBuilder.buildInstr(TargetOpcode::G_UDIV, {S32},
                                      {AbsDividend, AbsDivisor});
    MIRBuilder.buildBr(*TailBB);

    // Handle case 4: dividend negativ, divisor zero or positiv.
    // Result is negativ.
    MIRBuilder.setMBB(*DividendNegDivisorPosBB);
    auto AbsDividend2 = MIRBuilder.buildNeg(S32, Dividend);
    auto Cmp5 = MIRBuilder.buildICmp(CmpInst::ICMP_SLT, S1, Dividend, Zero);
    MIRBuilder.buildBrCond(Cmp5, *DividendNegDivisorNotPosBB);
    auto Div4 = MIRBuilder.buildNeg(
        S32, MIRBuilder.buildInstr(TargetOpcode::G_UDIV, {S32},
                                   {AbsDividend, Divisor}));
    MIRBuilder.buildBr(*TailBB);

    // Collect the results.
    MIRBuilder.setInsertPt(*TailBB, TailBB->begin());
    auto Phi = MIRBuilder.buildInstr(TargetOpcode::G_PHI, {Dst}, {});
    Phi.addUse(Div1.getReg(0)).addMBB(CurBB);
    Phi.addUse(Div2.getReg(0)).addMBB(DivisorNotPosBB);
    Phi.addUse(Div3.getReg(0)).addMBB(DividendNegDivisorNotPosBB);
    Phi.addUse(Div4.getReg(0)).addMBB(DividendNegDivisorPosBB);

    MI.eraseFromParent();
    break;
  }
#endif
  default:
    return false;
  }

  return true;
}
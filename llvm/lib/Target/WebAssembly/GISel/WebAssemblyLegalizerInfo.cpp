//===- WebAssemblyLegalizerInfo.cpp ------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the targeting of the Machinelegalizer class for
/// WebAssembly.
//===----------------------------------------------------------------------===//

#include "WebAssemblyLegalizerInfo.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssemblySubtarget.h"
#include "llvm/CodeGen/GlobalISel/LegalizerHelper.h"
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"

#define DEBUG_TYPE "wasm-legalinfo"

using namespace llvm;
using namespace LegalizeActions;

WebAssemblyLegalizerInfo::WebAssemblyLegalizerInfo(
    const WebAssemblySubtarget &ST) {
  using namespace TargetOpcode;

  const LLT i32 = LLT::integer(32);
  const LLT i64 = LLT::integer(64);

  const LLT f32 = LLT::floatIEEE(32);
  const LLT f64 = LLT::floatIEEE(64);

  const LLT s32 = LLT::scalar(32);
  const LLT s64 = LLT::scalar(64);

  getActionDefinitionsBuilder(G_IMPLICIT_DEF)
      .legalFor({i32, i64, f32, f64})
      .widenScalarToNextPow2(0)
      .clampScalar(0, s32, s64);

  getActionDefinitionsBuilder({G_CONSTANT, G_ADD, G_SUB, G_MUL, G_UDIV, G_SDIV,
                               G_UREM, G_SREM, G_AND, G_OR, G_XOR})
      .legalFor({i32, i64})
      .widenScalarToNextPow2(0)
      .clampScalar(0, s32, s64);

  getActionDefinitionsBuilder({G_ASHR, G_LSHR, G_SHL})
      .legalFor({{i32, i32}, {i64, i64}})
      .widenScalarToNextPow2(0)
      .clampScalar(0, s32, s64)
      .scalarSameSizeAs(1, 0);

  getActionDefinitionsBuilder({G_CTLZ, G_CTTZ, G_CTPOP})
      .legalFor({{i32, i32}, {i64, i64}})
      .widenScalarToNextPow2(1)
      .clampScalar(1, s32, s64)
      .scalarSameSizeAs(0, 1);

  getActionDefinitionsBuilder({G_CTLZ_ZERO_POISON, G_CTTZ_ZERO_POISON}).lower();

  getActionDefinitionsBuilder({G_ROTL, G_ROTR})
      .legalFor({{i32, i32}, {i64, i64}})
      .scalarSameSizeAs(1, 0)
      .lower();

  getActionDefinitionsBuilder({G_FSHL, G_FSHR}).lower();

  getActionDefinitionsBuilder(G_ICMP)
      .legalForCartesianProduct({i32}, {i32, i64})
      .widenScalarToNextPow2(1)
      .clampScalar(0, s32, s32)
      .clampScalar(1, s32, s64);

  getActionDefinitionsBuilder({G_UMIN, G_UMAX, G_SMIN, G_SMAX}).lower();

  getActionDefinitionsBuilder({G_SCMP, G_UCMP}).lower();

  getActionDefinitionsBuilder({G_ANYEXT, G_SEXT, G_ZEXT})
      .legalFor({{i64, i32}})
      .clampScalar(0, s64, s64)
      .clampScalar(1, s32, s32);

  getActionDefinitionsBuilder(G_TRUNC)
      .legalFor({{i32, i64}})
      .clampScalar(0, s32, s32)
      .clampScalar(1, s64, s64);

  getActionDefinitionsBuilder(G_SEXT_INREG)
      .customFor(ST.hasSignExt(), {i32, i64})
      .clampScalar(0, s32, s64)
      .lower();

  getActionDefinitionsBuilder({G_FCONSTANT, G_FABS, G_FNEG, G_FCEIL, G_FFLOOR,
                               G_INTRINSIC_TRUNC, G_FNEARBYINT, G_FRINT,
                               G_INTRINSIC_ROUNDEVEN, G_FSQRT, G_FADD, G_FSUB,
                               G_FMUL, G_FDIV})
      .legalFor({f32, f64})
      .minScalar(0, s32);

  getActionDefinitionsBuilder(G_FCOPYSIGN)
      .legalFor({f32, f64})
      .minScalar(0, s32)
      .scalarSameSizeAs(1, 0);

  getActionDefinitionsBuilder(G_FCMP)
      .customForCartesianProduct({i32}, {f32, f64})
      .clampScalar(0, s32, s32);

  getActionDefinitionsBuilder({G_FMINIMUM, G_FMAXIMUM})
      .legalFor({f32, f64})
      .minScalar(0, s32);

  getActionDefinitionsBuilder(
      {G_FMINNUM, G_FMAXNUM, G_FMINIMUMNUM, G_FMAXIMUMNUM})
      .customFor({f32, f64})
      .minScalar(0, s32);

  getActionDefinitionsBuilder(G_IS_FPCLASS)
      .lowerForCartesianProduct({i32}, {f32, f64})
      .clampScalar(0, s32, s32);

  getActionDefinitionsBuilder(G_FPEXT)
      .legalFor({{f64, f32}})
      .clampScalar(0, s64, s64)
      .clampScalar(1, s32, s32);

  getActionDefinitionsBuilder(G_FPTRUNC)
      .legalFor({{f32, f64}})
      .clampScalar(0, s32, s32)
      .clampScalar(1, s64, s64);

  getActionDefinitionsBuilder(G_BITCAST)
      .legalFor({{i32, f32}, {f32, i32}, {i64, f64}, {f64, i64}})
      .clampScalar(0, s32, s64)
      .clampScalar(1, s32, s64);

  getActionDefinitionsBuilder({G_FPTOSI, G_FPTOUI})
      .legalForCartesianProduct({i32, i64}, {f32, f64})
      .clampScalar(0, s32, s64)
      .minScalar(1, s32);

  // TODO: once comparison ops are in place
  /*if (ST.hasNontrappingFPToInt()) {
    getActionDefinitionsBuilder({G_FPTOSI_SAT, G_FPTOUI_SAT})
        .legalForCartesianProduct({i32, i64}, {f32, f64})
        .clampScalar(0, s32, s64)
        .minScalar(1, s32);
  } else {
    getActionDefinitionsBuilder({G_FPTOSI_SAT, G_FPTOUI_SAT})
        .lowerForCartesianProduct({i32, i64}, {f32, f64})
        .clampScalar(0, s32, s64)
        .minScalar(1, s32);
  }*/

  getActionDefinitionsBuilder({G_SITOFP, G_UITOFP})
      .legalForCartesianProduct({f32, f64}, {i32, i64})
      .minScalar(0, s32)
      .clampScalar(1, s32, s64);

  getActionDefinitionsBuilder(G_SELECT)
      .legalForCartesianProduct({i32, i64, f32, f64}, {i32})
      .widenScalarToNextPow2(0)
      .clampScalar(0, s32, s64)
      .clampScalar(1, s32, s32);

  getLegacyLegalizerInfo().computeTables();
}

bool WebAssemblyLegalizerInfo::legalizeCustom(
    LegalizerHelper &Helper, MachineInstr &MI,
    LostDebugLocObserver &LocObserver) const {
  MachineRegisterInfo &MRI = *Helper.MIRBuilder.getMRI();
  auto &MIRBuilder = Helper.MIRBuilder;

  const LLT i1 = LLT::integer(1);

  switch (MI.getOpcode()) {
  case TargetOpcode::G_SEXT_INREG: {
    assert(MI.getOperand(2).isImm() && "Expected immediate");

    // Mark only 8/16/32-bit SEXT_INREG as legal
    auto [DstType, SrcType] = MI.getFirst2LLTs();
    auto ExtFromWidth = MI.getOperand(2).getImm();

    if (ExtFromWidth == 8 || ExtFromWidth == 16 ||
        (DstType.getScalarSizeInBits() == 64 && ExtFromWidth == 32)) {
      return true;
    }

    return Helper.lower(MI, 0, DstType) != LegalizerHelper::UnableToLegalize;
  }
  case TargetOpcode::G_FCMP: {
    Register LHS = MI.getOperand(2).getReg();
    Register RHS = MI.getOperand(3).getReg();
    CmpInst::Predicate Pred =
        static_cast<CmpInst::Predicate>(MI.getOperand(1).getPredicate());

    Register Result = MI.getOperand(0).getReg();

    switch (Pred) {
    case CmpInst::FCMP_FALSE:
      MIRBuilder.buildBoolExt(Result, MIRBuilder.buildConstant(i1, 0), false);
      break;
    case CmpInst::FCMP_OEQ:
      return true;
    case CmpInst::FCMP_OGT:
      return true;
    case CmpInst::FCMP_OGE:
      return true;
    case CmpInst::FCMP_OLT:
      return true;
    case CmpInst::FCMP_OLE:
      return true;
    case CmpInst::FCMP_ONE: {
      MIRBuilder.buildBoolExt(
          Result,
          MIRBuilder.buildOr(
              i1,
              MIRBuilder.buildFCmp(CmpInst::FCMP_OLT, i1, LHS, RHS).getReg(0),
              MIRBuilder.buildFCmp(CmpInst::FCMP_OGT, i1, LHS, RHS).getReg(0)),
          false);
      break;
    }
    case CmpInst::FCMP_ORD: {
      MIRBuilder.buildBoolExt(
          Result,
          MIRBuilder.buildAnd(
              i1,
              MIRBuilder.buildFCmp(CmpInst::FCMP_OEQ, i1, RHS, RHS).getReg(0),
              MIRBuilder.buildFCmp(CmpInst::FCMP_OEQ, i1, LHS, LHS).getReg(0)),
          false);
      break;
    }
    case CmpInst::FCMP_UNO: {
      MIRBuilder.buildBoolExt(
          Result,
          MIRBuilder.buildOr(
              i1,
              MIRBuilder.buildFCmp(CmpInst::FCMP_UNE, i1, RHS, RHS).getReg(0),
              MIRBuilder.buildFCmp(CmpInst::FCMP_UNE, i1, LHS, LHS).getReg(0)),
          false);
      break;
    }
    case CmpInst::FCMP_UEQ: {
      MIRBuilder.buildBoolExt(
          Result,
          MIRBuilder.buildNot(
              i1,
              MIRBuilder.buildFCmp(CmpInst::FCMP_ONE, i1, LHS, RHS).getReg(0)),
          false);
      break;
    }
    case CmpInst::FCMP_UGT: {
      MIRBuilder.buildBoolExt(
          Result,
          MIRBuilder.buildNot(
              i1,
              MIRBuilder.buildFCmp(CmpInst::FCMP_OLE, i1, LHS, RHS).getReg(0)),
          false);
      break;
    }
    case CmpInst::FCMP_UGE: {
      MIRBuilder.buildBoolExt(
          Result,
          MIRBuilder.buildNot(
              i1,
              MIRBuilder.buildFCmp(CmpInst::FCMP_OLT, i1, LHS, RHS).getReg(0)),
          false);
      break;
    }
    case CmpInst::FCMP_ULT: {
      MIRBuilder.buildBoolExt(
          Result,
          MIRBuilder.buildNot(
              i1,
              MIRBuilder.buildFCmp(CmpInst::FCMP_OGE, i1, LHS, RHS).getReg(0)),
          false);
      break;
    }
    case CmpInst::FCMP_ULE: {
      MIRBuilder.buildBoolExt(
          Result,
          MIRBuilder.buildNot(
              i1,
              MIRBuilder.buildFCmp(CmpInst::FCMP_OGT, i1, LHS, RHS).getReg(0)),
          false);
      break;
    }
    case CmpInst::FCMP_UNE:
      return true;
    case CmpInst::FCMP_TRUE:
      MIRBuilder.buildBoolExt(Result, MIRBuilder.buildConstant(i1, 1), false);
      break;
    default:
      llvm_unreachable("Unknown FCMP predicate");
    }

    MI.eraseFromParent();

    return true;
  }
  case TargetOpcode::G_FMINNUM: {
    // We can only use G_FMINIMUM if we can be sure no NaN is present.
    // This is because G_FMINIMUM propogates NaN, while G_FMINNUM says that
    // the non-NaN operand must result.
    if (!MI.getFlag(MachineInstr::MIFlag::FmNoNans))
      return Helper.libcall(MI, LocObserver) !=
             LegalizerHelper::UnableToLegalize;

    // this respects minnum signed zero handling. G_FMINNUM has undefined zeros
    // handling, so G_FMINIMUM's specific choice of zero is irrelavent.
    MIRBuilder.buildInstr(TargetOpcode::G_FMINIMUM)
        .addDef(MI.getOperand(0).getReg())
        .addUse(MI.getOperand(1).getReg())
        .addUse(MI.getOperand(2).getReg())
        .setMIFlags(MI.getFlags());

    MI.eraseFromParent();
    return true;
  }
  case TargetOpcode::G_FMAXNUM: {
    // We can only use G_FMAXIMUM if we can be sure no NaN is present.
    // This is because G_FMAXIMUM propogates NaN, while G_MAXNUM says that
    // the non-NaN operand must result.
    if (!MI.getFlag(MachineInstr::MIFlag::FmNoNans))
      return Helper.libcall(MI, LocObserver) !=
             LegalizerHelper::UnableToLegalize;

    // this respects maxnum signed zero handling. G_FMAXNUM has undefined zeros
    // handling, so G_FMAXIMUM's specific choice of zero is irrelavent.
    MIRBuilder.buildInstr(TargetOpcode::G_FMAXIMUM)
        .addDef(MI.getOperand(0).getReg())
        .addUse(MI.getOperand(1).getReg())
        .addUse(MI.getOperand(2).getReg())
        .setMIFlags(MI.getFlags());

    MI.eraseFromParent();
    return true;
  }
  case WebAssembly::G_FMINIMUMNUM: {
    // This is a stripped down version of fminimumnum handling for SelectionDAG
    // TargetLowering
    Register Result = MI.getOperand(0).getReg();
    LLT ResultTy = MRI.getType(Result);

    if (MI.getFlag(MachineInstr::MIFlag::FmNoNans)) {
      MIRBuilder.buildInstr(TargetOpcode::G_FMINIMUM)
          .addDef(MI.getOperand(0).getReg())
          .addUse(MI.getOperand(1).getReg())
          .addUse(MI.getOperand(2).getReg())
          .setMIFlags(MI.getFlags());
    } else {
      Register LHS = MI.getOperand(1).getReg();
      Register RHS = MI.getOperand(2).getReg();

      LHS = MIRBuilder
                .buildSelect(ResultTy,
                             MIRBuilder.buildFCmp(CmpInst::Predicate::FCMP_UNO,
                                                  i1, LHS, LHS),
                             RHS, LHS)
                ->getOperand(0)
                .getReg();
      RHS = MIRBuilder
                .buildSelect(ResultTy,
                             MIRBuilder.buildFCmp(CmpInst::Predicate::FCMP_UNO,
                                                  i1, RHS, RHS),
                             LHS, RHS)
                ->getOperand(0)
                .getReg();

      if (MI.getFlag(MachineInstr::MIFlag::FmNsz)) {
        MIRBuilder.buildSelect(
            Result,
            MIRBuilder.buildFCmp(CmpInst::Predicate::FCMP_OLT, i1, LHS, RHS),
            LHS, RHS);
      } else {
        Register MinMax =
            MIRBuilder
                .buildSelect(ResultTy,
                             MIRBuilder.buildFCmp(CmpInst::Predicate::FCMP_OLT,
                                                  i1, LHS, RHS),
                             LHS, RHS)
                ->getOperand(0)
                .getReg();

        Register IsZero =
            MIRBuilder
                .buildFCmp(CmpInst::Predicate::FCMP_OEQ, i1, MinMax,
                           MIRBuilder.buildFConstant(ResultTy, 0.0))
                ->getOperand(0)
                .getReg();

        Register RetZero =
            MIRBuilder
                .buildSelect(ResultTy,
                             MIRBuilder.buildIsFPClass(i1, LHS, fcNegZero), LHS,
                             MinMax)
                .setMIFlags(MI.getFlags())
                ->getOperand(0)
                .getReg();
        MIRBuilder.buildSelect(Result, IsZero, RetZero, MinMax)
            .setMIFlags(MI.getFlags());
      }
    }

    MI.eraseFromParent();
    return true;
  }
  case WebAssembly::G_FMAXIMUMNUM: {
    // This is a stripped down version of fmaximumnum handling for SelectionDAG
    // TargetLowering
    Register Result = MI.getOperand(0).getReg();
    LLT ResultTy = MRI.getType(Result);

    if (MI.getFlag(MachineInstr::MIFlag::FmNoNans)) {
      MIRBuilder.buildInstr(TargetOpcode::G_FMINIMUM)
          .addDef(MI.getOperand(0).getReg())
          .addUse(MI.getOperand(1).getReg())
          .addUse(MI.getOperand(2).getReg())
          .setMIFlags(MI.getFlags());
    } else {
      Register LHS = MI.getOperand(1).getReg();
      Register RHS = MI.getOperand(2).getReg();

      LHS = MIRBuilder
                .buildSelect(ResultTy,
                             MIRBuilder.buildFCmp(CmpInst::Predicate::FCMP_UNO,
                                                  i1, LHS, LHS),
                             RHS, LHS)
                ->getOperand(0)
                .getReg();
      RHS = MIRBuilder
                .buildSelect(ResultTy,
                             MIRBuilder.buildFCmp(CmpInst::Predicate::FCMP_UNO,
                                                  i1, RHS, RHS),
                             LHS, RHS)
                ->getOperand(0)
                .getReg();

      if (MI.getFlag(MachineInstr::MIFlag::FmNsz)) {
        MIRBuilder.buildSelect(
            Result,
            MIRBuilder.buildFCmp(CmpInst::Predicate::FCMP_OGT, i1, LHS, RHS),
            LHS, RHS);
      } else {
        Register MinMax =
            MIRBuilder
                .buildSelect(ResultTy,
                             MIRBuilder.buildFCmp(CmpInst::Predicate::FCMP_OGT,
                                                  i1, LHS, RHS),
                             LHS, RHS)
                ->getOperand(0)
                .getReg();

        Register IsZero =
            MIRBuilder
                .buildFCmp(CmpInst::Predicate::FCMP_OEQ, i1, MinMax,
                           MIRBuilder.buildFConstant(ResultTy, 0.0))
                ->getOperand(0)
                .getReg();

        Register RetZero =
            MIRBuilder
                .buildSelect(ResultTy,
                             MIRBuilder.buildIsFPClass(i1, LHS, fcPosZero), LHS,
                             MinMax)
                .setMIFlags(MI.getFlags())
                ->getOperand(0)
                .getReg();
        MIRBuilder.buildSelect(Result, IsZero, RetZero, MinMax)
            .setMIFlags(MI.getFlags());
      }
    }

    MI.eraseFromParent();
    return true;
  }
  default:
    break;
  }
  return false;
}

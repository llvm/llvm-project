//===- WebAssemblyLegalizerInfo.h --------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the targeting of the Machinelegalizer class for
/// WebAssembly
//===----------------------------------------------------------------------===//

#include "WebAssemblyLegalizerInfo.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssemblySubtarget.h"
#include "llvm/CodeGen/GlobalISel/LegalizerHelper.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/IR/DerivedTypes.h"

#define DEBUG_TYPE "wasm-legalinfo"

using namespace llvm;
using namespace LegalizeActions;

WebAssemblyLegalizerInfo::WebAssemblyLegalizerInfo(
    const WebAssemblySubtarget &ST) {
  using namespace TargetOpcode;
  const LLT s8 = LLT::scalar(8);
  const LLT s16 = LLT::scalar(16);
  const LLT s32 = LLT::scalar(32);
  const LLT s64 = LLT::scalar(64);

  const LLT p0 = LLT::pointer(0, ST.hasAddr64() ? 64 : 32);
  const LLT p0s = LLT::scalar(ST.hasAddr64() ? 64 : 32);

  getActionDefinitionsBuilder(G_GLOBAL_VALUE).legalFor({p0});

  getActionDefinitionsBuilder(G_PHI)
      .legalFor({p0, s32, s64})
      .widenScalarToNextPow2(0)
      .clampScalar(0, s32, s64);
  getActionDefinitionsBuilder(G_BR).alwaysLegal();
  getActionDefinitionsBuilder(G_BRCOND).legalFor({s32}).clampScalar(0, s32,
                                                                    s32);
  getActionDefinitionsBuilder(G_BRJT)
      .legalFor({{p0, s32}})
      .clampScalar(1, s32, s32);

  getActionDefinitionsBuilder(G_SELECT)
      .legalFor({{s32, s32}, {s64, s32}, {p0, s32}})
      .widenScalarToNextPow2(0)
      .clampScalar(0, s32, s64)
      .clampScalar(1, s32, s32);

  getActionDefinitionsBuilder(G_JUMP_TABLE).legalFor({p0});

  getActionDefinitionsBuilder(G_ICMP)
      .legalFor({{s32, s32}, {s32, s64}, {s32, p0}})
      .widenScalarToNextPow2(1)
      .clampScalar(1, s32, s64)
      .clampScalar(0, s32, s32);

  getActionDefinitionsBuilder(G_FCMP)
      .legalFor({{s32, s32}, {s32, s64}})
      .clampScalar(0, s32, s32)
      .libcall();

  getActionDefinitionsBuilder(G_FRAME_INDEX).legalFor({p0});

  getActionDefinitionsBuilder(G_CONSTANT)
      .legalFor({s32, s64, p0})
      .widenScalarToNextPow2(0)
      .clampScalar(0, s32, s64);

  getActionDefinitionsBuilder(G_FCONSTANT)
      .legalFor({s32, s64})
      .clampScalar(0, s32, s64);

  getActionDefinitionsBuilder(G_IMPLICIT_DEF)
      .legalFor({s32, s64, p0})
      .widenScalarToNextPow2(0)
      .clampScalar(0, s32, s64);

  getActionDefinitionsBuilder(
      {G_ADD, G_SUB, G_MUL, G_UDIV, G_SDIV, G_UREM, G_SREM})
      .legalFor({s32, s64})
      .widenScalarToNextPow2(0)
      .clampScalar(0, s32, s64);

  getActionDefinitionsBuilder({G_ASHR, G_LSHR, G_SHL, G_CTLZ, G_CTLZ_ZERO_UNDEF,
                               G_CTTZ, G_CTTZ_ZERO_UNDEF, G_CTPOP})
      .legalFor({{s32, s32}, {s64, s64}})
      .widenScalarToNextPow2(0)
      .clampScalar(0, s32, s64)
      .minScalarSameAs(1, 0)
      .maxScalarSameAs(1, 0);

  getActionDefinitionsBuilder({G_FSHL, G_FSHR})
      .legalFor({{s32, s32}, {s64, s64}})
      .lower();

  getActionDefinitionsBuilder({G_SCMP, G_UCMP}).lower();

  getActionDefinitionsBuilder({G_AND, G_OR, G_XOR})
      .legalFor({s32, s64})
      .widenScalarToNextPow2(0)
      .clampScalar(0, s32, s64);

  getActionDefinitionsBuilder({G_UMIN, G_UMAX, G_SMIN, G_SMAX}).lower();

  getActionDefinitionsBuilder({G_FADD, G_FSUB, G_FDIV, G_FMUL, G_FNEG, G_FABS,
                               G_FCEIL, G_FFLOOR, G_FSQRT, G_INTRINSIC_TRUNC,
                               G_FNEARBYINT, G_FRINT, G_INTRINSIC_ROUNDEVEN,
                               G_FMINIMUM, G_FMAXIMUM})
      .legalFor({s32, s64})
      .minScalar(0, s32);

  // TODO: _IEEE not lowering correctly?
  getActionDefinitionsBuilder(
      {G_FMINNUM, G_FMAXNUM, G_FMINNUM_IEEE, G_FMAXNUM_IEEE})
      .lowerFor({s32, s64})
      .minScalar(0, s32);

  getActionDefinitionsBuilder({G_FMA, G_FREM})
      .libcallFor({s32, s64})
      .minScalar(0, s32);

  getActionDefinitionsBuilder(G_LROUND).libcallForCartesianProduct({s32},
                                                                   {s32, s64});

  getActionDefinitionsBuilder(G_LLROUND).libcallForCartesianProduct({s64},
                                                                    {s32, s64});

  getActionDefinitionsBuilder(G_FCOPYSIGN)
      .legalFor({s32, s64})
      .minScalar(0, s32)
      .minScalarSameAs(1, 0)
      .maxScalarSameAs(1, 0);

  getActionDefinitionsBuilder({G_FPTOUI, G_FPTOUI_SAT, G_FPTOSI, G_FPTOSI_SAT})
      .legalForCartesianProduct({s32, s64}, {s32, s64})
      .minScalar(1, s32)
      .widenScalarToNextPow2(0)
      .clampScalar(0, s32, s64);

  getActionDefinitionsBuilder({G_UITOFP, G_SITOFP})
      .legalForCartesianProduct({s32, s64}, {s32, s64})
      .minScalar(1, s32)
      .widenScalarToNextPow2(1)
      .clampScalar(1, s32, s64);

  getActionDefinitionsBuilder(G_PTRTOINT).legalFor({{p0s, p0}});
  getActionDefinitionsBuilder(G_INTTOPTR).legalFor({{p0, p0s}});
  getActionDefinitionsBuilder(G_PTR_ADD).legalFor({{p0, p0s}});

  getActionDefinitionsBuilder(G_LOAD)
      .legalForTypesWithMemDesc(
          {{s32, p0, s32, 1}, {s64, p0, s64, 1}, {p0, p0, p0, 1}})
      .legalForTypesWithMemDesc({{s32, p0, s8, 1},
                                 {s32, p0, s16, 1},

                                 {s64, p0, s8, 1},
                                 {s64, p0, s16, 1},
                                 {s64, p0, s32, 1}})
      .clampScalar(0, s32, s64)
      .lowerIfMemSizeNotByteSizePow2();

  getActionDefinitionsBuilder(G_STORE)
      .legalForTypesWithMemDesc(
          {{s32, p0, s32, 1}, {s64, p0, s64, 1}, {p0, p0, p0, 1}})
      .legalForTypesWithMemDesc({{s32, p0, s8, 1},
                                 {s32, p0, s16, 1},

                                 {s64, p0, s8, 1},
                                 {s64, p0, s16, 1},
                                 {s64, p0, s32, 1}})
      .clampScalar(0, s32, s64)
      .lowerIfMemSizeNotByteSizePow2();

  getActionDefinitionsBuilder({G_ZEXTLOAD, G_SEXTLOAD})
      .legalForTypesWithMemDesc({{s32, p0, s8, 1},
                                 {s32, p0, s16, 1},

                                 {s64, p0, s8, 1},
                                 {s64, p0, s16, 1},
                                 {s64, p0, s32, 1}})
      .clampScalar(0, s32, s64)
      .lowerIfMemSizeNotByteSizePow2();

  if (ST.hasBulkMemoryOpt()) {
    getActionDefinitionsBuilder(G_BZERO).unsupported();

    getActionDefinitionsBuilder(G_MEMSET)
        .legalForCartesianProduct({p0}, {s32}, {p0s})
        .customForCartesianProduct({p0}, {s8}, {p0s})
        .immIdx(0);

    getActionDefinitionsBuilder({G_MEMCPY, G_MEMMOVE})
        .legalForCartesianProduct({p0}, {p0}, {p0s})
        .immIdx(0);

    getActionDefinitionsBuilder(G_MEMCPY_INLINE)
        .legalForCartesianProduct({p0}, {p0}, {p0s});
  } else {
    getActionDefinitionsBuilder({G_BZERO, G_MEMCPY, G_MEMMOVE, G_MEMSET})
        .libcall();
  }

  // TODO: figure out how to combine G_ANYEXT of G_ASSERT_{S|Z}EXT (or
  // appropriate G_AND and G_SEXT_IN_REG?) to a G_{S|Z}EXT + G_ASSERT_{S|Z}EXT
  // for better optimization (since G_ANYEXT will lower to a ZEXT or SEXT
  // instruction anyway).

  getActionDefinitionsBuilder(G_ANYEXT)
      .legalFor({{s64, s32}})
      .clampScalar(0, s32, s64)
      .clampScalar(1, s32, s64);

  getActionDefinitionsBuilder({G_SEXT, G_ZEXT})
      .legalFor({{s64, s32}})
      .clampScalar(0, s32, s64)
      .clampScalar(1, s32, s64)
      .lower();

  if (ST.hasSignExt()) {
    getActionDefinitionsBuilder(G_SEXT_INREG)
        .clampScalar(0, s32, s64)
        .customFor({s32, s64});
  } else {
    getActionDefinitionsBuilder(G_SEXT_INREG).lower();
  }

  getActionDefinitionsBuilder(G_TRUNC)
      .legalFor({{s32, s64}})
      .clampScalar(0, s32, s64)
      .clampScalar(1, s32, s64)
      .lower();

  getActionDefinitionsBuilder(G_FPEXT).legalFor({{s64, s32}});

  getActionDefinitionsBuilder(G_FPTRUNC).legalFor({{s32, s64}});

  getActionDefinitionsBuilder(G_VASTART).legalFor({p0});
  getActionDefinitionsBuilder(G_VAARG)
      .legalForCartesianProduct({s32, s64}, {p0})
      .clampScalar(0, s32, s64);

  getActionDefinitionsBuilder(G_DYN_STACKALLOC).lowerFor({{p0, p0s}});

  getActionDefinitionsBuilder({G_STACKSAVE, G_STACKRESTORE}).lower();

  getLegacyLegalizerInfo().computeTables();
}

bool WebAssemblyLegalizerInfo::legalizeCustom(
    LegalizerHelper &Helper, MachineInstr &MI,
    LostDebugLocObserver &LocObserver) const {
  auto &MRI = *Helper.MIRBuilder.getMRI();
  auto &MIRBuilder = Helper.MIRBuilder;

  switch (MI.getOpcode()) {
  case TargetOpcode::G_SEXT_INREG: {
    assert(MI.getOperand(2).isImm() && "Expected immediate");

    // Mark only 8/16/32-bit SEXT_INREG as legal
    auto [DstReg, SrcReg] = MI.getFirst2Regs();
    auto DstType = MRI.getType(DstReg);
    auto ExtFromWidth = MI.getOperand(2).getImm();

    if (ExtFromWidth == 8 || ExtFromWidth == 16 ||
        (DstType.getScalarSizeInBits() == 64 && ExtFromWidth == 32)) {
      return true;
    }

    Register TmpRes = MRI.createGenericVirtualRegister(DstType);

    auto MIBSz = MIRBuilder.buildConstant(
        DstType, DstType.getScalarSizeInBits() - ExtFromWidth);
    MIRBuilder.buildShl(TmpRes, SrcReg, MIBSz->getOperand(0));
    MIRBuilder.buildAShr(DstReg, TmpRes, MIBSz->getOperand(0));
    MI.eraseFromParent();

    return true;
  }
  case TargetOpcode::G_MEMSET: {
    // Anyext the value being set to 32 bit (only the bottom 8 bits are read by
    // the instruction).
    Helper.Observer.changingInstr(MI);
    auto &Value = MI.getOperand(1);

    Register ExtValueReg =
        Helper.MIRBuilder.buildAnyExt(LLT::scalar(32), Value).getReg(0);
    Value.setReg(ExtValueReg);
    Helper.Observer.changedInstr(MI);
    return true;
  }
  default:
    break;
  }
  return false;
}

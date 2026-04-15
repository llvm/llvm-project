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

  const LLT s32 = LLT::scalar(32);
  const LLT s64 = LLT::scalar(64);

  getActionDefinitionsBuilder({G_CONSTANT, G_IMPLICIT_DEF, G_ADD, G_SUB, G_MUL,
                               G_UDIV, G_SDIV, G_UREM, G_SREM, G_AND, G_OR,
                               G_XOR})
      .legalFor({s32, s64})
      .widenScalarToNextPow2(0)
      .clampScalar(0, s32, s64);

  getActionDefinitionsBuilder({G_ASHR, G_LSHR, G_SHL})
      .legalFor({{s32, s32}, {s64, s64}})
      .widenScalarToNextPow2(0)
      .clampScalar(0, s32, s64)
      .scalarSameSizeAs(1, 0);

  getActionDefinitionsBuilder({G_CTLZ, G_CTTZ, G_CTPOP})
      .legalFor({{s32, s32}, {s64, s64}})
      .widenScalarToNextPow2(1)
      .clampScalar(1, s32, s64)
      .scalarSameSizeAs(0, 1);

  getActionDefinitionsBuilder({G_CTLZ_ZERO_UNDEF, G_CTTZ_ZERO_UNDEF}).lower();

  getActionDefinitionsBuilder({G_ROTL, G_ROTR})
      .legalFor({{s32, s32}, {s64, s64}})
      .scalarSameSizeAs(1, 0)
      .lower();

  getActionDefinitionsBuilder({G_FSHL, G_FSHR}).lower();

  getActionDefinitionsBuilder({G_ANYEXT, G_SEXT, G_ZEXT})
      .legalFor({{s64, s32}})
      .clampScalar(0, s64, s64)
      .clampScalar(1, s32, s32);

  getActionDefinitionsBuilder(G_TRUNC)
      .legalFor({{s32, s64}})
      .clampScalar(0, s32, s32)
      .clampScalar(1, s64, s64);

  getActionDefinitionsBuilder(G_SEXT_INREG)
      .customFor(ST.hasSignExt(), {s32, s64})
      .clampScalar(0, s32, s64)
      .lower();

  getLegacyLegalizerInfo().computeTables();
}

bool WebAssemblyLegalizerInfo::legalizeCustom(
    LegalizerHelper &Helper, MachineInstr &MI,
    LostDebugLocObserver &LocObserver) const {
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
  default:
    break;
  }
  return false;
}

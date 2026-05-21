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
using namespace LegalityPredicates;

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

  getActionDefinitionsBuilder({G_ANYEXT, G_SEXT, G_ZEXT})
      .legalFor({{i64, i32}})
      .clampScalar(0, s64, s64)
      .clampScalar(1, s32, s32);

  getActionDefinitionsBuilder(G_TRUNC)
      .legalFor({{i32, i64}})
      .clampScalar(0, s32, s32)
      .clampScalar(1, s64, s64);

  {
    LegalizeRuleSet &Builder = getActionDefinitionsBuilder(G_SEXT_INREG);

    if (ST.hasSignExt())
      Builder.legalIf(
          all(typeInSet(0, {i32, i64}),
              LegalityPredicates::any(immInSet(0, {8, 16}),
                                      all(typeIs(0, i64), immIs(0, 32)))));

    Builder.clampScalar(0, s32, s64).lower();
  }

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

  getActionDefinitionsBuilder(G_FPEXT)
      .legalFor({{f64, f32}})
      .clampScalar(0, s64, s64)
      .clampScalar(1, s32, s32);

  getActionDefinitionsBuilder(G_FPTRUNC)
      .legalFor({{f32, f64}})
      .clampScalar(0, s32, s32)
      .clampScalar(1, s64, s64);

  getLegacyLegalizerInfo().computeTables();
}

bool WebAssemblyLegalizerInfo::legalizeCustom(
    LegalizerHelper &Helper, MachineInstr &MI,
    LostDebugLocObserver &LocObserver) const {
  switch (MI.getOpcode()) {
  default:
    break;
  }
  return false;
}

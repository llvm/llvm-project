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
#include "llvm/CodeGen/GlobalISel/LegalizerHelper.h"
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"

using namespace llvm;

M88kLegalizerInfo::M88kLegalizerInfo(const M88kSubtarget &ST) {
  using namespace TargetOpcode;
  const LLT S1 = LLT::scalar(1);
  const LLT S8 = LLT::scalar(8);
  const LLT S16 = LLT::scalar(16);
  const LLT S32 = LLT::scalar(32);
  const LLT S64 = LLT::scalar(64);
  const LLT S80 = LLT::scalar(80);
  const LLT P0 = LLT::pointer(0, 32);
  getActionDefinitionsBuilder(G_PHI).legalFor({S32});
  getActionDefinitionsBuilder({G_IMPLICIT_DEF, G_FREEZE}).legalFor({S32});
  getActionDefinitionsBuilder(G_CONSTANT)
      .legalFor({S32, P0})
      .clampScalar(0, S32, S32);
  getActionDefinitionsBuilder(G_INTTOPTR)
      .legalFor({{P0, S32}})
      .minScalar(1, S32);
  getActionDefinitionsBuilder(G_PTRTOINT)
      .legalFor({{S32, P0}})
      .minScalar(0, S32);
  getActionDefinitionsBuilder({G_SEXTLOAD, G_ZEXTLOAD})
      .legalForTypesWithMemDesc({{S32, P0, S8, 8}, {S32, P0, S16, 16}});
  getActionDefinitionsBuilder(G_LOAD).legalForTypesWithMemDesc(
      {{S32, P0, S32, 32}, {S64, P0, S64, 64}});
  getActionDefinitionsBuilder(G_STORE).legalForTypesWithMemDesc(
      {{S8, P0, S8, 8},    // Truncating store.
       {S16, P0, S16, 16}, // Truncating store.
       {S32, P0, S32, 32},
       {S64, P0, S64, 64}});
  getActionDefinitionsBuilder(G_PTR_ADD)
      .legalFor({{P0, S32}})
      .clampScalar(1, S32, S32);
  getActionDefinitionsBuilder(G_ADD).legalFor({S32});
  getActionDefinitionsBuilder(G_SUB).legalFor({S32});
  getActionDefinitionsBuilder(G_MUL).legalFor({S32});
  getActionDefinitionsBuilder(G_UDIV).legalFor({S32});
  getActionDefinitionsBuilder({G_AND, G_OR, G_XOR})
      .legalFor({S32})
      .clampScalar(0, S32, S32);
  getActionDefinitionsBuilder({G_SBFX, G_UBFX})
      .legalFor({{S32, S32}})
      .clampScalar(2, S32, S32)
      .clampScalar(1, S32, S32)
      .clampScalar(0, S32, S32);
  getActionDefinitionsBuilder({G_SHL, G_LSHR, G_ASHR})
      .legalFor({{S32, S32}})
      .clampScalar(0, S32, S32)
      .clampScalar(1, S32, S32);

  getActionDefinitionsBuilder(G_ICMP)
      .legalForCartesianProduct({S1}, {S32, P0})
      .minScalar(1, S32);
  getActionDefinitionsBuilder(G_BRCOND).legalFor({S1});

  getActionDefinitionsBuilder(G_FRAME_INDEX).legalFor({P0});
  getActionDefinitionsBuilder(G_GLOBAL_VALUE).legalFor({P0});

  getActionDefinitionsBuilder({G_FADD, G_FSUB, G_FMUL, G_FDIV, G_FNEG})
      .legalFor({S32, S64, S80});
  getLegacyLegalizerInfo().computeTables();
}

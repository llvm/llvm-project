//===-- M68kLegalizerInfo.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the targeting of the Machinelegalizer class for M68k.
//===----------------------------------------------------------------------===//

#include "M68kLegalizerInfo.h"
#include "llvm/CodeGen/GlobalISel/LegalizerHelper.h"
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"

using namespace llvm;

M68kLegalizerInfo::M68kLegalizerInfo(const M68kSubtarget &ST) {
  using namespace TargetOpcode;
  const LLT S32 = LLT::scalar(32);
  const LLT P0 = LLT::pointer(0, 32);
  getActionDefinitionsBuilder(G_LOAD).legalFor({S32});
  getActionDefinitionsBuilder(G_FRAME_INDEX).legalFor({P0});
  getActionDefinitionsBuilder(G_ADD).legalFor({S32});
  getActionDefinitionsBuilder(G_SUB).legalFor({S32});
  getActionDefinitionsBuilder(G_MUL).legalFor({S32});
  getActionDefinitionsBuilder(G_UDIV).legalFor({S32});
  getLegacyLegalizerInfo().computeTables();
}

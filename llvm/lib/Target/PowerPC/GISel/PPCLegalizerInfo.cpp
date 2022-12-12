//===- PPCLegalizerInfo.h ----------------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the targeting of the Machinelegalizer class for PowerPC
//===----------------------------------------------------------------------===//

#include "PPCLegalizerInfo.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ppc-legalinfo"

using namespace llvm;
using namespace LegalizeActions;

PPCLegalizerInfo::PPCLegalizerInfo(const PPCSubtarget &ST) {
  using namespace TargetOpcode;
  const LLT P0 = LLT::pointer(0, 64);
  const LLT S8 = LLT::scalar(8);
  const LLT S16 = LLT::scalar(16);
  const LLT S32 = LLT::scalar(32);
  const LLT S64 = LLT::scalar(64);
  getActionDefinitionsBuilder(G_IMPLICIT_DEF).legalFor({S64});
  getActionDefinitionsBuilder(G_CONSTANT)
      .legalFor({S32, S64})
      .clampScalar(0, S64, S64);
  getActionDefinitionsBuilder({G_ZEXT, G_SEXT})
      .legalForCartesianProduct({S64}, {S8, S16, S32})
      .clampScalar(0, S64, S64);
  getActionDefinitionsBuilder({G_AND, G_OR, G_XOR})
      .legalFor({S64})
      .clampScalar(0, S64, S64);
  getActionDefinitionsBuilder({G_ADD, G_SUB})
      .legalFor({S64})
      .clampScalar(0, S64, S64);

  getActionDefinitionsBuilder({G_FADD, G_FSUB, G_FMUL, G_FDIV})
      .legalFor({S32, S64});

  getActionDefinitionsBuilder({G_FPTOSI, G_FPTOUI})
      .legalForCartesianProduct({S64}, {S32, S64});

  getActionDefinitionsBuilder({G_SITOFP, G_UITOFP})
      .legalForCartesianProduct({S32, S64}, {S64});

  getActionDefinitionsBuilder({G_LOAD, G_STORE})
      .legalForTypesWithMemDesc({{S64, P0, S64, 8}, {S32, P0, S32, 4}});

  getLegacyLegalizerInfo().computeTables();
}

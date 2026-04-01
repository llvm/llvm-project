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
#include "WebAssemblySubtarget.h"

#define DEBUG_TYPE "wasm-legalinfo"

using namespace llvm;
using namespace LegalizeActions;

WebAssemblyLegalizerInfo::WebAssemblyLegalizerInfo(
    const WebAssemblySubtarget &ST) {
  using namespace TargetOpcode;

  const LLT s32 = LLT::scalar(32);
  const LLT s64 = LLT::scalar(64);

  getActionDefinitionsBuilder({G_CONSTANT, G_ADD, G_AND})
      .legalFor({s32, s64})
      .widenScalarToNextPow2(0)
      .clampScalar(0, s32, s64);

  getActionDefinitionsBuilder({G_ASHR, G_SHL})
      .legalFor({{s32, s32}, {s64, s64}})
      .widenScalarToNextPow2(0)
      .clampScalar(0, s32, s64)
      .scalarSameSizeAs(1, 0);

  getActionDefinitionsBuilder(G_SEXT_INREG).lower();

  getLegacyLegalizerInfo().computeTables();
}

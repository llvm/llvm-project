//===- AMDGPUExpandPseudoIntrinsics.cpp - Pseudo Intrinsic Expander Pass --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file implements a pass that deals with expanding AMDGCN generic pseudo-
// intrinsics into target specific quantities / sequences. In this context, a
// pseudo-intrinsic is an AMDGCN intrinsic that does not directly map to a
// specific instruction, but rather is intended as a mechanism for abstractly
// conveying target specific info to a HLL / the FE, without concretely
// impacting the AST. An example of such an intrinsic is amdgcn.wavefrontsize.
// This pass should run as early as possible / immediately after Clang CodeGen,
// so that the optimisation pipeline and the BE operate with concrete target
// data.
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUTargetMachine.h"
#include "GCNSubtarget.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

using namespace llvm;

static inline PreservedAnalyses expandWaveSizeIntrinsic(const GCNSubtarget &ST,
                                                        Function *WaveSize) {
  if (WaveSize->hasZeroLiveUses())
    return PreservedAnalyses::all();

  for (auto &&U : WaveSize->users())
    U->replaceAllUsesWith(
        ConstantInt::get(WaveSize->getReturnType(), ST.getWavefrontSize()));

  return PreservedAnalyses::none();
}

PreservedAnalyses
AMDGPUExpandPseudoIntrinsicsPass::run(Module &M, ModuleAnalysisManager &) {
  if (M.empty())
    return PreservedAnalyses::all();

  const auto &ST = TM.getSubtarget<GCNSubtarget>(*M.begin());

  // This is not a concrete target, we should not fold early.
  if (ST.getCPU().empty() || ST.getCPU() == "generic")
    return PreservedAnalyses::all();

  if (auto WS = Intrinsic::getDeclarationIfExists(
          &M, Intrinsic::amdgcn_wavefrontsize))
    return expandWaveSizeIntrinsic(ST, WS);

  return PreservedAnalyses::all();
}

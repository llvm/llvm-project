//===- VPlanTestPass.cpp - Test VPlan transforms -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass is a lightweight testing harness for VPlan transforms. It builds
// VPlan0 for loops and runs specified transforms.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/VPlanTestPass.h"
#include "VPlan.h"
#include "VPlanTransforms.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/PassManager.h"

using namespace llvm;

PreservedAnalyses VPlanTestPass::run(Function &F, FunctionAnalysisManager &AM) {
  if (TransformPipeline.empty())
    return PreservedAnalyses::all();

  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);

  for (Loop *L : LI) {
    PredicatedScalarEvolution PSE(SE, *L);

    auto Plan = VPlanTransforms::buildVPlan0(
        L, LI, Type::getInt64Ty(F.getContext()), DebugLoc(), PSE);

    VPlanTransforms::runTestTransforms(*Plan, TransformPipeline, &TLI);
    outs() << *Plan << "\n";
  }

  return PreservedAnalyses::all();
}

void VPlanTestPass::printPipeline(
    raw_ostream &OS, function_ref<StringRef(StringRef)> MapClassName2PassName) {
  static_cast<PassInfoMixin<VPlanTestPass> *>(this)->printPipeline(
      OS, MapClassName2PassName);
  OS << '<';
  OS << TransformPipeline;
  OS << '>';
}

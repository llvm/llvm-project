//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/PGOForceFunctionAttrs.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/IR/PassManager.h"

using namespace llvm;

PreservedAnalyses PGOForceFunctionAttrsPass::run(Module &M,
                                                 ModuleAnalysisManager &AM) {
  if (ColdType == PGOOptions::ColdFuncOpt::Default)
    return PreservedAnalyses::all();
  ProfileSummaryInfo &PSI = AM.getResult<ProfileSummaryAnalysis>(M);
  if (!PSI.hasProfileSummary())
    return PreservedAnalyses::all();
  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  bool MadeChange = false;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    BlockFrequencyInfo &BFI = FAM.getResult<BlockFrequencyAnalysis>(F);
    if (!PSI.isFunctionColdInCallGraph(&F, BFI))
      continue;
    // Add optsize/minsize/optnone if requested.
    switch (ColdType) {
    case PGOOptions::ColdFuncOpt::Default:
      assert(false);
      break;
    case PGOOptions::ColdFuncOpt::OptSize:
      if (!F.hasFnAttribute(Attribute::OptimizeNone) &&
          !F.hasFnAttribute(Attribute::OptimizeForSize) &&
          !F.hasFnAttribute(Attribute::MinSize)) {
        F.addFnAttr(Attribute::OptimizeForSize);
        MadeChange = true;
      }
      break;
    case PGOOptions::ColdFuncOpt::MinSize:
      // Change optsize to minsize.
      if (!F.hasFnAttribute(Attribute::OptimizeNone) &&
          !F.hasFnAttribute(Attribute::MinSize)) {
        F.removeFnAttr(Attribute::OptimizeForSize);
        F.addFnAttr(Attribute::MinSize);
        MadeChange = true;
      }
      break;
    case PGOOptions::ColdFuncOpt::OptNone:
      // Strip optsize/minsize.
      F.removeFnAttr(Attribute::OptimizeForSize);
      F.removeFnAttr(Attribute::MinSize);
      F.addFnAttr(Attribute::OptimizeNone);
      F.addFnAttr(Attribute::NoInline);
      MadeChange = true;
      break;
    }
  }
  return MadeChange ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

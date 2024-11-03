//===- StripNonLineTableDebugInfo.cpp -- Strip parts of Debug Info --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/StripNonLineTableDebugInfo.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils.h"

using namespace llvm;

PreservedAnalyses
StripNonLineTableDebugInfoPass::run(Module &M, ModuleAnalysisManager &AM) {
  llvm::stripNonLineTableDebugInfo(M);
  return PreservedAnalyses::all();
}

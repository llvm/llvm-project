//===- NextSiliconImportRecursion.h - NS recursive-import inference pass ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//
//
// This file provides the prototypes and definitions related to the pass.
//
// The purpose of this pass is to set the `ns-import-recursion` function
// attribute, if either any loop or the function itself has an ns location
// attached to it.
//
//===--------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_NEXTSILICONIMPORTRECURSION_H
#define LLVM_TRANSFORMS_UTILS_NEXTSILICONIMPORTRECURSION_H

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class Function;
class FunctionPass;

class NextSiliconImportRecursionPass
    : public PassInfoMixin<NextSiliconImportRecursionPass> {
public:
  NextSiliconImportRecursionPass() {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_NEXTSILICONIMPORTRECURSION_H

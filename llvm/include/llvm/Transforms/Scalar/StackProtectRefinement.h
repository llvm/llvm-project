//===- StackProtectRefinement.h - Stack Protect Refinement ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_STACK_PROTECT_REFINEMENT_H
#define LLVM_TRANSFORMS_SCALAR_STACK_PROTECT_REFINEMENT_H

#include "llvm/Analysis/StackSafetyAnalysis.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class BasicBlock;
class Function;
class Instruction;

class StackProtectRefinementPass
    : public PassInfoMixin<StackProtectRefinementPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);

private:
  void processFunction(Function &F) const;

  const StackSafetyGlobalInfo *SSI;
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_STACK_PROTECT_REFINEMENT_H

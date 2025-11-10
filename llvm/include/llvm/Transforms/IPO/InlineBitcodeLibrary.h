//===----- InlineBitcodeLibrary.h - Link bitcode to the module-------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass links and inlines vector functions from a bitcode module
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_LBC_H
#define LLVM_TRANSFORMS_IPO_LBC_H

#include "llvm/IR/Analysis.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class Module;

class InlineBitcodeLibraryPass : public PassInfoMixin<InlineBitcodeLibraryPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_IPO_LBC_H

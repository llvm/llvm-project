//===- NextSiliconIRBuiltins.h - Convert unsupported instructions to NextSilicon
// IR builtins ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------===//
//
// NextSiliconIRBuiltins pass converts unsupported code into the code based on
// the NextSilicon IR builtins.
//
//===------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_NEXTSILICONIRBUILTINS_H
#define LLVM_TRANSFORMS_UTILS_NEXTSILICONIRBUILTINS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Module;
class ModulePass;

class NextSiliconIRBuiltinsPass
    : public PassInfoMixin<NextSiliconIRBuiltinsPass> {
public:
  NextSiliconIRBuiltinsPass() {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_NEXTSILICONIRBUILTINS_H
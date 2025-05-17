//=== AlwaysSpecializer.h - implementation of always_specialize -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_IPO_ALWAYSSPECIALIZER_H
#define LLVM_TRANSFORMS_IPO_ALWAYSSPECIALIZER_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Module;
class ModulePass;

class AlwaysSpecializerPass : public PassInfoMixin<AlwaysSpecializerPass> {
public:
  AlwaysSpecializerPass();
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

ModulePass *createAlwaysSpecializerPass();

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_ALWAYSSPECIALIZER_H

//===- HLSLFinalizeLinkage.h - Finalize linkage of functions --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// HLSLFinalizeLinkage pass updates the linkage of functions to make sure only
/// shader entry points and exported functions are visible from the module (have
/// program linkage). All other functions and variables will be updated to have
/// internal linkage.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_HLSL_HLSLFINALIZELINKAGE_H
#define LLVM_TRANSFORMS_HLSL_HLSLFINALIZELINKAGE_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class HLSLFinalizeLinkage : public PassInfoMixin<HLSLFinalizeLinkage> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
  static bool isRequired() { return true; }
};

class HLSLFinalizeLinkageLegacy : public ModulePass {
public:
  HLSLFinalizeLinkageLegacy() : ModulePass(ID) {}
  bool runOnModule(Module &M) override;

  static char ID; // Pass identification.
};
} // namespace llvm

#endif // LLVM_TRANSFORMS_HLSL_HLSLFINALIZELINKAGE_H

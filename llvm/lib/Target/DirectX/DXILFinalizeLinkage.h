//===- DXILFinalizeLinkage.h - Finalize linkage of functions --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// DXILFinalizeLinkage pass updates the linkage of functions to make sure only
/// shader entry points and exported functions are visible from the module (have
/// program linkage). All other functions will be updated to have internal
/// linkage.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_DIRECTX_DXILFINALIZELINKAGE_H
#define LLVM_TARGET_DIRECTX_DXILFINALIZELINKAGE_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class DXILFinalizeLinkage : public PassInfoMixin<DXILFinalizeLinkage> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
  static bool isRequired() { return true; }
};

class DXILFinalizeLinkageLegacy : public ModulePass {
public:
  DXILFinalizeLinkageLegacy() : ModulePass(ID) {}
  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  static char ID; // Pass identification.
};
} // namespace llvm

#endif // LLVM_TARGET_DIRECTX_DXILFINALIZELINKAGE_H

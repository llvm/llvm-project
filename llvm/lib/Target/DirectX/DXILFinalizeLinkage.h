//===- DXILFinalizeLinkage.h - Finalize linkage of functions --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
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

  static char ID; // Pass identification.

  bool runOnModule(Module &M) override;
};
} // namespace llvm

#endif // LLVM_TARGET_DIRECTX_DXILFINALIZELINKAGE_H

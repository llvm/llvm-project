//===- DXILTranslateMetadata.h - Pass to emit DXIL metadata -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_DIRECTX_DXILTRANSLATEMETADATA_H
#define LLVM_TARGET_DIRECTX_DXILTRANSLATEMETADATA_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

/// A pass that transforms LLVM Metadata in the module to it's DXIL equivalent,
/// then emits all recognized DXIL Metadata
class DXILTranslateMetadata : public PassInfoMixin<DXILTranslateMetadata> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

/// Wrapper pass for the legacy pass manager.
///
/// This is required because the passes that will depend on this are codegen
/// passes which run through the legacy pass manager.
class DXILTranslateMetadataLegacy : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit DXILTranslateMetadataLegacy() : ModulePass(ID) {}

  StringRef getPassName() const override { return "DXIL Translate Metadata"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnModule(Module &M) override;
};

} // namespace llvm

#endif // LLVM_TARGET_DIRECTX_DXILTRANSLATEMETADATA_H

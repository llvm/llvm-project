//===- DXILTranslateMetadata.cpp - Pass to emit DXIL metadata ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
//===----------------------------------------------------------------------===//

#include "DXILMetadata.h"
#include "DXILResource.h"
#include "DXILResourceAnalysis.h"
#include "DirectX.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

using namespace llvm;

namespace {
class DXILTranslateMetadata : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit DXILTranslateMetadata() : ModulePass(ID) {}

  StringRef getPassName() const override { return "DXIL Metadata Emit"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<DXILResourceWrapper>();
  }

  bool runOnModule(Module &M) override;
};

} // namespace

bool DXILTranslateMetadata::runOnModule(Module &M) {

  dxil::ValidatorVersionMD ValVerMD(M);
  if (ValVerMD.isEmpty())
    ValVerMD.update(VersionTuple(1, 0));
  dxil::createShaderModelMD(M);

  dxil::Resources &Res = getAnalysis<DXILResourceWrapper>().getDXILResource();
  Res.write(M);
  return false;
}

char DXILTranslateMetadata::ID = 0;

ModulePass *llvm::createDXILTranslateMetadataPass() {
  return new DXILTranslateMetadata();
}

INITIALIZE_PASS_BEGIN(DXILTranslateMetadata, "dxil-metadata-emit",
                      "DXIL Metadata Emit", false, false)
INITIALIZE_PASS_DEPENDENCY(DXILResourceWrapper)
INITIALIZE_PASS_END(DXILTranslateMetadata, "dxil-metadata-emit",
                    "DXIL Metadata Emit", false, false)

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

  bool runOnModule(Module &M) override;
};

} // namespace

bool DXILTranslateMetadata::runOnModule(Module &M) {

  dxil::ValidatorVersionMD ValVerMD(M);
  if (ValVerMD.isEmpty())
    ValVerMD.update(VersionTuple(1, 0));
  return false;
}

char DXILTranslateMetadata::ID = 0;

ModulePass *llvm::createDXILTranslateMetadataPass() {
  return new DXILTranslateMetadata();
}

INITIALIZE_PASS(DXILTranslateMetadata, "dxil-metadata-emit",
                "DXIL Metadata Emit", false, false)
